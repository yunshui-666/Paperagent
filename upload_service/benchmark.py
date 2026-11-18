"""用真实 HTTP 请求记录上传、排队、完成延迟与采样资源指标。"""

# 上传性能实验入口。客户端确认耗时、任务完成耗时和服务端资源采样采用各自定义记录。

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
import time

import httpx

from pdfprocess.ingest_common import write_json


def percentile(values, percent):
    """对有序样本做线性插值，返回指定百分位，空样本返回 None。"""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    # 百分位落在两个样本之间时进行插值，单样本时上下界相同。
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    """输出样本数、均值和 P50/P95/P99，供多个耗时指标复用。"""
    return {"count": len(values), "mean": statistics.mean(values) if values else None,
            "p50": percentile(values, 50), "p95": percentile(values, 95), "p99": percentile(values, 99)}


def upload_batch(client, url, paths, scheme):
    """发出一次批量上传请求，记录输入大小、HTTP 结果与接收阶段耗时。"""
    started = time.perf_counter()
    record = {"files": [str(path) for path in paths], "started": started,
              "input_bytes": sum(path.stat().st_size for path in paths)}
    try:
        with ExitStack() as stack:
            # 一次批量上传打开多份文件，ExitStack 在请求结束后统一关闭所有流。
            files = [("files", (path.name, stack.enter_context(path.open("rb")), "application/pdf"))
                     for path in paths]
            response = client.post(url + "/documents", files=files, data={"scheme": str(scheme)})
        record.update({"http_status": response.status_code, "response": response.json()})
    except Exception as exc:
        record.update({"http_status": None, "error": str(exc), "response": {"items": []}})
    record["accept_seconds"] = time.perf_counter() - started
    return record


def run_benchmark(url, paths, *, scheme=3, concurrency=1, batch_size=1, repeat=1,
                  timeout=1800, poll_interval=0.5, request_timeout=120):
    """并发上传固定工作负载，轮询任务及健康接口，汇总延迟、吞吐、缓存与资源采样。
    重复上传可能复用活动任务或解析缓存，报告分别记录对应数量。
    """
    if (any(isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (concurrency, batch_size, repeat))
            or any(not math.isfinite(value) or value <= 0 for value in (timeout, poll_interval, request_timeout))):
        raise ValueError("并发、批量、重复次数和各超时参数需要大于 0")
    paths = [Path(path).resolve() for path in paths]
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("请提供存在的文件路径")
    url = url.rstrip("/")
    workload = paths * repeat
    # 重复内容可能命中活动任务或入库缓存，报告会记录复用次数与新处理任务数。
    batches = [workload[start:start + batch_size] for start in range(0, len(workload), batch_size)]
    requests = []
    tasks = {}
    pending_tasks = set()
    observed_at = {}
    poll_errors = []
    samples = []
    with httpx.Client(timeout=request_timeout,
                      limits=httpx.Limits(max_connections=concurrency + 4)) as client:
        initial_response = client.get(url + "/health")
        initial_response.raise_for_status()
        initial_health = initial_response.json()
        samples.append(initial_health)
        started = time.perf_counter()
        deadline = started + timeout
        # 整个实验等待预算与单次 HTTP 超时分别控制，轮询期间持续采样服务资源。
        next_poll = started
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(upload_batch, client, url, batch, scheme) for batch in batches}
            while futures or pending_tasks:
                if time.perf_counter() >= deadline:
                    break
                if futures:
                    done, futures = wait(futures, timeout=max(0, next_poll - time.perf_counter()),
                                         return_when=FIRST_COMPLETED)
                    for future in done:
                        record = future.result()
                        requests.append(record)
                        for item in record["response"].get("items", []):
                            if item.get("accepted"):
                                task_id = item["task_id"]
                                if task_id not in observed_at:
                                    # 多次上传可能复用同一 task_id，任务完成状态按唯一身份观察。
                                    pending_tasks.add(task_id)
                else:
                    time.sleep(max(0, min(next_poll - time.perf_counter(), poll_interval)))
                if time.perf_counter() < next_poll:
                    continue
                for task_id in list(pending_tasks):
                    # 遍历副本，允许在发现完成时从 pending_tasks 中删除。
                    try:
                        response = client.get(url + f"/tasks/{task_id}")
                        response.raise_for_status()
                        task = response.json()
                        tasks[task_id] = task
                        if task["status"] in {"completed", "failed"}:
                            observed_at[task_id] = time.perf_counter()
                            pending_tasks.remove(task_id)
                    except Exception as exc:
                        poll_errors.append({"task_id": task_id, "error": str(exc)})
                try:
                    response = client.get(url + "/health")
                    response.raise_for_status()
                    samples.append(response.json())
                except Exception as exc:
                    poll_errors.append({"task_id": None, "error": str(exc)})
                next_poll = time.perf_counter() + poll_interval
            # 到达总等待预算后，仍记录已经发出的 HTTP 请求结果。
            for future in futures:
                record = future.result()
                requests.append(record)
                for item in record["response"].get("items", []):
                    if item.get("accepted") and item["task_id"] not in observed_at:
                        pending_tasks.add(item["task_id"])
        elapsed = time.perf_counter() - started
    items = [item for record in requests for item in record["response"].get("items", [])]
    accepted_items = [item for item in items if item.get("accepted")]
    completed = [task for task in tasks.values() if task["status"] == "completed"]
    failed = [task for task in tasks.values() if task["status"] == "failed"]
    skipped = [task for task in completed if task["result"].get("skipped", False)]
    fresh = [task for task in completed if not task["result"].get("skipped", False)]
    e2e = [max(0, observed_at[item["task_id"]] - record["started"])
           # 客户端从发起上传到观察到完成的耗时包含上传、排队、处理及轮询间隔。
           for record in requests for item in record["response"].get("items", [])
           if item.get("accepted") and item["task_id"] in observed_at
           and tasks[item["task_id"]]["status"] == "completed"]
    upload_seconds = max(record["started"] + record["accept_seconds"] for record in requests) - started
    submitted_bytes = sum(record["input_bytes"] for record in requests)
    http_errors = sum(record["http_status"] != 202 for record in requests)
    unique_tasks = len({item["task_id"] for item in accepted_items})
    initial_process = initial_health.get("process", {})
    last_process = samples[-1].get("process", {})
    cpu_delta = None
    if initial_process.get("pid") == last_process.get("pid"):
        # 服务 PID 一致时累计 CPU 时间差才具有同一进程生命周期的可比性。
        delta = last_process.get("cpu_seconds", 0) - initial_process.get("cpu_seconds", 0)
        if delta >= 0:
            cpu_delta = delta
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {"url": url, "scheme": scheme, "concurrency": concurrency,
                          "batch_size": batch_size, "repeat": repeat, "poll_interval": poll_interval,
                          "timeout": timeout, "files": [str(path) for path in paths]},
        "server_settings": {key: initial_health.get(key) for key in (
            "worker_count", "parse_processes", "max_file_bytes", "sqlite_journal_mode", "runner_name")},
        "elapsed_seconds": elapsed, "http_requests": len(requests), "submitted_files": len(workload),
        "submitted_file_bytes": submitted_bytes, "upload_phase_seconds": upload_seconds,
        "payload_mib_per_second": submitted_bytes / (1024 * 1024) / upload_seconds if not http_errors else None,
        "accepted_files": len(accepted_items), "rejected_files": sum(not item.get("accepted") for item in items),
        "http_202_count": sum(record["http_status"] == 202 for record in requests),
        "http_error_count": http_errors,
        "unique_tasks": unique_tasks,
        "completed_tasks": len(completed), "failed_tasks": len(failed),
        "unfinished_tasks": len(pending_tasks), "cache_hit_tasks": len(skipped),
        "new_ingestion_tasks": len(fresh),
        "file_acceptance_rate": len(accepted_items) / len(workload),
        "task_completion_rate": len(completed) / unique_tasks if unique_tasks else None,
        "reused_active_task_responses": sum(item.get("reused_task", False) for item in accepted_items),
        "accept_latency_seconds": distribution([record["accept_seconds"] for record in requests]),
        "client_end_to_end_seconds": distribution(e2e),
        "queue_wait_seconds": distribution([task["queue_wait_seconds"] for task in completed + failed]),
        "processing_seconds": distribution([task["processing_seconds"] for task in completed]),
        "failed_processing_seconds": distribution([task["processing_seconds"] for task in failed]),
        "server_total_seconds": distribution([task["total_seconds"] for task in completed]),
        "completed_tasks_per_second": len(completed) / elapsed,
        "new_pages_per_second": sum(task["page_count"] for task in fresh) / elapsed,
        "sampled_peak_rss_bytes": max(sample.get("process", {}).get("rss_bytes", 0) for sample in samples),
        # RSS 峰值来自离散采样，采样间隔越大越可能漏掉短时内存尖峰。
        "cpu_seconds_delta": cpu_delta,
        "successful_task_model_usage": {name: sum(task["result"].get("metrics", {}).get(name, 0)
                                                   for task in completed)
                                        for name in ("ocr_calls", "vlm_calls", "input_tokens", "output_tokens")},
        "requests": requests, "tasks": tasks, "resource_samples": samples, "poll_errors": poll_errors
    }
    return report


def main(argv=None):
    """读取上传实验参数，运行基准、保存完整报告并按失败情况返回退出码。"""
    parser = argparse.ArgumentParser(description="PaperAgent 上传服务的 HTTP 与任务完成基准")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--pdfs", nargs="+", type=Path, required=True)
    parser.add_argument("--scheme", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(args.url, args.pdfs, scheme=args.scheme, concurrency=args.concurrency,
                               batch_size=args.batch_size, repeat=args.repeat, timeout=args.timeout,
                               poll_interval=args.poll_interval)
        write_json(args.output, report)
    except Exception as exc:
        print(f"基准执行失败：{exc}")
        return 1
    print(f"HTTP 请求 {report['http_requests']}，接受文件 {report['accepted_files']}，完成任务 {report['completed_tasks']}")
    print(f"失败任务 {report['failed_tasks']}，未完成任务 {report['unfinished_tasks']}")
    print(f"上传确认时延：{report['accept_latency_seconds']}")
    print(f"结果：{args.output.resolve()}")
    return int(bool(report["http_error_count"] or report["rejected_files"]
                    or report["failed_tasks"] or report["unfinished_tasks"]))


if __name__ == "__main__":
    raise SystemExit(main())
