"""按问题序列调用公开 API，记录多轮延迟、Token 与证据快照。"""

# 问答性能实验入口。每个场景创建新会话，按原定问题顺序采集答案、证据、延迟与模型用量。

import argparse
import json
from pathlib import Path
import time
from uuid import uuid4

import httpx

from pdfprocess.ingest_common import write_json
from .benchmark import distribution


def run_chat_benchmark(url, scenarios, *, collection, mode="hybrid_rerank", dense_k=20,
                       keyword_k=20, top_k=10, rewrite_query=True, repeat=1, timeout=180):
    """按场景和重复次数创建会话，依次提交问题，汇总成功率、端到端延迟和四类模型用量。"""
    if not isinstance(scenarios, list) or not scenarios or repeat < 1:
        raise ValueError("需要非空场景列表和正整数 repeat")
    for scenario in scenarios:
        if (not isinstance(scenario, dict) or not isinstance(scenario.get("questions"), list)
                or not scenario["questions"] or any(not isinstance(q, str) or not q.strip() for q in scenario["questions"])):
            raise ValueError("每个场景需要 questions 非空字符串列表")
    records = []
    with httpx.Client(base_url=url, timeout=timeout) as client:
        for run in range(repeat):
            for scenario in scenarios:
                # 每次重复创建新会话，避免上一次实验历史进入本次模型上下文。
                response = client.post("/sessions", json={
                    "title": f"实验 {scenario.get('name', 'scenario')} {run + 1}", "collection_name": collection,
                    "document_ids": scenario.get("document_ids", []), "mode": mode,
                    "dense_k": dense_k, "keyword_k": keyword_k, "top_k": top_k,
                    "rewrite_query": rewrite_query
                })
                response.raise_for_status()
                session = response.json()
                for index, question in enumerate(scenario["questions"], start=1):
                    # 顺序提交追问，让后续问题能够使用同场景前面成功轮次的历史。
                    started = time.perf_counter()
                    response = client.post("/chat", json={"thread_id": session["id"],
                        "request_id": uuid4().hex, "question": question})
                    elapsed = time.perf_counter() - started
                    record = {"run": run + 1, "scenario": scenario.get("name"), "turn": index,
                              "session_id": session["id"], "question": question,
                              "client_seconds": elapsed, "http_status": response.status_code}
                    if response.is_success:
                        record["result"] = response.json()
                    else:
                        record["error"] = response.text
                    records.append(record)
                    if not response.is_success:
                        # 当前会话失败后记录错误并结束该序列，保留后续场景独立执行。
                        break
    successful = [row for row in records if "result" in row]
    # 以下 Token 合计以成功轮次保存的 usage 为准，失败与 SDK 重试费用参照服务商日志。
    tokens = {}
    for row in successful:
        # 摘要可能分批调用，使用列表保存每个成功请求返回的真实 Token 用量。
        # 评估总成本时累计摘要、改写、工具调用生成与答案四部分。
        metrics = row["result"]["metrics"]
        for stage in ("rewrite_usage", "generation_usage", "summary_usage", "tool_selection_usage", "paper_agent_usage"):
            usages = metrics.get(stage, {})
            if isinstance(usages, dict):
                # 摘要是多次调用列表，其余阶段通常是一个 usage 字典，统一为列表后求和。
                usages = [usages]
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                name = f"{stage}_{key}"
                tokens[name] = tokens.get(name, 0) + sum(usage.get(key, 0) for usage in usages)
    return {"config": {"collection": collection, "mode": mode, "dense_k": dense_k,
                       "keyword_k": keyword_k, "top_k": top_k, "rewrite_query": rewrite_query,
                       "repeat": repeat},
            "attempted_turns": len(records), "completed_turns": len(successful),
            "success_rate": len(successful) / len(records) if records else 0,
            "client_latency_seconds": distribution([row["client_seconds"] for row in successful]),
            "tokens": tokens, "records": records}


def main(argv=None):
    """读取问题场景 JSON，通过公开 HTTP 接口执行实验并保存报告。"""
    parser = argparse.ArgumentParser(description="PaperAgent 多轮问答实验，实验会话保留在界面供复查")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--mode", choices=("dense", "bm25", "hybrid", "hybrid_rerank"), default="hybrid_rerank")
    parser.add_argument("--dense-k", type=int, default=20)
    parser.add_argument("--keyword-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-rewrite", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    scenarios = json.loads(args.scenarios.read_text(encoding="utf-8-sig"))
    report = run_chat_benchmark(args.url, scenarios, collection=args.collection, mode=args.mode,
        dense_k=args.dense_k, keyword_k=args.keyword_k, top_k=args.top_k,
        rewrite_query=not args.no_rewrite, repeat=args.repeat)
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("attempted_turns", "completed_turns", "success_rate", "client_latency_seconds", "tokens")}, ensure_ascii=False, indent=2))
    return int(report["success_rate"] != 1)


if __name__ == "__main__":
    raise SystemExit(main())
