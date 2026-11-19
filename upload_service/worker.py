"""任务线程领取 SQLite 队列，独立进程执行 PDF 解析。"""

# 后台执行层。线程从 SQLite 领取任务，PDF 解析通过 spawn 子进程执行，进度回写业务队列。

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import logging
import multiprocessing
from threading import Event, Lock, Thread

from pdfprocess.ingest_config import PipelineConfig
from pdfprocess.ingest_common import ProcessingBusyError
from pdfprocess.pipelines import get_pipeline


logger = logging.getLogger(__name__)


class ParserPool:
    """按需创建的 PDF 解析进程池，使用 spawn 隔离解析库状态。"""
    def __init__(self, process_count):
        """保存解析进程数，并为首次创建或失败重建准备互斥锁。"""
        self.process_count = process_count
        self._pool = None
        self._lock = Lock()

    def parse(self, parser, path, config):
        """向子进程提交页解析函数并等待结果，进程池损坏时清理引用以便下次重建。"""
        with self._lock:
            if self._pool is None:
                # spawn 使用新的 Python 解释器，任务函数和配置通过 pickle 跨进程传递。
                self._pool = ProcessPoolExecutor(
                    max_workers=self.process_count,
                    mp_context=multiprocessing.get_context("spawn")
                )
            pool = self._pool
        try:
            return pool.submit(parser, path, config=config).result()
        except BrokenProcessPool as exc:
            with self._lock:
                if self._pool is pool:
                    # 只清理本次失败的池实例，避免覆盖其他线程已创建的新进程池。
                    self._pool = None
                    pool.shutdown(wait=False, cancel_futures=True)
            raise RuntimeError("PDF 解析子进程异常退出，进程池已重置，可重新上传重试") from exc

    def stop(self):
        """取出并关闭当前进程池，等待已执行任务完成。"""
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


def run_pipeline(task, progress, *, parser_pool):
    """按任务方案号选择同命名入口，恢复配置并传入原文件身份和进度回调。"""
    module = get_pipeline(task["scheme"])
    # 三套方案保持相同函数命名约定和参数结构，调度器只需读取方案号。
    function = getattr(module, f"ingest_pdf{task['scheme']}")
    return function(
        task["path"], config=PipelineConfig(**task["config"]),
        source_name=task["document_name"], progress_callback=progress,
        parse_executor=parser_pool, expected_paper_id=task["document_id"]
    )


class WorkerPool:
    """由若干后台线程组成的 SQLite 任务消费者，负责领取任务和提交最终状态。"""
    def __init__(self, store, runner, worker_count):
        """保存任务存储、执行函数、线程数量及停止和唤醒事件。"""
        self.store = store
        self.runner = runner
        self.worker_count = worker_count
        self._stop = Event()
        self._wake = Event()
        # Event 仅用于减少空队列轮询延迟，任务状态由 SQLite 持久化。
        self._threads = []

    def start(self):
        """创建具名后台线程，每个线程独立进入领取循环。"""
        for index in range(self.worker_count):
            thread = Thread(target=self._loop, name=f"paperagent-ingest-{index}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def notify(self):
        """通知等待线程尽快查询新任务，数据库仍是任务状态的持久化来源。"""
        self._wake.set()

    def stop(self):
        """设置停止事件并唤醒线程，等待当前执行中的任务收尾。"""
        self._stop.set()
        self._wake.set()
        for thread in self._threads:
            thread.join()

    def alive_count(self):
        """统计实际存活的工作线程数量，供健康检查观察。"""
        return sum(thread.is_alive() for thread in self._threads)

    def _loop(self):
        """持续领取到期任务，空队列时短暂等待通知，领取异常记录日志后继续循环。"""
        while not self._stop.is_set():
            try:
                task = self.store.claim()
                if task is not None:
                    # claim 已原子转为 running，当前线程负责该任务的后续阶段与结束状态。
                    self._run_task(task)
                    continue
            except Exception:
                logger.exception("领取或更新上传任务失败")
            self._wake.wait(0.2)
            self._wake.clear()

    def _run_task(self, task):
        """运行一个任务，回写进度与结果；处理锁忙时延迟重排，其余异常保存失败原因。"""
        stage = "starting"

        def progress(name, details=None):
            """先保存阶段事件，再更新当前阶段变量，便于异常报告定位失败位置。"""
            nonlocal stage
            self.store.set_stage(task["id"], name, details)
            stage = name

        try:
            result = self.runner(task, progress)
            if not isinstance(result, dict) or result.get("status") != "completed":
                raise RuntimeError("入库函数未返回有效的完成结果")
            self.store.finish(task["id"], result=result)
        except ProcessingBusyError as exc:
            # 文件/集合处理锁忙时退避，ready_at 会阻止工作线程立即重领同一个任务。
            delay = min(30, 2 ** min(task["attempts"], 5))
            self.store.requeue(task["id"], {
                "reason": "processing_busy", "message": str(exc), "retry_after_seconds": delay
            }, delay)
        except Exception as exc:
            # 将最后成功记录的阶段一起保存，便于定位解析、Embedding 或保存环节。
            logger.exception("入库任务失败 %s", task["id"])
            self.store.finish(task["id"], error={
                "type": type(exc).__name__, "message": str(exc), "stage": stage
            })
