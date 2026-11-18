"""SQLite 保存文件、任务队列及阶段事件。每次操作独立连接。"""

# 任务数据访问层。每次操作使用独立连接，写事务统一提交或回滚，任务状态和事件同步更新。

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time
from uuid import uuid4
from .errors import ProjectError


def encode(value):
    """将配置、结果或错误编码成 UTF-8 友好的 JSON 文本，特殊值按字符串保存。"""
    return json.dumps(value, ensure_ascii=False, default=str)


class TaskStore:
    """SQLite 持久化任务队列，负责文档登记、任务状态、阶段事件和恢复。"""
    def __init__(self, path, journal_mode="WAL"):
        """保存数据库路径与日志模式，数据库结构在 initialize 时建立。"""
        if journal_mode not in {"WAL", "DELETE"}:
            raise ValueError("不支持的 SQLite journal_mode")
        self.path = Path(path)
        self.journal_mode = journal_mode

    @contextmanager
    def _connection(self):
        """为一次访问打开独立 SQLite 连接，开启外键和 FULL 同步，结束时关闭连接。"""
        connection = sqlite3.connect(self.path, timeout=5)
        # Row 同时支持字段名与下标访问，业务层可统一转换成 dict。
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        # FULL 同步用于任务状态持久化，进程异常退出后仍可按已提交状态恢复。
        connection.execute("PRAGMA synchronous=FULL")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self):
        """使用 BEGIN IMMEDIATE 组织写事务，成功提交，任何异常时回滚。"""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            # 读取待领取任务之前先获得写锁，让“查询 + 更新”成为同一个原子操作。
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def initialize(self):
        """创建任务相关表与索引，并为旧数据库补齐 ready_at 和文档生命周期字段。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(f"PRAGMA journal_mode={self.journal_mode}")
            # active_task_key 是部分唯一索引，仅约束 queued/running 的任务键。
            # 相同配置的历史完成任务可以再次提交，并由入库缓存决定是否复用结果。
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    page_count INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id),
                    upload_name TEXT NOT NULL,
                    job_key TEXT NOT NULL,
                    scheme INTEGER NOT NULL,
                    collection_name TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed')),
                    stage TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    queued_at REAL NOT NULL,
                    ready_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    result_json TEXT,
                    error_json TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS active_task_key
                ON tasks(job_key) WHERE status IN ('queued','running');
                CREATE INDEX IF NOT EXISTS queued_tasks ON tasks(status, queued_at);
                CREATE TABLE IF NOT EXISTS task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES tasks(id),
                    stage TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    details_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS task_event_lookup ON task_events(task_id,id);
            """)
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(tasks)")}
            # 老数据库逐字段升级，已有任务和文档身份继续用于恢复与证据回查。
            if "ready_at" not in columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN ready_at REAL NOT NULL DEFAULT 0")
                connection.execute("UPDATE tasks SET ready_at=queued_at")
                connection.commit()
            document_columns = {row["name"] for row in connection.execute("PRAGMA table_info(documents)")}
            if "lifecycle" not in document_columns:
                connection.execute("ALTER TABLE documents ADD COLUMN lifecycle TEXT NOT NULL DEFAULT 'active'")
            if "deleted_at" not in document_columns:
                connection.execute("ALTER TABLE documents ADD COLUMN deleted_at REAL")
            connection.commit()

    def enqueue(self, document, scheme, collection_name, pipeline_config, job_key):
        """原子登记文档与任务，合并同任务键的活动请求，并追加 queued 事件。"""
        now = time.time()
        with self._transaction() as connection:
            # 文档用内容哈希去重，原文件名保留为显示信息。
            connection.execute("""
                INSERT INTO documents(id,name,path,size_bytes,page_count,created_at)
                VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING
            """, (document["id"], document["name"], document["path"],
                  document["size_bytes"], document["page_count"], now))
            lifecycle = connection.execute("SELECT lifecycle FROM documents WHERE id=?", (document["id"],)).fetchone()[0]
            # 删除进行中与上传互斥；已删除内容再次上传时恢复活动生命周期。
            if lifecycle == "deleting":
                raise ProjectError("文件正在删除，请稍后重新上传")
            if lifecycle == "deleted":
                connection.execute("UPDATE documents SET lifecycle='active',deleted_at=NULL,name=?,path=?,created_at=? WHERE id=?",
                                   (document["name"], document["path"], now, document["id"]))
            existing = connection.execute(
                "SELECT id,status FROM tasks WHERE job_key=? AND status IN ('queued','running')", (job_key,)
            ).fetchone()
            if existing is not None:
                # 客户端仍拿到同一 task_id，后台只需要执行一次活动任务。
                return {"task_id": existing["id"], "status": existing["status"], "reused_task": True}
            task_id = uuid4().hex
            connection.execute("""
                INSERT INTO tasks(id,document_id,upload_name,job_key,scheme,collection_name,
                                  config_json,status,stage,created_at,queued_at,ready_at)
                VALUES (?,?,?,?,?,?,?,'queued','queued',?,?,?)
            """, (task_id, document["id"], document["name"], job_key, scheme,
                  collection_name, encode(pipeline_config), now, now, now))
            connection.execute(
                "INSERT INTO task_events(task_id,stage,timestamp,details_json) VALUES (?,'queued',?,?)",
                (task_id, now, encode({"reason": "upload_accepted"}))
            )
        return {"task_id": task_id, "status": "queued", "reused_task": False}

    def recover_interrupted(self):
        """把中断遗留的 running 任务重新排队，保留任务身份并记录恢复事件。"""
        now = time.time()
        with self._transaction() as connection:
            rows = connection.execute("SELECT id FROM tasks WHERE status='running'").fetchall()
            # 任务结果按稳定 Chunk ID 写入，恢复后再次执行可以补齐中断的写入。
            for row in rows:
                connection.execute("""
                    UPDATE tasks SET status='queued',stage='queued',queued_at=?,ready_at=?,started_at=NULL,
                                     finished_at=NULL,details_json='{}' WHERE id=?
                """, (now, now, row["id"]))
                connection.execute(
                    "INSERT INTO task_events(task_id,stage,timestamp,details_json) VALUES (?,'queued',?,?)",
                    (row["id"], now, encode({"reason": "service_restart"}))
                )
        return len(rows)

    def claim(self):
        """在写事务内领取最早到期的 queued 任务，标记 running 并递增尝试次数。"""
        now = time.time()
        with self._transaction() as connection:
            # ready_at 支持退避，queued_at 决定同一批到期任务的领取顺序。
            row = connection.execute("""
                SELECT t.*,d.path,d.name AS document_name FROM tasks t
                JOIN documents d ON d.id=t.document_id
                WHERE t.status='queued' AND t.ready_at<=? ORDER BY t.queued_at,t.id LIMIT 1
            """, (now,)).fetchone()
            if row is None:
                return None
            connection.execute("""
                UPDATE tasks SET status='running',stage='starting',started_at=?,attempts=attempts+1 WHERE id=?
            """, (now, row["id"]))
            connection.execute(
                "INSERT INTO task_events(task_id,stage,timestamp,details_json) VALUES (?,'starting',?,'{}')",
                (row["id"], now)
            )
            task = dict(row)
            task["config"] = json.loads(task.pop("config_json"))
            task["attempts"] += 1
            # 返回值来自 UPDATE 前读到的行，因此这里同步反映本次领取后的状态。
            task.update({"status": "running", "stage": "starting", "started_at": now})
            return task

    def requeue(self, task_id, details, delay_seconds):
        """将持锁冲突等可恢复任务延迟入队，写入 ready_at 避免立即争抢。"""
        now = time.time()
        with self._transaction() as connection:
            updated = connection.execute("""
                UPDATE tasks SET status='queued',stage='queued',queued_at=?,ready_at=?,started_at=NULL,
                                 details_json=? WHERE id=? AND status='running'
            """, (now, now + delay_seconds, encode(details), task_id)).rowcount
            if updated != 1:
                # 状态前提被破坏时立即报错，避免覆盖已经完成或被其他操作更新的任务。
                raise RuntimeError("任务当前状态不允许重新入队")
            connection.execute(
                "INSERT INTO task_events(task_id,stage,timestamp,details_json) VALUES (?,'queued',?,?)",
                (task_id, now, encode(details))
            )

    def set_stage(self, task_id, stage, details=None):
        """更新运行中任务的阶段及细节，并同步追加阶段事件。"""
        with self._transaction() as connection:
            updated = connection.execute("""
                UPDATE tasks SET stage=?,details_json=? WHERE id=? AND status='running'
            """, (stage, encode(details or {}), task_id)).rowcount
            if updated != 1:
                raise RuntimeError("任务当前状态不允许更新处理阶段")
            connection.execute(
                "INSERT INTO task_events(task_id,stage,timestamp,details_json) VALUES (?,?,?,?)",
                (task_id, stage, time.time(), encode(details or {}))
            )

    def finish(self, task_id, *, result=None, error=None):
        """将运行中任务原子转为 completed 或 failed，保存结果或错误及完成事件。"""
        if error is None and not isinstance(result, dict):
            raise ValueError("成功任务需要提供字典形式的 result")
        status = "completed"
        if error is not None:
            status = "failed"
        with self._transaction() as connection:
            # 状态、结果/错误和完成事件共享一个写事务，查询端可以得到一致快照。
            updated = connection.execute("""
                UPDATE tasks SET status=?,stage=?,finished_at=?,result_json=?,error_json=?
                WHERE id=? AND status='running'
            """, (status, status, time.time(), encode(result), encode(error), task_id)).rowcount
            if updated != 1:
                raise RuntimeError("任务当前状态不允许结束")
            connection.execute(
                "INSERT INTO task_events(task_id,stage,timestamp,details_json) VALUES (?,?,?,?)",
                (task_id, status, time.time(), encode(error or {"skipped": result.get("skipped", False)}))
            )

    def get(self, task_id):
        """查询任务及事件，解码 JSON、计算等待处理耗时，并把时间戳转成 UTC 字符串。"""
        with self._connection() as connection:
            row = connection.execute("""
                SELECT t.*,d.name AS document_name,d.size_bytes,d.page_count
                FROM tasks t JOIN documents d ON d.id=t.document_id WHERE t.id=?
            """, (task_id,)).fetchone()
            if row is None:
                return None
            events = connection.execute(
                "SELECT stage,timestamp,details_json FROM task_events WHERE task_id=? ORDER BY id", (task_id,)
            ).fetchall()
        task = dict(row)
        # 内部 JSON 列转成接口字段，配置快照和幂等键供执行器使用。
        for field, target in (("details_json", "details"), ("result_json", "result"), ("error_json", "error")):
            value = task.pop(field)
            task[target] = json.loads(value) if value is not None else None
        task.pop("config_json")
        task.pop("job_key")
        now = time.time()
        start = task["started_at"]
        finish = task["finished_at"] or now
        # perf_counter 用于进程内测量；数据库保存 wall clock，支持跨重启计算任务耗时。
        task["queue_wait_seconds"] = max(0.0, (start or now) - task["queued_at"])
        task["processing_seconds"] = max(0.0, finish - start) if start is not None else 0.0
        task["total_seconds"] = max(0.0, finish - task["created_at"])
        for field in ("created_at", "queued_at", "ready_at", "started_at", "finished_at"):
            # 对外时间采用 UTC ISO 字符串，前端可以按用户时区显示。
            value = task[field]
            if value is not None:
                task[field] = datetime.fromtimestamp(value, timezone.utc).isoformat()
        task["events"] = [{"stage": event["stage"],
                           "timestamp": datetime.fromtimestamp(event["timestamp"], timezone.utc).isoformat(),
                           "details": json.loads(event["details_json"])} for event in events]
        return task

    def counts(self):
        """统计四种任务状态数量，缺失状态补零供健康检查使用。"""
        with self._connection() as connection:
            rows = connection.execute("SELECT status,count(*) AS count FROM tasks GROUP BY status").fetchall()
        counts = {name: 0 for name in ("queued", "running", "completed", "failed")}
        counts.update({row["status"]: row["count"] for row in rows})
        return counts
