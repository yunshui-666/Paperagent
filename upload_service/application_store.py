"""业务数据访问层。在任务存储基础上管理文档生命周期、会话、幂等问答和结果快照。"""

import json
import time
from uuid import uuid4

from .database import TaskStore, encode
from .errors import ProjectError


class ApplicationStore(TaskStore):
    """扩展任务数据库，提供文档生命周期、会话和每轮答案的持久化访问。"""
    def initialize(self):
        """建立会话表、问答表和约束索引，复用 TaskStore 的文档及任务表。"""
        super().initialize()
        with self._connection() as connection:
            # request_id 在同一会话中唯一；部分唯一索引约束同时运行的问答数量。
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, collection_name TEXT NOT NULL,
                    options_json TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turns (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    request_id TEXT NOT NULL, question TEXT NOT NULL, status TEXT NOT NULL,
                    answer TEXT, sources_json TEXT, metrics_json TEXT, error TEXT,
                    created_at REAL NOT NULL, finished_at REAL,
                    UNIQUE(session_id,request_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS active_session_turn ON turns(session_id) WHERE status='running';
                CREATE INDEX IF NOT EXISTS session_turn_order ON turns(session_id,created_at,id);
            """)

    def document(self, document_id, include_deleted=False):
        """按 ID 获取文档，按调用需要决定是否包含已经删除的记录。"""
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        if row is None or (not include_deleted and row["lifecycle"] == "deleted"):
            raise ProjectError("文档不存在", 404)
        return dict(row)

    def document_tasks(self, document_id):
        """返回文档关联的全部任务，最新任务排在前面。"""
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM tasks WHERE document_id=? ORDER BY created_at DESC,id DESC",
                                      (document_id,)).fetchall()
        return [dict(row) for row in rows]

    def documents(self):
        """列出文档，并为每个集合选取最新任务生成前端索引状态摘要。"""
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM documents WHERE lifecycle!='deleted' ORDER BY created_at DESC").fetchall()
        result = []
        for row in rows:
            document = dict(row)
            indexes = {}
            for task in self.document_tasks(document["id"]):
                # 任务已按时间降序排列，每个集合首次遇到的任务代表最新索引状态。
                name = task["collection_name"]
                if name not in indexes:
                    data = json.loads(task["result_json"] or "null") or {}
                    indexes[name] = {"collection_name": name, "scheme": task["scheme"],
                                     "status": task["status"], "task_id": task["id"],
                                     "chunk_count": data.get("chunk_count", 0),
                                     "error": json.loads(task["error_json"] or "null")}
            document["indexes"] = list(indexes.values())
            result.append(document)
        return result

    def availability(self, collection_name):
        """返回服务管理的全部文档和该集合中已发布的文档。旧索引可继续读取。"""
        with self._connection() as connection:
            documents = connection.execute("SELECT id,lifecycle FROM documents").fetchall()
            tasks = connection.execute("SELECT document_id,status FROM tasks WHERE collection_name=? ORDER BY created_at DESC,id DESC",
                                       (collection_name,)).fetchall()
        latest = {}
        for task in tasks:
            # setdefault 只记录首条状态，较旧的成功任务不能覆盖正在重建的最新任务。
            latest.setdefault(task["document_id"], task["status"])
        known = {row["id"] for row in documents}
        # known 包含所有登记过的身份，ready 只包含活动且最新入库完成的文档。
        # ScopedStore 据此同时过滤 Dense 和 BM25 的可见语料。
        ready = {row["id"] for row in documents
                 if row["lifecycle"] == "active" and latest.get(row["id"]) == "completed"}
        return known, ready

    def begin_delete(self, document_id):
        """在事务中确认文档没有活动任务，再标记 deleting，使检索范围即时收敛。"""
        with self._transaction() as connection:
            row = connection.execute("SELECT lifecycle FROM documents WHERE id=?", (document_id,)).fetchone()
            if row is None:
                raise ProjectError("文档不存在", 404)
            active = connection.execute("SELECT 1 FROM tasks WHERE document_id=? AND status IN ('queued','running')",
                                        (document_id,)).fetchone()
            if active:
                raise ProjectError("该文档正在处理，任务结束后可删除")
            if row["lifecycle"] != "deleted":
                # 先从可检索范围撤下，再执行可能跨 Chroma 和文件系统的清理操作。
                connection.execute("UPDATE documents SET lifecycle='deleting' WHERE id=?", (document_id,))

    def finish_delete(self, document_id):
        """向量与文件清理成功后写入 deleted 状态及删除时间。"""
        with self._transaction() as connection:
            connection.execute("UPDATE documents SET lifecycle='deleted',deleted_at=? WHERE id=?", (time.time(), document_id))

    def unfinished_deletions(self):
        """返回服务重启后需要继续清理的 deleting 文档 ID。"""
        with self._connection() as connection:
            return [row[0] for row in connection.execute("SELECT id FROM documents WHERE lifecycle='deleting'")]

    def create_session(self, title, collection_name, options):
        """以独立 UUID 保存会话标题、集合及完整检索选项快照。"""
        session_id = uuid4().hex
        now = time.time()
        with self._transaction() as connection:
            connection.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?)",
                               (session_id, title, collection_name, encode(options), now, now))
        return self.session(session_id)

    def session(self, session_id):
        """读取会话并将 options_json 解码为选项字典，缺失时返回业务 404。"""
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise ProjectError("会话不存在", 404)
        result = dict(row)
        result["options"] = json.loads(result.pop("options_json"))
        return result

    def sessions(self):
        """按更新时间降序查询会话摘要，供侧栏展示。"""
        with self._connection() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT id,title,collection_name,created_at,updated_at FROM sessions ORDER BY updated_at DESC")]

    def rename_session(self, session_id, title):
        """验证会话存在后更新标题与更新时间，并返回更新后的会话。"""
        self.session(session_id)
        with self._transaction() as connection:
            connection.execute("UPDATE sessions SET title=?,updated_at=? WHERE id=?", (title, time.time(), session_id))
        return self.session(session_id)

    def delete_session(self, session_id):
        """删除会话，由外键级联清理其问答业务记录。"""
        with self._transaction() as connection:
            connection.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    def reserve_turn(self, session_id, request_id, question):
        """按 session_id 与 request_id 预留幂等轮次，并限制每会话只有一个 running 轮次。
        成功重复请求返回原结果，问题冲突或其他活动请求返回业务冲突错误。
        """
        self.session(session_id)
        with self._transaction() as connection:
            existing = connection.execute("SELECT * FROM turns WHERE session_id=? AND request_id=?",
                                          (session_id, request_id)).fetchone()
            if existing:
                # 同一请求重试必须对应相同问题，已完成时直接复用答案及证据快照。
                if existing["question"] != question:
                    raise ProjectError("同一个 request_id 已用于另一个问题")
                if existing["status"] == "completed":
                    return self._turn(existing)
            active = connection.execute("SELECT request_id FROM turns WHERE session_id=? AND status='running'",
                                        (session_id,)).fetchone()
            if active and active[0] != request_id:
                raise ProjectError("该会话有待恢复的问答，请先重试上一条问题")
            now = time.time()
            if existing:
                # failed 轮次沿用 turn_id，图中的稳定消息 ID 也会沿用这个身份。
                connection.execute("UPDATE turns SET status='running',error=NULL,finished_at=NULL WHERE id=?", (existing["id"],))
                turn_id = existing["id"]
            else:
                turn_id = uuid4().hex
                connection.execute("INSERT INTO turns(id,session_id,request_id,question,status,created_at) VALUES (?,?,?,?,'running',?)",
                                   (turn_id, session_id, request_id, question, now))
            connection.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id))
            row = connection.execute("SELECT * FROM turns WHERE id=?", (turn_id,)).fetchone()
        return self._turn(row)

    @staticmethod
    def _turn(row):
        """将数据库行转换为业务字典，并恢复 sources 和 metrics 的 JSON 字段。"""
        result = dict(row)
        result["sources"] = json.loads(result.pop("sources_json") or "[]")
        result["metrics"] = json.loads(result.pop("metrics_json") or "{}")
        return result

    def finish_turn(self, turn_id, answer, sources, metrics):
        """在一个事务中保存成功答案、证据快照、指标及完成时间。"""
        with self._transaction() as connection:
            connection.execute("UPDATE turns SET status='completed',answer=?,sources_json=?,metrics_json=?,error=NULL,finished_at=? WHERE id=?",
                               (answer, encode(sources), encode(metrics), time.time(), turn_id))

    def order_turn_for_execution(self, turn_id):
        # 旧失败轮次重试时按实际执行顺序展示，和图中的消息顺序保持一致。
        """把失败旧请求的重试轮次调整到实际执行顺序，以便和消息历史对齐。"""
        with self._transaction() as connection:
            connection.execute("UPDATE turns SET created_at=? WHERE id=? AND status='running'",
                               (time.time(), turn_id))

    def fail_turn(self, turn_id, error):
        """记录失败原因及完成时间，使相同请求后续可以显式重试。"""
        with self._transaction() as connection:
            connection.execute("UPDATE turns SET status='failed',error=?,finished_at=? WHERE id=?",
                               (error, time.time(), turn_id))

    def turns(self, session_id):
        """按实际业务顺序返回会话中的全部成功与失败轮次。"""
        with self._connection() as connection:
            return [self._turn(row) for row in connection.execute(
                "SELECT * FROM turns WHERE session_id=? ORDER BY created_at,id", (session_id,))]

    def interrupted_turns(self):
        """列出上次中断遗留的 running 轮次，供检查点恢复流程核对。"""
        with self._connection() as connection:
            return [self._turn(row) for row in connection.execute("SELECT * FROM turns WHERE status='running'")]
