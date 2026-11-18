"""使用 ONNX Runtime 联合编码问题和候选文本，输出相关性 logits。"""

# 本地模型适配层。问题和候选正文联合编码，原始 logit 交由 retrieval.py 排序与路由。

from pathlib import Path
from threading import Lock

import numpy as np


MODEL_REPOSITORY = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
# 固定 revision，使本地模型文件版本可复现。量化 ONNX 用于 CPU 推理。
MODEL_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
MODEL_FILE = "onnx/model_quint8_avx2.onnx"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "data" / "models" / "mmarco-minilm"


class OnnxCrossEncoder:
    """读取本地权重，对 (问题, 候选正文) 联合编码并输出原始 logit。"""
    def __init__(self, model_dir=None, *, batch_size=8, max_length=512, cpu_threads=4):
        # 构造阶段只保存配置，真正需要打分时才加载权重和 tokenizer。
        """保存模型目录、批量、序列长度和 CPU 线程配置，首次 score 时加载权重。"""
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in (batch_size, max_length, cpu_threads)):
            raise ValueError("Cross-Encoder 的批量、长度和线程参数需要为整数")
        if batch_size < 1 or not 8 <= max_length <= 512 or cpu_threads < 1:
            raise ValueError("批量、线程数必须为正整数，当前模型 max_length 范围为 8 到 512")
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self.repository = MODEL_REPOSITORY
        self.revision = MODEL_REVISION
        self.batch_size = batch_size
        self.max_length = max_length
        self.cpu_threads = cpu_threads
        self._session = None
        self._tokenizer = None
        self._query_tokenizer = None
        self._lock = Lock()
        # 同一个实例可能被多个会话共享，用锁保护初始化和 tokenizer 状态。

    def _load(self):
        # 权重成功加载后直接复用 ONNX session，避免每轮重新初始化。
        """加载固定版本 tokenizer 和量化 ONNX 权重，缺少文件时按固定 revision 获取。"""
        if self._session is not None:
            return
        import onnxruntime as ort
        from tokenizers import Tokenizer

        files = ("config.json", "tokenizer.json", "tokenizer_config.json", MODEL_FILE)
        missing = [name for name in files if not (self.model_dir / name).is_file()]
        if missing:
            # 本地缺文件才下载固定版本。已有完整权重时可以离线运行。
            from huggingface_hub import hf_hub_download
            for name in missing:
                hf_hub_download(MODEL_REPOSITORY, name, revision=MODEL_REVISION,
                                local_dir=self.model_dir)
        tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        # 用两个独立 tokenizer：一个测量问题长度，一个对文本对截断和补齐。
        query_tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        query_tokenizer.no_truncation()
        query_tokenizer.no_padding()
        tokenizer.enable_truncation(max_length=self.max_length, strategy="only_second")
        # only_second 将候选正文作为截断对象，问题先在 score() 内限长。
        pad_id = tokenizer.token_to_id("<pad>")
        if pad_id is None:
            raise ValueError("Cross-Encoder tokenizer 缺少 <pad> token")
        tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")
        # 同一批输入补齐为相同长度，才能堆叠为 [batch, sequence] 数组。
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.cpu_threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(self.model_dir / MODEL_FILE), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        self._tokenizer = tokenizer
        self._query_tokenizer = query_tokenizer
        self._session = session
        # 所有初始化步骤成功后才设置 _session，失败时下次可以重新尝试加载。

    def score(self, question: str, documents: list[str]) -> list[float]:
        """按输入顺序返回一个候选对应一个分数，排序由 retrieval.rerank 完成。"""
        if not documents:
            return []
        scores = []
        with self._lock:
            self._load()
            query_ids = self._query_tokenizer.encode(question, add_special_tokens=False).ids
            # 为正文预留大部分模型窗口，问题最多使用 128 个 tokenizer Token。
            query_budget = min(128, self.max_length // 3)
            if len(query_ids) > query_budget:
                # 只截断重排器内部的问题副本，原始提问仍由 RAG 状态保存。
                question = self._query_tokenizer.decode(query_ids[:query_budget], skip_special_tokens=True)
            for start in range(0, len(documents), self.batch_size):
                # 一个 batch 内每个候选都与相同问题组成独立的文本对。
                pairs = [(question, text) for text in documents[start:start + self.batch_size]]
                encoded = self._tokenizer.encode_batch(pairs)
                arrays = {
                    # token ID、注意力掩码和句段 ID 均使用 ONNX 需要的 int64。
                    "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
                    "attention_mask": np.asarray([item.attention_mask for item in encoded], dtype=np.int64),
                    "token_type_ids": np.asarray([item.type_ids for item in encoded], dtype=np.int64)
                }
                inputs = {item.name: arrays[item.name] for item in self._session.get_inputs()}
                # 只传实际 ONNX 模型声明的输入，兼容不需要 token_type_ids 的导出。
                logits = np.asarray(self._session.run(None, inputs)[0])
                if logits.shape != (len(pairs), 1) or not np.isfinite(logits).all():
                    # 当前模型每对文本输出一个 logit，任何形状或数值异常都明确报错。
                    raise RuntimeError("Cross-Encoder 返回了异常分数或不支持的输出形状")
                scores.extend(float(value) for value in logits[:, 0])
                # [:, 0] 将 [batch, 1] 转成一维，保持与原始候选一一对应。
        return scores
