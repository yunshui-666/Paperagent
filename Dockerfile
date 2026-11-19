FROM python:3.14-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PAPERAGENT_SERVICE_ROOT=/data/service \
    PAPERAGENT_INGEST_ROOT=/data/ingestion \
    PAPERAGENT_CHROMA_DIR=/data/chroma \
    PAPERAGENT_RERANKER_MODEL_DIR=/data/models/mmarco-minilm

WORKDIR /app

# OCR 和 ONNX 使用 CPU。应用文件只读，运行数据集中写入持久化卷。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 paperagent \
    && useradd --uid 10001 --gid paperagent --no-create-home paperagent \
    && mkdir -p /data \
    && chown paperagent:paperagent /data

COPY requirements*.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY main.py README.md ./
COPY pdfprocess ./pdfprocess
COPY upload_service ./upload_service
COPY web ./web
COPY docs ./docs

USER paperagent
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).read()"

# 单个 Uvicorn 服务进程协调 SQLite 队列与会话锁，解析任务由内部进程池处理。
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]
