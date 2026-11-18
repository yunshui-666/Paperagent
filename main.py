"""PaperAgent 本机应用入口。"""

# 启动入口。main() 处理命令行、环境配置和 Uvicorn 单进程启动。
import argparse
import multiprocessing
from pathlib import Path

from dotenv import load_dotenv
import uvicorn


def main(argv=None):
    """解析启动参数并运行单进程服务，返回 CLI 退出码。"""
    # Windows 的 spawn 子进程需要此入口保护，以便正确初始化解析进程池。
    multiprocessing.freeze_support()
    # 环境配置在显式启动时加载，其他模块可以直接导入 main() 做本地测试。
    load_dotenv(Path(__file__).resolve().parent / ".env")
    parser = argparse.ArgumentParser(description="启动 PaperAgent 网页与 API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    # 业务依赖与 SQLite 会话锁按一个服务进程组织，PDF 解析另有独立进程池。
    uvicorn.run("upload_service.app:app", host=args.host, port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
