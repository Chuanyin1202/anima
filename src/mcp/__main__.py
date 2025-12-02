"""Entry point for running MCP server as a module.

Usage:
    python -m src.mcp

Note:
    MCP 協議要求 stdout 只能輸出 JSON-RPC 訊息。
    所有日誌必須輸出到 stderr，否則會破壞協議。
"""

import logging
import sys

# 在導入任何模組前配置日誌，確保所有輸出到 stderr
logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stderr,
    format="%(name)s - %(levelname)s - %(message)s",
)

# 靜音特定庫的調試輸出
for logger_name in ["httpx", "httpcore", "openai", "mem0", "qdrant_client"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# 配置 structlog 輸出到 stderr（mem0_adapter 使用）
import structlog

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
)

from .server import run_server

if __name__ == "__main__":
    run_server()
