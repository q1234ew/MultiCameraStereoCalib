"""Application logging setup for desktop and packaged builds."""

from __future__ import annotations

import logging
import os
import platform
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PySide6.QtCore import QtMsgType, qInstallMessageHandler

from .runtime_paths import exe_dir, logs_dir

LOG_FILE_NAME = "multicamera.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 5


def setup_logging(level: int = logging.INFO) -> Path:
    """Configure rotating file logs and exception hooks.

    Returns the active log file path so the UI can show/open it.
    """
    log_path = logs_dir() / LOG_FILE_NAME
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-7s "
        "[%(threadName)s] %(name)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    if not getattr(sys, "frozen", False):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root.addHandler(console_handler)

    _install_exception_hooks()
    _install_qt_message_handler()
    _log_startup_context(log_path)
    return log_path


def _install_exception_hooks() -> None:
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger(__name__).critical(
            "Unhandled exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    def handle_thread_exception(args):
        logging.getLogger(__name__).critical(
            "Unhandled thread exception",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = handle_exception
    if hasattr(threading, "excepthook"):
        threading.excepthook = handle_thread_exception


def _install_qt_message_handler() -> None:
    qt_logger = logging.getLogger("qt")

    def handler(mode, context, message):
        if mode == QtMsgType.QtDebugMsg:
            level = logging.DEBUG
        elif mode == QtMsgType.QtInfoMsg:
            level = logging.INFO
        elif mode == QtMsgType.QtWarningMsg:
            level = logging.WARNING
        elif mode == QtMsgType.QtCriticalMsg:
            level = logging.ERROR
        elif mode == QtMsgType.QtFatalMsg:
            level = logging.CRITICAL
        else:
            level = logging.INFO
        file_name = getattr(context, "file", "") or ""
        line = getattr(context, "line", 0) or 0
        qt_logger.log(level, "%s (%s:%s)", message, file_name, line)

    qInstallMessageHandler(handler)


def _log_startup_context(log_path: Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info("==== MultiCamera startup ====")
    logger.info("log_file=%s", log_path)
    logger.info("frozen=%s executable=%s exe_dir=%s", getattr(sys, "frozen", False), sys.executable, exe_dir())
    logger.info("python=%s", sys.version.replace("\n", " "))
    logger.info("platform=%s", platform.platform())
    logger.info("cwd=%s", Path.cwd())
    logger.info("pid=%s", os.getpid())
