"""Logging with a per-request ID, so one request can be followed across modules."""
import contextvars
import logging
import os
from logging.handlers import RotatingFileHandler

request_id_var = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True


def setup_logging(level="INFO", log_file=None):
    """Configure the root logger once (safe to call repeatedly)."""
    root = logging.getLogger()
    if getattr(root, "_chess_api_configured", False):
        return
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(RotatingFileHandler(
            log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(fmt)
        handler.addFilter(RequestIdFilter())  # handler-level, so it covers every logger
    root.handlers = handlers
    root.setLevel(level)
    root._chess_api_configured = True
