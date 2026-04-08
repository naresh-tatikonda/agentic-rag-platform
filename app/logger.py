"""
app/logger.py
-------------
Structured JSON logger for every query request.

Why structured JSON instead of plain text logs?
  - Plain text: "Query processed in 2.3s"  → only humans can read it
  - Structured JSON: {"latency_ms": 2300}  → machines can query, aggregate, alert on it

These logs feed into:
  - CloudWatch Logs Insights (AWS)
  - Grafana Loki (self-hosted)
  - Any log aggregation system that ingests JSON

Every log line is one complete JSON object — one request = one log line.
This makes it trivial to grep, filter, and build dashboards on top of.
"""

import logging
import json
from datetime import datetime, timezone


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger configured to output structured JSON to stdout.
    Stdout is the correct output for containerized apps —
    Docker and Kubernetes capture stdout automatically.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # StreamHandler writes to stdout — correct for Docker/K8s environments
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    # Prevent log messages from bubbling up to root logger
    logger.propagate = False

    return logger


class JSONFormatter(logging.Formatter):
    """
    Custom log formatter that outputs every log record as a single JSON line.
    Standard logging formatters output plain text — this outputs machine-readable JSON.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Base log structure present on every log line
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge any extra fields passed via extra={} into the log entry
        # This is how we attach query-specific fields to each log line
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry)


def log_query_request(
    logger: logging.Logger,
    query: str,
    ticker: str,
    fiscal_year: int,
    intent: str,
    quality_score: float,
    retry_count: int,
    latency_ms: float,
    error: str = None,
) -> None:
    """
    Logs a structured record for every query request.
    Called from routes/query.py after the graph completes.

    One function call = one complete log line with all query metadata.
    This is the single place where query observability data is captured.
    """

    # Determine log level — errors get ERROR, successful queries get INFO
    level = logging.ERROR if error else logging.INFO

    # Build the structured fields specific to this query
    extra_fields = {
        "event": "query_processed",     # event type — useful for log filtering
        "query": query,
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "intent": intent,
        "quality_score": quality_score,
        "retry_count": retry_count,
        "latency_ms": round(latency_ms, 2),
        "status": "error" if error else "success",
        "error": error,                  # None on success, exception string on failure
    }

    # logging.Logger doesn't support extra fields natively in our JSON formatter
    # We attach them as a custom attribute on the LogRecord
    logger.log(
        level,
        "query_processed",
        extra={"extra_fields": extra_fields}
    )
