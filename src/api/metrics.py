"""Service metrics for /predict, exposed at GET /metrics.

Plain-text Prometheus format, no extra dependency. A pure ASGI middleware
measures every /predict call from the outside, so the endpoint code stays
untouched.
"""

import json
import threading
import time
from collections import Counter
from datetime import datetime, timezone

from fastapi.responses import PlainTextResponse

# Upper bounds in seconds. Latency is time spent inside the service;
# age is "now minus the reading's own timestamp" when the service receives it.
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
AGE_BUCKETS = (0.5, 1, 2, 5, 10, 30, 60, 120)


class Histogram:
    """Counts observations per bucket, plus their sum and number."""

    def __init__(self, bounds):
        self.bounds = tuple(bounds)
        self.counts = [0] * len(self.bounds)
        self.sum = 0.0
        self.count = 0

    def observe(self, value):
        self.sum += value
        self.count += 1
        for i, bound in enumerate(self.bounds):
            if value <= bound:
                self.counts[i] += 1
                break

    def render(self, name, help_text):
        lines = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]
        running = 0  # Prometheus buckets are cumulative
        for bound, n in zip(self.bounds, self.counts):
            running += n
            lines.append(f'{name}_bucket{{le="{bound:g}"}} {running}')
        lines.append(f'{name}_bucket{{le="+Inf"}} {self.count}')
        lines.append(f"{name}_sum {self.sum:.6f}")
        lines.append(f"{name}_count {self.count}")
        return lines


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.requests = Counter()  # by HTTP status code
        self.results = Counter()  # by scorer status (ok, warming_up, ...)
        self.anomalies = 0
        self.latency = Histogram(LATENCY_BUCKETS)
        self.age = Histogram(AGE_BUCKETS)

    def record(self, http_status, duration, answer=None, age=None):
        with self._lock:
            self.requests[str(http_status)] += 1
            self.latency.observe(duration)
            if http_status == 200 and answer is not None:
                self.results[str(answer.get("status", "unknown"))] += 1
                if answer.get("is_anomaly"):
                    self.anomalies += 1
                if age is not None:
                    self.age.observe(age)

    def render(self):
        with self._lock:
            lines = [
                "# HELP predict_requests_total Calls to /predict by HTTP status.",
                "# TYPE predict_requests_total counter",
            ]
            for code, n in sorted(self.requests.items()):
                lines.append(f'predict_requests_total{{code="{code}"}} {n}')
            lines += [
                "# HELP predict_results_total Accepted readings by scorer status.",
                "# TYPE predict_results_total counter",
            ]
            for status, n in sorted(self.results.items()):
                lines.append(f'predict_results_total{{status="{status}"}} {n}')
            lines += [
                "# HELP predict_anomalies_total Readings flagged as anomalies.",
                "# TYPE predict_anomalies_total counter",
                f"predict_anomalies_total {self.anomalies}",
            ]
            lines += self.latency.render(
                "predict_latency_seconds", "Time spent inside the service per /predict call."
            )
            lines += self.age.render(
                "reading_age_seconds", "Age of a reading when the service received it."
            )
            return "\n".join(lines) + "\n"


def _loads(raw):
    """Parse bytes as a JSON object, or return None. Never raises."""
    try:
        value = json.loads(bytes(raw))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _reading_age(request_body, now):
    """Seconds between the reading's timestamp and `now`, or None if unusable."""
    payload = _loads(request_body)
    if payload is None:
        return None
    raw = payload.get("timestamp")
    if not isinstance(raw, str):
        return None
    try:
        sent = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if sent.tzinfo is None:
        return None
    return max(0.0, (now - sent).total_seconds())


class MetricsMiddleware:
    """Pure ASGI middleware: times one path and records what went in and out."""

    def __init__(self, app, metrics, path="/predict"):
        self.app = app
        self.metrics = metrics
        self.path = path

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != self.path:
            await self.app(scope, receive, send)
            return

        received_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        request_body = bytearray()
        response_body = bytearray()
        status = 500  # what we record if the app crashes before answering

        async def recv():
            message = await receive()
            if message["type"] == "http.request":
                request_body.extend(message.get("body", b""))
            return message

        async def snd(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            elif message["type"] == "http.response.body":
                response_body.extend(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, recv, snd)
        finally:
            self.metrics.record(
                status,
                time.perf_counter() - started,
                answer=_loads(response_body),
                age=_reading_age(request_body, received_at),
            )


def install_metrics(app, metrics=None):
    """Add the middleware and the GET /metrics route to a FastAPI app."""
    metrics = metrics or Metrics()
    app.add_middleware(MetricsMiddleware, metrics=metrics)

    async def read_metrics():
        return PlainTextResponse(metrics.render())

    app.add_api_route("/metrics", read_metrics, methods=["GET"])
    app.state.metrics = metrics
    return metrics