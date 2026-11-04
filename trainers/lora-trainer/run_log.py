"""A run's whole log, kept in object storage so it can be read after the worker is gone.

RunPod's own logs disappear with the worker, and a crash can end a run before it reports anything.
Uploading every couple of minutes means even a hard crash leaves nearly everything behind.
"""
import os
import threading
import time
from urllib.parse import parse_qs, urlparse

UPLOAD_EVERY_S = 120
MAX_BYTES = 20 * 1024 * 1024


def url_secrets(url):
    """What to hide for a URL with a token in it. Errors often quote only its path and query."""
    if not url:
        return []
    values = [v for vs in parse_qs(urlparse(url).query).values() for v in vs]
    return [url, *values]


class RunLog:
    def __init__(self, path, key, upload, secrets=(), every_s=UPLOAD_EVERY_S, max_bytes=MAX_BYTES, clock=time.time):
        self.path = path
        self.key = key
        self._upload = upload
        self._secrets = [s for s in secrets if s]
        self._every_s = every_s
        self._max_bytes = max_bytes
        self._clock = clock
        self._lock = threading.Lock()
        self._fh = open(path, "w", encoding="utf-8")
        self._written = 0
        self._dirty = False
        self._last_upload = None

    def write(self, line: str) -> None:
        for secret in self._secrets:
            line = line.replace(secret, "[redacted]")
        text = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(self._clock()))} {line}\n"
        with self._lock:
            if self._fh.closed or self._written >= self._max_bytes:
                return
            self._written += len(text.encode("utf-8"))
            if self._written >= self._max_bytes:
                text += "log truncated: too long\n"
            self._fh.write(text)
            self._fh.flush()
            self._dirty = True

    def upload(self, force: bool = False) -> None:
        """Never raises: logging must not be able to end a training."""
        now = self._clock()
        with self._lock:
            due = self._last_upload is None or now - self._last_upload >= self._every_s
            if not self._dirty or not (force or due):
                return
            self._dirty = False
            self._last_upload = now
        try:
            self._upload(self.path)
        except Exception as exc:
            print(f"lora-trainer - run log upload failed: {exc}", flush=True)
            with self._lock:
                self._dirty = True

    def close(self) -> None:
        with self._lock:
            self._fh.close()
        try:
            os.remove(self.path)
        except OSError:
            pass
