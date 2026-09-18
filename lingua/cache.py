"""Translation cache: an in-memory LRU in front of an optional SQLite file.

Two reasons this earns its place, one obvious and one not:

* Speed. A repeated sentence costs a full model pass — measured at 245 ms on
  the RTX 3050 for a short Hindi sentence, every single time.
* Rate limits. Google's keyless endpoints throttle datacentre IPs far harder
  than home connections, which is the main fragility of the Render
  deployment. Every cache hit is a call not made.

SQLite is used rather than a pile of JSON files because it is in the standard
library, survives restarts, and handles two workers touching it at once. The
memory layer in front means the common case never touches disk.

Nothing here is required: with ``LINGUA_CACHE=0`` the whole thing becomes a
no-op and the router behaves exactly as before.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_TTL = 60 * 60 * 24 * 30      # 30 days
_DEFAULT_MEMORY_ENTRIES = 512


def _truthy(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def cache_path() -> Path:
    raw = os.getenv("LINGUA_CACHE_DIR")
    base = Path(raw) if raw else Path.home() / ".cache" / "lingua"
    return base / "translations.sqlite3"


class TranslationCache:
    """Keyed on the whole request, because tier and provider change the answer."""

    def __init__(self, enabled: bool | None = None, memory_entries: int | None = None,
                 ttl: int | None = None, path: Path | None = None):
        self.enabled = _truthy(os.getenv("LINGUA_CACHE"), True) \
            if enabled is None else enabled
        self.ttl = ttl if ttl is not None else int(
            os.getenv("LINGUA_CACHE_TTL", str(_DEFAULT_TTL)))
        self.max_memory = memory_entries if memory_entries is not None else int(
            os.getenv("LINGUA_CACHE_SIZE", str(_DEFAULT_MEMORY_ENTRIES)))

        self._memory: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._lock = threading.RLock()
        self._db: sqlite3.Connection | None = None
        self._path = path or cache_path()
        self.hits = 0
        self.misses = 0
        self.writes = 0

        if self.enabled and _truthy(os.getenv("LINGUA_CACHE_DISK"), True):
            self._open_db()

    # --- storage -----------------------------------------------------------
    def _open_db(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False: reads happen from the asyncio thread
            # pool, and every access here is already guarded by self._lock.
            self._db = sqlite3.connect(self._path, check_same_thread=False)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS entries ("
                " key TEXT PRIMARY KEY,"
                " value TEXT NOT NULL,"
                " created REAL NOT NULL)")
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS entries_created ON entries(created)")
            self._db.commit()
        except sqlite3.Error as exc:
            # A broken cache must never break translation.
            log.warning("disk cache unavailable (%s); using memory only", exc)
            self._db = None

    @staticmethod
    def key(namespace: str, **parts) -> str:
        payload = json.dumps({"ns": namespace, **parts}, sort_keys=True,
                             ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # --- api ---------------------------------------------------------------
    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        now = time.time()
        with self._lock:
            entry = self._memory.get(key)
            if entry is not None:
                created, value = entry
                if self.ttl <= 0 or now - created < self.ttl:
                    self._memory.move_to_end(key)
                    self.hits += 1
                    return value
                del self._memory[key]

            if self._db is not None:
                try:
                    row = self._db.execute(
                        "SELECT value, created FROM entries WHERE key = ?",
                        (key,)).fetchone()
                except sqlite3.Error as exc:
                    log.debug("cache read failed: %s", exc)
                    row = None
                if row is not None:
                    raw, created = row
                    if self.ttl <= 0 or now - created < self.ttl:
                        try:
                            value = json.loads(raw)
                        except json.JSONDecodeError:
                            value = None
                        if value is not None:
                            self._remember(key, created, value)
                            self.hits += 1
                            return value
                    self._forget(key)

            self.misses += 1
            return None

    def set(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        now = time.time()
        with self._lock:
            self._remember(key, now, value)
            self.writes += 1
            if self._db is not None:
                try:
                    self._db.execute(
                        "INSERT OR REPLACE INTO entries (key, value, created) "
                        "VALUES (?, ?, ?)",
                        (key, json.dumps(value, ensure_ascii=False), now))
                    self._db.commit()
                except sqlite3.Error as exc:
                    log.debug("cache write failed: %s", exc)

    def _remember(self, key: str, created: float, value: dict) -> None:
        self._memory[key] = (created, value)
        self._memory.move_to_end(key)
        while len(self._memory) > self.max_memory:
            self._memory.popitem(last=False)

    def _forget(self, key: str) -> None:
        self._memory.pop(key, None)
        if self._db is not None:
            try:
                self._db.execute("DELETE FROM entries WHERE key = ?", (key,))
                self._db.commit()
            except sqlite3.Error:
                pass

    def clear(self) -> int:
        with self._lock:
            removed = len(self._memory)
            self._memory.clear()
            if self._db is not None:
                try:
                    cur = self._db.execute("DELETE FROM entries")
                    self._db.commit()
                    removed = max(removed, cur.rowcount)
                except sqlite3.Error:
                    pass
            return removed

    def stats(self) -> dict:
        with self._lock:
            stored = None
            if self._db is not None:
                try:
                    stored = self._db.execute(
                        "SELECT COUNT(*) FROM entries").fetchone()[0]
                except sqlite3.Error:
                    stored = None
            total = self.hits + self.misses
            return {
                "enabled": self.enabled,
                "memory_entries": len(self._memory),
                "memory_limit": self.max_memory,
                "disk_entries": stored,
                "disk_path": str(self._path) if self._db is not None else None,
                "hits": self.hits,
                "misses": self.misses,
                "writes": self.writes,
                "hit_rate": round(self.hits / total, 3) if total else None,
                "ttl_seconds": self.ttl,
            }

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                try:
                    self._db.close()
                except sqlite3.Error:
                    pass
                self._db = None
