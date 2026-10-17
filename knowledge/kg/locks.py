"""Advisory inter-process locks for authoritative stores and heavyweight jobs.

Several agents may operate on one checkout. JSON stores are read, modified and
replaced as whole files, so two concurrent writers can silently lose each
other's records. ``writer()`` serializes those read-modify-write sections
across processes and across threads of one process; it is reentrant only for
the thread that holds it, so nested helpers do not deadlock.

``heavy_job()`` is non-blocking: at most one Docling inference process may run
on this machine at a time, and a second request is refused instead of queued.
Locks are advisory ``flock`` locks; they disappear with the holding process,
so an interrupted job cannot leave a stale lock behind.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import threading
import time
from pathlib import Path

from . import store

_HELD: dict[str, list] = {}  # lock path -> [fd, depth, holding thread ident]
_GUARD = threading.RLock()
_THREAD_LOCKS: dict[str, threading.Lock] = {}


class LockBusyError(TimeoutError):
    """The named lock is held by another process (or another thread of this one)."""

    def __init__(self, name: str, path: Path, holder: dict | None = None):
        self.name, self.path, self.holder = name, path, holder or {}
        super().__init__(f"{name} lock is held by another process: {self.holder or path}")


def lock_dir() -> Path:
    # Resolve at call time so tests that relocate KNOWLEDGE_ROOT stay isolated.
    return store.KNOWLEDGE_ROOT / ".work" / "locks"


def _holder(path: Path) -> dict:
    try:
        return json.loads(path.read_text() or "{}")
    except (OSError, ValueError):
        return {}


@contextlib.contextmanager
def exclusive(name: str, *, timeout: float | None = 600.0, poll: float = 0.2, purpose: str = ""):
    """Hold ``name`` exclusively across processes and threads.

    Reentrant for the holding thread only, so nested helpers do not deadlock;
    another thread of the same process waits (or is refused at ``timeout``)
    exactly as another process does. ``timeout=0`` refuses immediately when busy.
    """
    path = lock_dir() / f"{name}.lock"
    key = str(path)
    me = threading.get_ident()
    with _GUARD:
        held = _HELD.get(key)
        reentrant = held is not None and held[2] == me
        if reentrant:
            held[1] += 1
        thread_lock = _THREAD_LOCKS.setdefault(key, threading.Lock())
    if reentrant:
        try:
            yield path
        finally:
            with _GUARD:
                _HELD[key][1] -= 1
        return
    deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
    if timeout == 0:
        acquired = thread_lock.acquire(blocking=False)
    else:
        acquired = thread_lock.acquire(timeout=-1 if timeout is None else max(0.0, timeout))
    if not acquired:
        raise LockBusyError(name, path, {"holder": "another thread of this process"})
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise LockBusyError(name, path, _holder(path)) from None
                    time.sleep(poll)
            os.ftruncate(fd, 0)
            os.write(fd, json.dumps({"pid": os.getpid(), "thread": me, "purpose": purpose,
                                     "acquired_monotonic": time.monotonic()}).encode())
            with _GUARD:
                _HELD[key] = [fd, 1, me]
            try:
                yield path
            finally:
                with _GUARD:
                    _HELD.pop(key, None)
                os.ftruncate(fd, 0)
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    finally:
        thread_lock.release()


def writer(purpose: str = "", timeout: float | None = 1800.0):
    """Serialize writers of the authoritative JSON stores across processes."""
    return exclusive("store-writer", timeout=timeout, purpose=purpose)


def heavy_job(purpose: str = ""):
    """Refuse a second concurrent heavyweight (Docling) inference process."""
    return exclusive("docling-heavy", timeout=0, purpose=purpose)


def writes_stores(purpose: str):
    """Decorate a read-modify-write entry point with the store writer lock."""
    def decorate(function):
        import functools

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with writer(purpose=purpose):
                return function(*args, **kwargs)
        return wrapper
    return decorate
