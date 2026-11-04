"""Keeps the base-model cache from filling the container disk.

A worker keeps every base model it has trained on, and together they are bigger than the disk. Without this,
a worker that has trained several models fails on its next download, then fails every job it picks up after
that: it is free again within seconds, so it drains the queue.
"""
from __future__ import annotations

import glob
import os
import shutil
from typing import Callable, Iterable

GB = 1024 ** 3
# The most a job's own files can need: its base model, plus latents and checkpoints in the job folder.
DEFAULT_MIN_FREE_GB = 120.0
# Never ask for more than this share of a small disk, which could otherwise empty the cache for nothing.
MAX_FREE_SHARE = 0.6


def repo_dir(hf_home: str, repo: str) -> str:
    return os.path.join(hf_home, "hub", "models--" + repo.replace("/", "--"))


def _size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                st = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            total += st.st_size
    return total


def _remove_partial_downloads(hf_home: str) -> int:
    """A worker runs one job at a time, so nothing is still writing these. A failed download can leave many GB."""
    freed = 0
    for path in glob.glob(os.path.join(hf_home, "hub", "*", "blobs", "*.incomplete")):
        try:
            size = os.path.getsize(path)
            os.remove(path)
            freed += size
        except OSError:
            continue
    return freed


def _entries(hf_home: str, keep: set[str]) -> list[tuple[float, str]]:
    """What may be removed, oldest first."""
    found = []
    candidates = glob.glob(os.path.join(hf_home, "hub", "models--*")) + glob.glob(os.path.join(hf_home, "hub", "datasets--*"))
    candidates.append(os.path.join(hf_home, "xet"))
    for path in candidates:
        if path in keep or not os.path.isdir(path):
            continue
        try:
            found.append((os.path.getmtime(path), path))
        except OSError:
            continue
    return sorted(found)


def make_room(
    hf_home: str,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    keep: Iterable[str | None] = (),
    log: Callable[[str], None] = print,
) -> dict:
    """Removes what was used longest ago until the disk has room for a job. Marks this job's models as just used."""
    keep_dirs = set()
    for repo in keep:
        # A local path is not a cached repo, and a repo that isn't cached yet has nothing to keep.
        if repo and not os.path.isabs(repo):
            path = repo_dir(hf_home, repo)
            keep_dirs.add(path)
            if os.path.isdir(path):
                os.utime(path)

    total = shutil.disk_usage(hf_home).total
    target = min(min_free_gb * GB, total * MAX_FREE_SHARE)
    start_free = shutil.disk_usage(hf_home).free
    removed = []
    partial = _remove_partial_downloads(hf_home)
    for _, path in _entries(hf_home, keep_dirs):
        if shutil.disk_usage(hf_home).free >= target:
            break
        size = _size(path)
        shutil.rmtree(path, ignore_errors=True)
        removed.append({"path": os.path.basename(path), "gb": round(size / GB, 1)})
    free = shutil.disk_usage(hf_home).free
    log(
        f"disk: {free / GB:.0f} GB free of {total / GB:.0f} GB (was {start_free / GB:.0f}, want {target / GB:.0f}); "
        f"partial downloads {partial / GB:.1f} GB; removed {removed or 'nothing'}"
    )
    if free < target:
        log("disk: still short after clearing the cache; this job's own model may not fit")
    return {"free_gb": round(free / GB, 1), "removed": removed, "partial_gb": round(partial / GB, 1)}
