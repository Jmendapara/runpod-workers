"""Unit tests for the trainer's model cache cleanup.

Run: python3 tests/test_hf_cache.py   (no GPU, no ai-toolkit, no network)
"""
import os
import shutil
import sys
import tempfile
import unittest
from collections import namedtuple
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "trainers", "lora-trainer"))

import hf_cache  # noqa: E402

GB = hf_cache.GB
Usage = namedtuple("Usage", "total used free")


class FakeDisk:
    """A disk whose free space grows by what the real files on it add up to, where 1 byte stands for 1 GB."""

    def __init__(self, root, total_gb, other_used_gb=0):
        self.root, self.total_gb, self.other_used_gb = root, total_gb, other_used_gb

    def usage(self, _path):
        used = self.other_used_gb
        for folder, _, files in os.walk(self.root):
            used += sum(os.path.getsize(os.path.join(folder, f)) for f in files)
        return Usage(self.total_gb * GB, used * GB, (self.total_gb - used) * GB)


class MakeRoomTest(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.home)
        self.logs = []

    def repo(self, name, gb, age):
        path = hf_cache.repo_dir(self.home, name)
        os.makedirs(os.path.join(path, "blobs"))
        with open(os.path.join(path, "blobs", "weights"), "wb") as fh:
            fh.write(b"x" * gb)
        os.utime(path, (1000 - age, 1000 - age))
        return path

    def run_make_room(self, total_gb=200, min_free_gb=120, keep=(), other_used_gb=0):
        disk = FakeDisk(self.home, total_gb, other_used_gb)
        with mock.patch.object(hf_cache.shutil, "disk_usage", disk.usage), mock.patch.object(hf_cache, "_size", lambda p: sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs) * GB):
            return hf_cache.make_room(self.home, min_free_gb, keep, self.logs.append)

    def test_leaves_the_cache_alone_when_there_is_room(self):
        path = self.repo("org/a", 50, 10)
        result = self.run_make_room()
        self.assertEqual(result["removed"], [])
        self.assertTrue(os.path.isdir(path))

    def test_removes_what_was_used_longest_ago_and_stops_once_there_is_room(self):
        oldest = self.repo("org/oldest", 40, 30)
        middle = self.repo("org/middle", 40, 20)
        newest = self.repo("org/newest", 40, 10)
        result = self.run_make_room()  # 80 free of 200, 120 wanted
        self.assertFalse(os.path.exists(oldest))
        self.assertTrue(os.path.isdir(middle))
        self.assertTrue(os.path.isdir(newest))
        self.assertEqual(result["removed"], [{"path": "models--org--oldest", "gb": 40.0}])
        self.assertEqual(result["free_gb"], 120.0)

    def test_keeps_this_jobs_model_even_if_it_is_the_oldest(self):
        mine = self.repo("org/mine", 40, 99)
        other = self.repo("org/other", 80, 10)
        self.run_make_room(keep=["org/mine", None])
        self.assertTrue(os.path.isdir(mine))
        self.assertFalse(os.path.exists(other))

    def test_marks_this_jobs_model_as_just_used(self):
        mine = self.repo("org/mine", 10, 99)
        self.run_make_room(keep=["org/mine"])
        self.assertGreater(os.path.getmtime(mine), 1000)

    def test_a_local_path_or_uncached_repo_as_the_model_is_fine(self):
        self.run_make_room(keep=["/app/models/local", "org/not-cached-yet"])
        self.assertFalse(os.path.exists(hf_cache.repo_dir(self.home, "org/not-cached-yet")))

    def test_a_failed_download_is_cleared_first(self):
        path = self.repo("org/a", 90, 10)
        partial = os.path.join(path, "blobs", "abc.incomplete")
        with open(partial, "wb") as fh:
            fh.write(b"x" * 30)
        self.run_make_room()  # 80 free; the 30 of leftovers makes 110, still short, so the repo goes too
        self.assertFalse(os.path.exists(partial))
        self.assertFalse(os.path.exists(path))

    def test_a_failed_download_alone_can_be_enough(self):
        path = self.repo("org/a", 50, 10)
        partial = os.path.join(path, "blobs", "abc.incomplete")
        with open(partial, "wb") as fh:
            fh.write(b"x" * 60)
        result = self.run_make_room()  # 90 free; 150 once the leftovers go
        self.assertEqual(result["removed"], [])
        self.assertTrue(os.path.isdir(path))

    def test_the_xet_cache_can_go_too(self):
        xet = os.path.join(self.home, "xet")
        os.makedirs(xet)
        with open(os.path.join(xet, "chunks"), "wb") as fh:
            fh.write(b"x" * 100)
        os.utime(xet, (1, 1))
        self.run_make_room()
        self.assertFalse(os.path.exists(xet))

    def test_a_small_disk_is_not_emptied_for_a_target_it_cannot_reach(self):
        path = self.repo("org/a", 40, 10)
        result = self.run_make_room(total_gb=100)  # 120 wanted can't exist; 60 is enough, and 60 are free
        self.assertEqual(result["removed"], [])
        self.assertTrue(os.path.isdir(path))

    def test_says_so_when_it_cannot_make_enough_room(self):
        self.repo("org/mine", 150, 10)
        self.run_make_room(keep=["org/mine"])
        self.assertTrue(any("still short" in line for line in self.logs))

    def test_a_missing_cache_folder_is_fine(self):
        result = self.run_make_room()
        self.assertEqual(result["removed"], [])


if __name__ == "__main__":
    unittest.main()
