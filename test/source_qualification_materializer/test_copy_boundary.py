"""Real ordinary-copy failure boundaries; no qualification verifier is mocked."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SUBJECT = ROOT / "tools/actual_b_chain/materialize_source_qualification.py"
ORACLE = HERE / "copy_boundary_oracles_v1.json"
ORACLE_SHA = "3d04346734aca03ba1a4b254da8f8aa955e36e3f0569fdcb61f3ba129c789eb9"
OBSERVATIONS = []

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

class CopyBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if digest(ORACLE.read_bytes()) != ORACLE_SHA:
            raise RuntimeError("Independent copy-boundary oracle changed")
        spec = importlib.util.spec_from_file_location("independent_sourceq_copy_boundary_subject", SUBJECT)
        cls.m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.m
        spec.loader.exec_module(cls.m)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sourceq-copy-boundary-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "ordinary-original.bin"
        self.payload = b"unqualified ordinary evidence block\n" * 10000
        self.source.write_bytes(self.payload)
        self.sha = digest(self.payload)
        positive = self.root / "positive.bin"
        result = self.m.copy_verified_file(self.source, positive, expected_sha256=self.sha,
                                           expected_bytes=len(self.payload))
        self.assertEqual(result["sha256"], self.sha)
        self.assertEqual(positive.read_bytes(), self.payload)

    def code_is_n20(self, call):
        with self.assertRaises(self.m.SourceQualificationError) as error:
            call()
        self.assertEqual(error.exception.code, "partial_or_late_publication")

    def test_real_midcopy_write_failure_never_installs_target(self):
        target = self.root / "write-failure.bin"
        real_open = Path.open
        observed = dict(write_calls=0, actual_first_block_bytes=0)
        class FaultingOutput:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.stream.close()
            def write(self, raw):
                observed["write_calls"] += 1
                if observed["write_calls"] > 1:
                    raise OSError("independent write fault after actual fsynced partial block")
                written = self.stream.write(raw)
                self.stream.flush()
                os.fsync(self.stream.fileno())
                observed["actual_first_block_bytes"] += written
                return written
            def __getattr__(self, name):
                return getattr(self.stream, name)
        def opening(path, mode="r", *args, **kwargs):
            stream = real_open(path, mode, *args, **kwargs)
            if path.parent == self.root and mode == "xb":
                return FaultingOutput(stream)
            return stream
        with mock.patch.object(Path, "open", opening):
            self.code_is_n20(lambda: self.m.copy_verified_file(
                self.source, target, expected_sha256=self.sha, expected_bytes=len(self.payload)))
        observed.update(case="C01", final_target_exists=target.exists(),
                        original_source_unchanged=self.source.read_bytes() == self.payload)
        OBSERVATIONS.append(dict(observed))
        self.assertGreaterEqual(observed["write_calls"], 2)
        self.assertGreaterEqual(observed["actual_first_block_bytes"], 65536)
        self.assertFalse(target.exists())
        self.assertEqual(self.source.read_bytes(), self.payload)

    def test_postrename_late_clock_never_leaves_accepted_target(self):
        target = self.root / "late-target.bin"
        real_rename = Path.rename
        previous_deadline = self.m._DEADLINE
        observed = dict(actual_rename_observed=False)
        def renaming(path, destination):
            result = real_rename(path, destination)
            if Path(destination) == target:
                observed["actual_rename_observed"] = True
                info = target.stat()
                observed["actual_installed_bytes"] = info.st_size
                self.m._DEADLINE = time.monotonic_ns() - 1
            return result
        try:
            with mock.patch.object(Path, "rename", renaming):
                self.code_is_n20(lambda: self.m.copy_verified_file(
                    self.source, target, expected_sha256=self.sha, expected_bytes=len(self.payload)))
        finally:
            self.m._DEADLINE = previous_deadline
        observed.update(case="C02", final_target_exists=target.exists(),
                        original_source_unchanged=self.source.read_bytes() == self.payload)
        OBSERVATIONS.append(dict(observed))
        self.assertIs(observed["actual_rename_observed"], True)
        self.assertEqual(observed["actual_installed_bytes"], len(self.payload))
        self.assertFalse(target.exists())
        self.assertEqual(self.source.read_bytes(), self.payload)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    output = Path(args.output_directory)
    if not __debug__ or not output.is_absolute() or output.exists() or output.is_symlink() or ROOT in output.parents:
        raise RuntimeError("Unoptimized test and fresh external absolute output required")
    paths = [Path(__file__), ORACLE, SUBJECT, SUBJECT.with_name("sourceq_full_scale_verifier.py")]
    before = {str(path): digest(path.read_bytes()) for path in paths}
    output.mkdir()
    started = time.monotonic_ns()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CopyBoundary))
    after = {str(path): digest(path.read_bytes()) for path in paths}
    record = dict(schema="sourceq_copy_boundary_independent_tests_v1",
                  complete=result.wasSuccessful() and not result.skipped and result.testsRun == 2 and before == after,
                  unittest_methods=result.testsRun, expected_methods=2, observations=OBSERVATIONS,
                  failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                  source_before=before, source_after=after, source_unchanged=before == after,
                  started_ns=started, finished_ns=time.monotonic_ns(),
                  verification_or_authority_mocked=False, full_qualification_fixture_issued=False,
                  actual_numerical_or_owner_launches_requested=0,
                  source_qualified=False, software_qualified=False, scientific_qualified=False)
    (output / "RESULT.json").write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    raise SystemExit(0 if record["complete"] else 1)

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
