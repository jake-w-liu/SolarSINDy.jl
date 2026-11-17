"""Frozen event-predicate probes; no process observation or qualification issuance."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import stat
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ORACLE = HERE / "lifecycle_event_oracles_v1.json"
ORACLE_SHA = "59b1af3cd6cc1c1feeeb51f35b28f5656c1befeffbc8d2cee2e1a0e033b835d0"
SUBJECT = ROOT / "tools/actual_b_chain/materialize_source_qualification.py"
HELPER = ROOT / "tools/actual_b_chain/sourceq_full_scale_verifier.py"
ORIGINAL = ROOT / "test/matched_initial342/test_native_owner_lifecycle.py"
FIXED = {
    ORACLE: ORACLE_SHA,
    SUBJECT: "2a7d4ae5e93b7b3436ae95dc9ae05e153268fd37cc7b4620193c769c4ef5dd6f",
    HELPER: "6859faa129f73759f3772d4f8a790641a831a0bc519596a97f303c9437034011",
    ORIGINAL: "14fcd2b2331e4e105507cbe1ccdb8bd9c8ce2334fa0c6bb032b4349cf4cb2f73",
}


def pinned_bytes(path, limit=4 * 1024**2):
    for member in (path, *path.parents):
        info = member.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise RuntimeError("Ordinary pinned test input required")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise RuntimeError("Bounded ordinary test input required")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) != info.st_size or len(raw) > limit:
        raise RuntimeError("Test input changed or exceeded its bound")
    return raw


def source_map():
    result = {}
    for path in (Path(__file__).resolve(), *FIXED):
        result[str(path)] = hashlib.sha256(pinned_bytes(path)).hexdigest()
        if path in FIXED and result[str(path)] != FIXED[path]:
            raise RuntimeError("Frozen lifecycle-predicate input changed: " + str(path))
    return result


def load_subject():
    spec = importlib.util.spec_from_file_location("independent_lifecycle_event_subject", SUBJECT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LifecycleEventPredicateControls(unittest.TestCase):
    observed = []

    @classmethod
    def setUpClass(cls):
        source_map()
        cls.oracle = json.loads(pinned_bytes(ORACLE))
        if (cls.oracle["schema"] != "sourceq_lifecycle_event_oracles_v1"
                or cls.oracle["positive_rows"] != 3 or cls.oracle["negative_rows"] != 12
                or cls.oracle["planned_unittest_methods"] != 2
                or [row["id"] for row in cls.oracle["rows"]]
                != ["P01", "P02", "P03"] + ["N%02d" % value for value in range(1, 13)]):
            raise RuntimeError("Exact frozen15-row oracle required")
        cls.subject = load_subject()

    def assert_valid_baseline(self):
        probe = copy.deepcopy(self.oracle["rows"][0]["row"])
        value = self.subject.verify_lifecycle_zero_session_events(probe)
        self.assertEqual(value, self.oracle["expected_success"])
        for name in ("source_qualified", "software_qualified", "scientific_qualified"):
            self.assertIs(value[name], False)

    def test_supporting_events_are_unqualified_metadata(self):
        for case in self.oracle["rows"][:3]:
            with self.subTest(case=case["id"]):
                self.assertIs(case["accepted"], True)
                row = copy.deepcopy(case["row"])
                before = copy.deepcopy(row)
                value = self.subject.verify_lifecycle_zero_session_events(row)
                self.assertEqual(value, self.oracle["expected_success"])
                for name in ("source_qualified", "software_qualified", "scientific_qualified"):
                    self.assertIs(value[name], False)
                self.assertEqual(row, before)
                self.observed.append(dict(id=case["id"], accepted=True,
                                          source_qualified=False, software_qualified=False,
                                          scientific_qualified=False))

    def test_unsupported_or_contradictory_events_are_rejected(self):
        for case in self.oracle["rows"][3:]:
            with self.subTest(case=case["id"]):
                self.assert_valid_baseline()
                self.assertIs(case["accepted"], False)
                with self.assertRaises(self.subject.SourceQualificationError) as caught:
                    self.subject.verify_lifecycle_zero_session_events(copy.deepcopy(case["row"]))
                self.assertEqual(caught.exception.code, self.oracle["expected_error_code"])
                self.observed.append(dict(id=case["id"], accepted=False,
                                          code=caught.exception.code))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    output = Path(args.output_directory)
    if (not __debug__ or not output.is_absolute() or output.exists() or output.is_symlink()
            or ROOT in output.parents or output.parent.resolve(strict=True) != output.parent):
        raise RuntimeError("Unoptimized test and fresh external absolute output required")
    started = time.monotonic_ns()
    deadline = started + 90 * 10**9
    before = source_map()
    output.mkdir()
    with (output / "unittest.log").open("x", encoding="utf-8", newline="\n") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(LifecycleEventPredicateControls))
        stream.flush()
        os.fsync(stream.fileno())
    transcript = pinned_bytes(output / "unittest.log")
    sys.stderr.write(transcript.decode("utf-8"))
    sys.stderr.flush()
    after = source_map()
    finished = time.monotonic_ns()
    observed = LifecycleEventPredicateControls.observed
    record = dict(schema="sourceq_lifecycle_event_independent_tests_v1",
                  complete=(result.wasSuccessful() and not result.skipped and result.testsRun == 2
                            and len(observed) == 15 and before == after and finished < deadline),
                  unittest_methods=result.testsRun, expected_methods=2,
                  literal_rows=len(observed), expected_rows=15, rows=observed,
                  failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                  source_before=before, source_after=after, source_unchanged=before == after,
                  raw_unittest_sha256=hashlib.sha256(transcript).hexdigest(),
                  raw_unittest_bytes=len(transcript), started_ns=started,
                  finished_ns=finished, deadline_ns=deadline,
                  fixtures_are_metadata_not_native_observations=True,
                  actual_OS_observations_or_owner_launches=0, Windows_native_cases=0,
                  POSIX_native_cases=0, qualification_issued=False,
                  source_qualified=False, software_qualified=False, scientific_qualified=False)
    raw = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with (output / "RESULT.json").open("xb") as stream:
        if stream.write(raw) != len(raw):
            raise RuntimeError("Short result write")
        stream.flush()
        os.fsync(stream.fileno())
    if time.monotonic_ns() >= deadline:
        (output / "RESULT.json").rename(output / "RESULT.before-late-write.json")
        record.update(complete=False, deadline_exhausted=True)
        (output / "RESULT.json").write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(0 if record["complete"] else 1)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
