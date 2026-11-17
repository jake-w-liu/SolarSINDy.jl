"""Unqualified metadata boundary probes for the original canonical 21600s cap."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SUBJECT = ROOT / "tools/actual_b_chain/materialize_source_qualification.py"
ORACLE = HERE / "canonical_native_boundary_oracles_v1.json"
ORACLE_SHA = "3e1957e74d0c29a49d59e3e3f2f0b19583bbfd9a74cf532feb12b59f91f01114"

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def native_shape(wall):
    # Deliberately a shape fixture: no process/PID/closure is claimed observed.
    return dict(pid=424242, native_exit=0, timed_out=False, leader_reaped=True,
                ownership_boundary_closed=True, all_descendants_os_reaped=False,
                ownership="CREATE_SUSPENDED -> AssignProcessToJobObject -> ResumeThread",
                descendant_cleanup="private Windows Job Object closed with KILL_ON_JOB_CLOSE",
                descendant_evidence="unqualified predicate fixture; no OS observation",
                wall_seconds=wall, wrapper_observed_wall_seconds=wall)

class CanonicalNativeBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if digest(ORACLE.read_bytes()) != ORACLE_SHA:
            raise RuntimeError("Independent canonical boundary oracle changed")
        cls.oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
        spec = importlib.util.spec_from_file_location("independent_canonical_native_boundary_subject", SUBJECT)
        cls.m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.m
        spec.loader.exec_module(cls.m)

    def accepted(self, native):
        result = self.m.verify_canonical_native(native)
        self.assertEqual(result, dict(complete_as_native_structure=True, native_allowance_seconds=21600,
                                      source_qualified=False, software_qualified=False, scientific_qualified=False))

    def rejected(self, native):
        with self.assertRaises(self.m.SourceQualificationError) as raised:
            self.m.verify_canonical_native(native)
        self.assertEqual(raised.exception.code, "nonpositive_owned_stage")

    def test_original_allowance_accepts_above900_below21600(self):
        self.assertEqual(self.m.CANONICAL_NATIVE_SECONDS, 21600)
        for wall in self.oracle["positive_walls"]:
            with self.subTest(wall=wall):
                self.accepted(native_shape(wall))

    def test_wall_exact_limit_late_negative_and_nonfinite_reject(self):
        self.accepted(native_shape(1000))
        for wall in (21600, 21601, -1, float("inf"), float("nan")):
            with self.subTest(wall=str(wall)):
                self.rejected(native_shape(wall))

    def test_nonpositive_native_receipts_never_accept(self):
        self.accepted(native_shape(1000))
        for field, value in self.oracle["negative_receipt_fields"]:
            with self.subTest(field=field):
                self.rejected(dict(native_shape(1000), **{field: value}))

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
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CanonicalNativeBoundary))
    after = {str(path): digest(path.read_bytes()) for path in paths}
    record = dict(schema="sourceq_canonical_native_boundary_independent_tests_v1",
                  complete=result.wasSuccessful() and not result.skipped and result.testsRun == 3 and before == after,
                  unittest_methods=result.testsRun, expected_methods=3,
                  literal_positive_rows=3, literal_negative_rows=9,
                  failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                  source_before=before, source_after=after, source_unchanged=before == after,
                  started_ns=started, finished_ns=time.monotonic_ns(),
                  actual_numerical_or_owner_launches_requested=0, metadata_fixtures_are_native_evidence=False,
                  source_qualified=False, software_qualified=False, scientific_qualified=False)
    (output / "RESULT.json").write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    raise SystemExit(0 if record["complete"] else 1)

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
