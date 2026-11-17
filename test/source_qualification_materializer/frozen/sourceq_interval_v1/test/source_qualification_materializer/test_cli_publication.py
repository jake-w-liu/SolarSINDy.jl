"""Ordinary unqualified completion-file/directory protocol probes only."""
from pathlib import Path
import argparse
import copy
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
ORACLE = HERE / "cli_publication_oracles_v1.json"
ORACLE_SHA = "8f82b0b80afa97c3683b265dcd20aab9de2d6cae7c5cc1cb6107d105b5ff5e42"
MATERIALIZER = ROOT / "tools/actual_b_chain/materialize_source_qualification.py"
RUNNER = ROOT / "tools/actual_b_chain/run_interval_workers_qualification.py"
OBSERVATIONS = []

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

class PublicationBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if digest(ORACLE.read_bytes()) != ORACLE_SHA:
            raise RuntimeError("Independent CLI publication oracle changed")
        cls.m = load(MATERIALIZER, "independent_sourceq_completion_protocol")
        cls.r = load(RUNNER, "independent_interval_completion_protocol")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="unqualified-publication-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.payload = b'{"schema":"unqualified_protocol_probe_v1","source_qualified":false,"software_qualified":false,"scientific_qualified":false}\n'

    def staged(self, label):
        pending = self.root / (label + ".pending")
        final = self.root / (label + ".accepted")
        pending.mkdir()
        path = pending / "UNQUALIFIED_METADATA.json"
        with path.open("xb") as stream:
            self.assertEqual(stream.write(self.payload), len(self.payload))
            stream.flush()
            os.fsync(stream.fileno())
        return pending, final

    def recheck(self, pending):
        self.assertEqual((pending / "UNQUALIFIED_METADATA.json").read_bytes(), self.payload)

    def positive_materializer(self, label):
        pending, final = self.staged(label)
        notice = []
        def emit():
            notice.append(not final.exists())
        returned = self.m.finalize_materialization(pending, final, emit=emit,
                                                   recheck=lambda: self.recheck(pending))
        self.assertEqual(returned, final)
        self.assertEqual(notice, [True])
        self.assertEqual((final / "UNQUALIFIED_METADATA.json").read_bytes(), self.payload)
        return final

    def record(self):
        return dict(schema="unqualified_interval_protocol_probe_v1", complete=True,
                    native={"observed": False, "kind": "unqualified non-owner metadata marker"},
                    source_qualified=False, software_qualified=False, scientific_qualified=False)

    def positive_runner(self, label):
        path = self.root / (label + ".json")
        record = self.record()
        notices = []
        self.assertIs(self.r.finalize_interval_terminal(path, record, time.monotonic_ns() + 10**10,
                                                       emit=lambda: notices.append(not path.exists())), True)
        self.assertEqual(notices, [True])
        published = json.loads(path.read_text(encoding="utf-8"))
        self.assertIs(published["complete"], True)
        for key in ("source_qualified", "software_qualified", "scientific_qualified"):
            self.assertIs(published[key], False)

    def test_materializer_notice_precedes_accepted_directory(self):
        final = self.positive_materializer("normal")
        self.assertFalse((final / "ROOT_QUALIFICATION.json").exists())
        self.assertFalse((final / "COMPONENT_SOFTWARE.json").exists())

    def test_materializer_output_failure_never_accepts_directory(self):
        self.positive_materializer("baseline-before-output-fault")
        pending, final = self.staged("output-fault")
        def emit():
            self.assertFalse(final.exists())
            raise BrokenPipeError("independent unqualified stdout failure")
        with self.assertRaises(self.m.SourceQualificationError) as error:
            self.m.finalize_materialization(pending, final, emit=emit, recheck=lambda: self.recheck(pending))
        self.assertEqual(error.exception.code, "partial_or_late_publication")
        self.assertFalse(final.exists())
        OBSERVATIONS.append(dict(case="materializer_output", accepted_directory_exists=final.exists()))

    def test_materializer_postcommit_clock_rolls_back_only_owned_directory(self):
        self.positive_materializer("baseline-before-late")
        pending, final = self.staged("late")
        previous = self.m._DEADLINE
        real_rename, real_replace = os.rename, os.replace
        observed = dict(actual_commit=False)
        def atomic(original):
            def wrapped(source, destination, *args, **kwargs):
                result = original(source, destination, *args, **kwargs)
                if Path(destination) == final and not observed["actual_commit"]:
                    observed["actual_commit"] = True
                    self.assertEqual((final / "UNQUALIFIED_METADATA.json").read_bytes(), self.payload)
                    self.m._DEADLINE = time.monotonic_ns() - 1
                return result
            return wrapped
        try:
            with mock.patch.object(os, "rename", atomic(real_rename)), mock.patch.object(os, "replace", atomic(real_replace)):
                with self.assertRaises(self.m.SourceQualificationError) as error:
                    self.m.finalize_materialization(pending, final, emit=lambda: None,
                                                    recheck=lambda: self.recheck(pending))
        finally:
            self.m._DEADLINE = previous
        self.assertEqual(error.exception.code, "partial_or_late_publication")
        self.assertIsInstance(error.exception.rollback_status, dict)
        observed["accepted_directory_exists"] = final.exists()
        OBSERVATIONS.append(dict(case="materializer_late", **observed))
        self.assertIs(observed["actual_commit"], True)
        self.assertFalse(final.exists())

    def test_interval_output_failure_keeps_no_positive_completion(self):
        self.positive_runner("baseline-before-output")
        path = self.root / "output-fault.json"
        record = self.record()
        original_native = copy.deepcopy(record["native"])
        def emit():
            self.assertFalse(path.exists())
            raise BrokenPipeError("independent interval stdout failure")
        with self.assertRaises(RuntimeError):
            self.r.finalize_interval_terminal(path, record, time.monotonic_ns() + 10**10, emit=emit)
        if path.exists():
            self.assertIs(json.loads(path.read_text(encoding="utf-8"))["complete"], False)
        self.assertEqual(record["native"], original_native)
        OBSERVATIONS.append(dict(case="interval_output", positive_completion_exists=path.exists() and
                                 json.loads(path.read_text(encoding="utf-8")).get("complete") is True))

    def test_interval_postcommit_late_and_replaced_paths(self):
        for replace_owned in (False, True):
            with self.subTest(replaced=replace_owned):
                self.positive_runner("baseline-" + str(replace_owned))
                path = self.root / ("late-" + str(replace_owned) + ".json")
                record = self.record()
                original_native = copy.deepcopy(record["native"])
                deadline = time.monotonic_ns() + 10**10
                real_now = time.monotonic_ns
                real_rename, real_replace = os.rename, os.replace
                observed = dict(actual_commit=False, replacement_written=False)
                foreign = b"foreign replacement in test-owned namespace; no accepted record\n"
                def atomic(original):
                    def wrapped(source, destination, *args, **kwargs):
                        result = original(source, destination, *args, **kwargs)
                        if Path(destination) == path and not observed["actual_commit"]:
                            observed["actual_commit"] = True
                            self.assertIs(json.loads(path.read_text(encoding="utf-8"))["complete"], True)
                            if replace_owned:
                                real_rename(path, self.root / ("original-owned-" + str(replace_owned) + ".json"))
                                path.write_bytes(foreign)
                                observed["replacement_written"] = True
                        return result
                    return wrapped
                def now():
                    return deadline + 1 if observed["actual_commit"] else real_now()
                with mock.patch.object(os, "rename", atomic(real_rename)), mock.patch.object(os, "replace", atomic(real_replace)), \
                     mock.patch.object(time, "monotonic_ns", now):
                    with self.assertRaises(RuntimeError) as error:
                        self.r.finalize_interval_terminal(path, record, deadline, emit=lambda: None)
                self.assertIsInstance(error.exception.rollback_status, dict)
                self.assertIs(observed["actual_commit"], True)
                self.assertEqual(record["native"], original_native)
                if replace_owned:
                    self.assertIs(observed["replacement_written"], True)
                    self.assertEqual(path.read_bytes(), foreign)
                elif path.exists():
                    self.assertIs(json.loads(path.read_text(encoding="utf-8"))["complete"], False)
                OBSERVATIONS.append(dict(case="interval_late", replaced=replace_owned,
                                         path_exists=path.exists(), **observed))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    output = Path(args.output_directory)
    if not __debug__ or not output.is_absolute() or output.exists() or output.is_symlink() or ROOT in output.parents:
        raise RuntimeError("Unoptimized test and fresh external absolute output required")
    paths = [Path(__file__), ORACLE, MATERIALIZER, RUNNER]
    before = {str(path): digest(path.read_bytes()) for path in paths}
    output.mkdir()
    started = time.monotonic_ns()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PublicationBoundary))
    after = {str(path): digest(path.read_bytes()) for path in paths}
    record = dict(schema="sourceq_and_interval_cli_publication_independent_tests_v1",
                  complete=result.wasSuccessful() and not result.skipped and result.testsRun == 5 and before == after,
                  unittest_methods=result.testsRun, expected_methods=5, observations=OBSERVATIONS,
                  failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                  source_before=before, source_after=after, source_unchanged=before == after,
                  started_ns=started, finished_ns=time.monotonic_ns(),
                  full_qualification_or_actual_owner_fixture_supplied=False,
                  numerical_or_owner_launches_requested=0, authority_or_evidence_verifier_mocked=False,
                  source_qualified=False, software_qualified=False, scientific_qualified=False)
    (output / "RESULT.json").write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    raise SystemExit(0 if record["complete"] else 1)

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
