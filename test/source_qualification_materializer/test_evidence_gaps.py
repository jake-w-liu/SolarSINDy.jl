"""Finite evidence-reader predicates; fixtures never issue SourceQ or observe children."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
ORACLE = HERE / "evidence_gap_oracles_v1.json"
ELAPSED = HERE / "stage_elapsed_oracles_v1.json"
ORACLE_SHA = "2ca2cb56576bb8aac805eb4acaac5d5c17686f546d961b7426b15cc55e939bf7"
ELAPSED_SHA = "611d707f47368286338ed8fef9eb7e425239bc83bff60620ca8b9af0941f3d42"
MATERIALIZER = ROOT / "tools/actual_b_chain/materialize_source_qualification.py"
HELPER = ROOT / "tools/actual_b_chain/sourceq_full_scale_verifier.py"
HEAD = "69a8f1106930eb1cd3b02d6c1c488e1da9d44841"
PROOF_SHA = "1c32f26c06c06c7d9e17850cb4b56bb9f089eed0b4395961109c8e7153f04fa1"
FIXED = {
    "native69a8-source-proof.json": PROOF_SHA,
    "native69a8-posix-macos.zip": "cb9775aeebc2acb0b82a9e5edbb31628821e6c79546849e41073b0378697c520",
    "native69a8-anchor-macos.zip": "3781e526631985414946abfbd02f8e99cefa4c02d7854fd4e59130f08998e778",
}
PAIRS = (("start_new_session=True", "owned POSIX session has no live members after stage"),
         ("CREATE_SUSPENDED -> AssignProcessToJobObject -> ResumeThread",
          "private Windows Job Object closed with KILL_ON_JOB_CLOSE"))

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def native_shape(pair=PAIRS[0], wall=1):
    # Legacy six fields, expressly an unqualified predicate fixture.
    return dict(pid=424242, native_exit=0, timed_out=False, wall_seconds=wall,
                ownership=pair[0], descendant_cleanup=pair[1])

def acquisition_shapes(proof):
    """Parsed-header format probes, not retained original execution/collector evidence."""
    owners, reviews = [], []
    guard_actor = digest(b"unqualified guarding-helper fixture")
    for offset, (cohort, run) in enumerate((("ci18", 42797906679), ("native3", 42797153766))):
        actor = digest(("unqualified collector " + cohort).encode())
        context = dict(actor_sha256=guard_actor, bindings_sha256=digest(b"unqualified binding"),
                       control_sha256={"/unqualified/guard.py": guard_actor},
                       head=HEAD, tree=proof["tree"], source_files=proof["source_files"],
                       source_proof_sha256=PROOF_SHA)
        context["all" + str(proof["source_files"]) + "_raw_source_hashes_match"] = True
        start = 100000000000 + offset * 10**12
        native = dict(native_shape(PAIRS[1], 2), pid=41000 + offset, leader_reaped=True,
                      ownership_boundary_closed=True, all_descendants_os_reaped=False,
                      wrapper_observed_wall_seconds=2.1)
        raw_sha = digest(b"")
        terminal = dict(complete=True, head=HEAD, run_id=run, actor_sha256=actor,
                        source_before=copy.deepcopy(context), source_after=copy.deepcopy(context),
                        source_unchanged=True, native=native, raw_sha256=raw_sha, raw_bytes=0,
                        started_ns=start, finished_ns=start + 5 * 10**9,
                        worker_deadline_ns=start + 200 * 10**9, deadline_ns=start + 240 * 10**9,
                        original_deadline_ns=388382680849500, API_POSTs=0, reruns=0, cancellations=0,
                        source_qualified=False, software_qualified=False, scientific_qualified=False)
        contract = dict(actor_sha256=actor, source_before=copy.deepcopy(context), started_ns=start,
                        deadline_ns=terminal["worker_deadline_ns"], whole_deadline_ns=terminal["deadline_ns"],
                        original_deadline_ns=terminal["original_deadline_ns"])
        launch = dict(pid=native["pid"], monotonic_ns=start + 10**9)
        item = dict(cohort=cohort, terminal=terminal, contract=contract, launch=launch,
                    actor_sha256=actor, raw_sha256=raw_sha, raw_bytes=0,
                    terminal_sha256=digest(encoded(terminal)), contract_sha256=digest(encoded(contract)),
                    launch_sha256=digest(encoded(launch)), proof=copy.deepcopy(proof), proof_sha256=PROOF_SHA)
        owners.append(item)
        data = dict(complete_as_data_only_artifact_reconciliation=True, head=HEAD, run_id=run,
                    source_proof_sha256=PROOF_SHA, terminal_sha256=item["terminal_sha256"],
                    actor_sha256=actor, source_qualified=False, software_qualified=False, scientific_qualified=False)
        reviews.append(dict(sha256=digest(encoded(data)), data=data))
    return owners, reviews

def rewrite_zip(source, target, *, omit=None, mutate=None):
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as changed:
        for entry in original.infolist():
            if entry.filename == omit:
                continue
            if entry.file_size > 4 * 1024**2:
                raise RuntimeError("Frozen inert fixture member bound")
            raw = original.read(entry)
            if mutate is not None:
                raw = mutate(entry.filename, raw)
            changed.writestr(entry, raw)

class EvidenceGapControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if digest(ORACLE.read_bytes()) != ORACLE_SHA or digest(ELAPSED.read_bytes()) != ELAPSED_SHA:
            raise RuntimeError("Independent evidence/elapsed oracles changed")
        for name, pin in FIXED.items():
            if digest((FIXTURES / name).read_bytes()) != pin:
                raise RuntimeError("Original inert native fixture changed")
        cls.proof = json.loads((FIXTURES / "native69a8-source-proof.json").read_text(encoding="utf-8"))
        cls.m = load(MATERIALIZER, "independent_finite_evidence_subject")
        cls.h = load(HELPER, "independent_finite_stage_subject")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sourceq-evidence-predicates-")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)

    def unqualified(self, result):
        for field in ("source_qualified", "software_qualified", "scientific_qualified"):
            self.assertIs(result[field], False)

    def rejects(self, code, fn, *args, **kwargs):
        with self.assertRaises(self.m.SourceQualificationError) as error:
            fn(*args, **kwargs)
        self.assertEqual(error.exception.code, code)

    def test_exact_native_pairs_preserve_legacy6_and_check_enrichment(self):
        for pair in PAIRS:
            with self.subTest(pair=pair[0]):
                self.unqualified(self.m.verify_canonical_native(native_shape(pair, 1000)))
        for pair in (("borrowed PID", "ignored"), (PAIRS[0][0], PAIRS[1][1])):
            self.rejects("nonpositive_owned_stage", self.m.verify_canonical_native, native_shape(pair, 1000))
        for field, value in (("leader_reaped", False), ("ownership_boundary_closed", False),
                             ("all_descendants_os_reaped", True)):
            bad = dict(native_shape(wall=1000), **{field: value})
            self.rejects("nonpositive_owned_stage", self.m.verify_canonical_native, bad)

    def test_original_whole_stage_elapsed_bounds_are_separate(self):
        for name, elapsed, bound in (("runtime", 59.5, 60), ("case-workload", 7199, 7200),
                                     ("native-agreement", 7199.5, 7200)):
            stage = dict(native=native_shape(), elapsed_seconds=elapsed, timeout_seconds=bound)
            self.assertIs(self.h.verify_stage_evidence(stage, name), stage)
        for name, elapsed, bound in (("runtime", 60, 60), ("case-workload", 7200, 7200),
                                     ("native-agreement", 7201, 7200), ("runtime", -1, 60),
                                     ("runtime", float("nan"), 60), ("runtime", float("inf"), 60),
                                     ("runtime", True, 60)):
            with self.subTest(stage=name, elapsed=str(elapsed)):
                stage = dict(native=native_shape(), elapsed_seconds=elapsed, timeout_seconds=bound)
                with self.assertRaises(self.h.FullScaleVerificationError) as error:
                    self.h.verify_stage_evidence(stage, name)
                self.assertEqual(error.exception.code, "nonpositive_owned_stage")
        with self.assertRaises(self.h.FullScaleVerificationError) as error:
            self.h.verify_stage_evidence(dict(native=native_shape(), timeout_seconds=60), "runtime")
        self.assertEqual(error.exception.code, "nonpositive_owned_stage")

    def test_acquisition_context_requires_distinct_identity_and_clock_joins(self):
        owners, reviews = acquisition_shapes(self.proof)
        kwargs = dict(head=HEAD, ci_run=42797906679, native_run=42797153766, proof=self.proof)
        baseline = self.m.verify_acquisition_context(owners, reviews, **kwargs)
        self.unqualified(baseline)
        self.assertEqual(baseline["cohorts"], ["ci18", "native3"])
        self.assertEqual((baseline["owners"], baseline["reviews"]), (2, 2))
        self.rejects("coherent_header_raw_counterfeit", self.m.verify_acquisition_context,
                     [copy.deepcopy(owners[0]), copy.deepcopy(owners[0])], reviews, **kwargs)
        self.rejects("coherent_header_raw_counterfeit", self.m.verify_acquisition_context,
                     owners, [copy.deepcopy(reviews[0]), copy.deepcopy(reviews[0])], **kwargs)
        bad = copy.deepcopy(owners)
        bad[0]["terminal"]["head"] = "f" * 40
        bad[0]["terminal_sha256"] = digest(encoded(bad[0]["terminal"]))
        self.rejects("wrong_original_identity", self.m.verify_acquisition_context, bad, reviews, **kwargs)
        bad = copy.deepcopy(owners)
        bad[0]["terminal"]["finished_ns"] = bad[0]["terminal"]["deadline_ns"] + 1
        bad[0]["terminal_sha256"] = digest(encoded(bad[0]["terminal"]))
        self.rejects("coherent_header_raw_counterfeit", self.m.verify_acquisition_context, bad, reviews, **kwargs)

    def test_owner_case_files_and_predicates_cannot_be_replaced_by_headers(self):
        for source_name in ("native69a8-posix-macos.zip", "native69a8-anchor-macos.zip"):
            self.unqualified(self.m.verify_owner_artifact(FIXTURES / source_name,
                                                         platform="macos-26-intel", proof=self.proof))
        source = FIXTURES / "native69a8-posix-macos.zip"
        for index, missing in enumerate(("native/0-no_descendant.json", "native/old-0-transient_stop.json")):
            target = self.work / f"missing-v3-{index}.zip"
            rewrite_zip(source, target, omit=missing)
            self.rejects("missing_current_owner_evidence", self.m.verify_owner_artifact,
                         target, platform="macos-26-intel", proof=self.proof)
        def false_eperm(name, raw):
            if name == "native/1-other_group_12s.json":
                row = json.loads(raw)
                self.assertIs(row["checks"]["eperm_native_zombie_only"], True)
                self.assertIs(row["passed"], True)
                row["checks"]["eperm_native_zombie_only"] = False
                return encoded(row)
            return raw
        target = self.work / "false-eperm.zip"
        rewrite_zip(source, target, mutate=false_eperm)
        self.rejects("coherent_header_raw_counterfeit", self.m.verify_owner_artifact,
                     target, platform="macos-26-intel", proof=self.proof)
        target = self.work / "missing-anchor-fault.zip"
        rewrite_zip(FIXTURES / "native69a8-anchor-macos.zip", target,
                    omit="native/failure-cleanup-identity-fault.json")
        self.rejects("missing_current_owner_evidence", self.m.verify_owner_artifact,
                     target, platform="macos-26-intel", proof=self.proof)
        def wrong_raw(name, raw):
            return raw + b"inert tampered raw\n" if name == "native/0-no_descendant/native.raw" else raw
        target = self.work / "wrong-native-raw.zip"
        rewrite_zip(source, target, mutate=wrong_raw)
        self.rejects("coherent_header_raw_counterfeit", self.m.verify_owner_artifact,
                     target, platform="macos-26-intel", proof=self.proof)

    def test_worker_thread_counts_require_integer1_not_numeric_equality(self):
        good = dict(blas_threads=1, julia_threads=1)
        baseline = self.m.verify_worker_threads(good)
        self.unqualified(baseline)
        self.assertEqual((baseline["blas_threads"], baseline["julia_threads"]), (1, 1))
        for field in good:
            for value in (True, 1.0, "1", 0, 2):
                with self.subTest(field=field, value=str(value)):
                    self.rejects("workers17_unready", self.m.verify_worker_threads,
                                 dict(good, **{field: value}))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    output = Path(args.output_directory)
    if not __debug__ or not output.is_absolute() or output.exists() or output.is_symlink() or ROOT in output.parents:
        raise RuntimeError("Unoptimized test and fresh external absolute output required")
    paths = [Path(__file__), ORACLE, ELAPSED, MATERIALIZER, HELPER, *[FIXTURES / name for name in FIXED]]
    before = {str(path): digest(path.read_bytes()) for path in paths}
    output.mkdir()
    started = time.monotonic_ns()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EvidenceGapControls))
    after = {str(path): digest(path.read_bytes()) for path in paths}
    record = dict(schema="sourceq_finite_evidence_gaps_independent_tests_v1",
                  complete=result.wasSuccessful() and not result.skipped and result.testsRun == 5 and before == after,
                  unittest_methods=result.testsRun, expected_methods=5,
                  failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                  source_before=before, source_after=after, source_unchanged=before == after,
                  started_ns=started, finished_ns=time.monotonic_ns(),
                  metadata_native_and_context_fixtures_are_actual_collector_evidence=False,
                  actual_numerical_or_owner_launches_requested=0, full_qualification_fixture_or_bypass=False,
                  source_qualified=False, software_qualified=False, scientific_qualified=False)
    (output / "RESULT.json").write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    raise SystemExit(0 if record["complete"] else 1)

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
