"""Independent W01-W10 metadata/routing probes; no Julia or owner is launched."""
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

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SUBJECT = ROOT / "tools/actual_b_chain/run_interval_workers_qualification.py"
ORACLE = HERE / "workers17_oracles_v1.json"
ORACLE_SHA = "94331d0a6ea299803414e3b9243cc1777468da2971612d9bf22d9839466f7058"
DRIVER = "test/test_interval_mean_forecaster.jl"
WORKER = "test/interval_mean_forecaster/worker.jl"
BINDINGS = "test/interval_mean_forecaster/bindings.jl"
PINS = {
    DRIVER: "72dd65b36d313b39703c93a60a4a92a91146c786aa4e4069fac4e4aa99d1fbcd",
    WORKER: "e3cd1ca92c85c544b812b8debf56920c47ba7fba49cd156dd9387e4b9df56842",
    BINDINGS: "5cbef72d434cae309d382c110f832566aefe068f1661b31c7fd9464f11c7c939",
}
JOBS = ("smoke", "scanner", "independent2272", "public-baseline", "public-candidate",
        "cache514", "original", "stage4", "stream-baseline", "stream-candidate",
        "training362", "warm44", "tenk-baseline", "tenk-candidate", "grammar-producer",
        "large79", "disk-fit")
IMF_PINS = {
    "DstIntervalObserver.jl": "48e2af386d2b9b1a2b8afb4366e1c48508261e54a81447d438fd8a419581b816",
    "DstIntervalReplay.jl": "f2962c092bc6bfd6be82a5b8ad95e2c494dddbc4a2addda4f57140e31d0c6fa5",
    "DstIntervalAdapter.jl": "b432a8fc65d3e425a67ff2b225a6c0e5323ae7fdb8822dd228677b7998a5fe5a",
    "DstIntervalStream.jl": "20a63654c429aa8be219f857b1e83a2529f334499e38198d8f332a07f22fcd44",
    "DstIntervalTraining.jl": "2e13631e04cd48b36b033721935cee7dc47da76bd692f84ef0d3daae5c142b89",
    "DstIntervalArtifacts.jl": "fbeb40ecfe1c9ff32a17ba6c97b9eadccdd4c88e924aeddeb7ce00616fc3c29e",
    "DstIntervalExperiment.jl": "50e9bd1876af69a927f1bf6211925285f532e2f6a040cff5460b12c308d0fedd",
    "sandu_fullchannel_features.jl": "2306f63cadfb5d4ae16579cabb114b813f1f41678aaa6dbdd90caad316ff792c",
}
GIB = 1024**3
SEMANTIC_ERRORS = (RuntimeError, ValueError, TypeError, KeyError)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def json_bytes(data):
    return (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")

def load_subject():
    spec = importlib.util.spec_from_file_location("independent_workers17_subject", SUBJECT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def metadata_fixture(directory, runtime="1.12.6", *, historical_prefix=True, trailing_project=False):
    """Original-shaped literal metadata, not real worker/PID/runtime evidence."""
    runs, receipts, logs = [], {}, {}
    for index, job in enumerate(JOBS):
        output = directory / "interval-functional" / job
        project = str(ROOT) + ("/" if trailing_project else "")
        prefix = [str(directory / "unexecuted-julia")]
        if historical_prefix:
            prefix += ["-C", "generic", "-J" + str(directory / "unexecuted-sysimage"),
                       "--pkgimages=no", "-g1", "--startup-file=no"]
        argv = prefix + ["--startup-file=no", "--pkgimages=no", "--threads=1",
                         "--project=" + project, str(ROOT / WORKER), job, str(output)]
        raw = ("unqualified literal job log\nPACKAGE_BOUND_WORKER_COMPLETE=" + job + "\n").encode("utf-8")
        logs[job] = dict(bytes=len(raw), sha256=digest(raw), complete_marker=job, unsafe_runtime_warning=False)
        runs.append(dict(job=job, argv=argv, pid=42000 + index,
                         started_utc=f"2026-11-10T10:{index:02d}:00", completed_utc=f"2026-11-10T10:{index:02d}:02",
                         native_exit=0, timed_out=False, receipt_valid=True,
                         environment_controls={key: str(output / "tmp") for key in ("TMPDIR", "TMP", "TEMP")},
                         log_sha256=digest(raw)))
        receipts[job] = dict(complete=True, job=job, julia_version=runtime, kernel="Linux", architecture="x86_64",
                             package_source=str(ROOT / "src/SolarSINDy.jl"), source_sha256=dict(IMF_PINS),
                             worker_sha256=PINS[WORKER], bindings_sha256=PINS[BINDINGS],
                             blas_threads=1, julia_threads=1,
                             scope="bounded synthetic functional qualification; no empirical skill or deployment claim")
    terminal = dict(complete=True, required_jobs=list(JOBS), runs=runs, worker_sha256=PINS[WORKER],
                    scope="synthetic correctness and same-platform before/after bits; no benchmark or empirical skill")
    return terminal, receipts, logs

def native_shape():
    """Explicit predicate fixture, never a genuine NativeResult."""
    start = 1000000000000
    native = dict(pid=424242, native_exit=0, timed_out=False, leader_reaped=True,
                  ownership_boundary_closed=True, all_descendants_os_reaped=False,
                  wall_seconds=2.0, wrapper_observed_wall_seconds=2.1, ownership="start_new_session=True",
                  descendant_cleanup="owned POSIX session has no live members after stage")
    launch = dict(pid=424242, monotonic_ns=start + 1000000)
    return native, launch, dict(native_started_ns=start, native_deadline_ns=start + 3600 * 10**9)

class Workers17Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if digest(ORACLE.read_bytes()) != ORACLE_SHA:
            raise RuntimeError("Frozen W01-W10 oracle changed")
        oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
        if len(oracle["negative_rows"]) != 10 or tuple(oracle["driver"]["ordered_jobs"]) != JOBS:
            raise RuntimeError("Independent oracle inventory changed")
        for member, expected in PINS.items():
            if digest((ROOT / member).read_bytes()) != expected:
                raise RuntimeError("Original unchanged Julia driver/worker/bindings changed")
        cls.m = load_subject()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="workers17-independent-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def validate(self, terminal, receipts, logs, runtime="1.12.6"):
        return self.m.validate_workers(terminal, receipts, logs, runtime=runtime,
                                       worker_sha256=PINS[WORKER], bindings_sha256=PINS[BINDINGS],
                                       science_checkout=ROOT)

    def baseline(self, runtime="1.12.6", **kwargs):
        fixture = metadata_fixture(self.directory, runtime, **kwargs)
        unchanged = json_bytes(fixture)
        checked = self.validate(*fixture, runtime=runtime)
        self.assertEqual(json_bytes(fixture), unchanged)
        self.assertEqual(checked, [dict(job=job, pid=42000 + i) for i, job in enumerate(JOBS)])
        self.no_publication()
        return fixture

    def no_publication(self):
        self.assertFalse((self.directory / "TERMINAL.json").exists())
        self.assertFalse((self.directory / "qualifications").exists())

    def rejects_workers(self, fixture, runtime="1.12.6"):
        with self.assertRaises(SEMANTIC_ERRORS):
            self.validate(*fixture, runtime=runtime)
        self.no_publication()

    def test_w01_missing_runtime_cannot_be_substituted(self):
        for runtime in ("1.12.6", "1.13.1"):
            with self.subTest(runtime=runtime):
                terminal, receipts, logs = self.baseline(runtime)
                other = "1.13.1" if runtime == "1.12.6" else "1.12.6"
                for receipt in receipts.values():
                    receipt["julia_version"] = other
                self.rejects_workers((terminal, receipts, logs), runtime)
        # One runtime's pure format return is not a both-runtime aggregate or SourceQ.
        self.no_publication()

    def test_w02_exact_original17_order_and_inventory(self):
        driver = (ROOT / DRIVER).read_bytes()
        self.assertEqual(self.m.parse_jobs(driver), JOBS)
        mutated = driver.replace(b'const JOBS = ("smoke", ', b"const JOBS = (", 1)
        self.assertNotEqual(mutated, driver)
        with self.assertRaises(SEMANTIC_ERRORS):
            self.m.parse_jobs(mutated)
        fixture = self.baseline()
        fixture[0]["runs"][1], fixture[0]["runs"][0] = fixture[0]["runs"][0], fixture[0]["runs"][1]
        self.rejects_workers(fixture)

    def test_w03_nonzero_timeout_or_invalid_receipt_stays_negative(self):
        for field, value in (("native_exit", 1), ("timed_out", True), ("receipt_valid", False)):
            with self.subTest(field=field):
                fixture = self.baseline()
                fixture[0]["runs"][0][field] = value
                self.rejects_workers(fixture)
        native, launch, clocks = native_shape()
        self.assertIs(self.m.validate_native(native, launch, **clocks), True)
        for field, value in (("native_exit", 1), ("timed_out", True)):
            with self.subTest(native_field=field):
                bad = dict(native, **{field: value})
                with self.assertRaises(SEMANTIC_ERRORS):
                    self.m.validate_native(bad, launch, **clocks)
        self.no_publication()

    def test_w04_runtime_threads_and_common_kernel_identity(self):
        for field, value in (("blas_threads", 2), ("julia_threads", True),
                             ("julia_version", "1.11.0"), ("architecture", "different-architecture")):
            with self.subTest(field=field):
                fixture = self.baseline()
                fixture[1]["smoke"][field] = value
                self.rejects_workers(fixture)

    def test_w05_current_source_and_worker_binding(self):
        fixture = self.baseline()
        fixture[1]["smoke"]["source_sha256"]["DstIntervalTraining.jl"] = "0" * 64
        self.rejects_workers(fixture)
        fixture = self.baseline()
        fixture[1]["smoke"]["worker_sha256"] = "f" * 64
        self.rejects_workers(fixture)
        self.assertEqual(self.m.IMF_PINS, IMF_PINS)
        for member, pin in PINS.items():
            self.assertEqual(self.m.PINS[member], pin)

    def test_w06_raw_marker_hash_and_warning_join(self):
        for field, value in (("sha256", "f" * 64), ("complete_marker", None), ("unsafe_runtime_warning", True)):
            with self.subTest(field=field):
                fixture = self.baseline()
                fixture[2]["smoke"][field] = value
                self.rejects_workers(fixture)

    def test_w07_only_fixed_argv_and_original_literal_driver(self):
        julia = self.directory / "unexecuted Julia binary"
        expected = [str(julia), "--startup-file=no", "--pkgimages=no", "--threads=1",
                    "--project=" + str(ROOT), str(ROOT / DRIVER)]
        self.assertEqual(self.m.build_driver_argv(ROOT, julia), expected)
        # Both genuine Base.julia_cmd prefix and trailing --project separator are valid syntax.
        self.baseline("1.12.6", historical_prefix=True, trailing_project=True)
        self.baseline("1.13.1", historical_prefix=True, trailing_project=True)
        fixture = self.baseline(historical_prefix=True)
        argv = fixture[0]["runs"][0]["argv"]
        project_index = next(i for i, item in enumerate(argv) if item.startswith("--project="))
        argv[project_index] = "--project=" + str(self.directory / "wrong-project")
        self.rejects_workers(fixture)
        extra = self.baseline(historical_prefix=True)
        extra[0]["runs"][0]["argv"].insert(1, "--load=unrequested-file" )
        self.rejects_workers(extra)
        driver = (ROOT / DRIVER).read_bytes()
        expression = driver.replace(b'const JOBS = ("smoke", ', b'const JOBS = (("smoke",)[1], ', 1)
        self.assertNotEqual(expression, driver)
        with self.assertRaises(SEMANTIC_ERRORS):
            self.m.parse_jobs(expression)
        self.no_publication()

    def test_w08_unknown_or_borrowed_boundary_cannot_be_closed(self):
        native, launch, clocks = native_shape()
        self.assertIs(self.m.validate_native(native, launch, **clocks), True)
        for field, value in (("ownership_boundary_closed", False), ("leader_reaped", False),
                             ("all_descendants_os_reaped", True), ("pid", True),
                             ("ownership", "borrowed process group"),
                             ("descendant_cleanup", "process group kill attempted")):
            with self.subTest(field=field):
                with self.assertRaises(SEMANTIC_ERRORS):
                    self.m.validate_native(dict(native, **{field: value}), launch, **clocks)
        wrong_launch = dict(launch, pid=424243)
        with self.assertRaises(SEMANTIC_ERRORS):
            self.m.validate_native(native, wrong_launch, **clocks)
        self.no_publication()

    def test_w09_admission_full_reserve_and_clock_boundaries(self):
        self.assertEqual(self.m.check_admission(8 * GIB, 16 * GIB),
                         dict(available_physical_bytes=8 * GIB, free_disk_bytes=16 * GIB))
        for memory, disk in ((8 * GIB - 1, 16 * GIB), (8 * GIB, 16 * GIB - 1), (True, 16 * GIB)):
            with self.subTest(memory=memory, disk=disk):
                with self.assertRaises(SEMANTIC_ERRORS):
                    self.m.check_admission(memory, disk)
        now = 1000000000000
        reserve = (3600 + 900 + 20 + 80) * 10**9
        self.assertIsNone(self.m.require_native_budget(now + reserve + 1, now_ns=now))
        for delta in (reserve, reserve - 1, 0):
            with self.subTest(delta=delta):
                with self.assertRaises(SEMANTIC_ERRORS):
                    self.m.require_native_budget(now + delta, now_ns=now)
        native, launch, clocks = native_shape()
        self.assertIs(self.m.validate_native(native, launch, **clocks), True)
        for field, value in (("wall_seconds", 3600), ("wrapper_observed_wall_seconds", -1),
                             ("wall_seconds", float("nan"))):
            with self.subTest(wall_field=field):
                with self.assertRaises(SEMANTIC_ERRORS):
                    self.m.validate_native(dict(native, **{field: value}), launch, **clocks)
        late = dict(clocks, native_deadline_ns=clocks["native_deadline_ns"] + 1)
        with self.assertRaises(SEMANTIC_ERRORS):
            self.m.validate_native(native, launch, **late)
        self.no_publication()

    def test_w10_partial_workers_and_direct_error_never_publish(self):
        fixture = self.baseline()
        fixture[1].pop("disk-fit")
        self.rejects_workers(fixture)
        owner = object()
        argv = ["unexecuted-file", "unexecuted-argument"]
        environment = {"UNQUALIFIED_ROUTING_PROBE": "1"}
        logpath = self.directory / "no-native-log"
        callback = lambda _pid: self.fail("Routing mock must not report a real native launch")
        expected_kwargs = dict(cwd=str(self.directory), environment=environment, logpath=logpath,
                               timeout=3600, on_launch=callback)
        sentinel = object()
        calls = []
        class Stage:
            @staticmethod
            def _run_owned_stage(actual_owner, actual_argv, **kwargs):
                calls.append((actual_owner, actual_argv, kwargs))
                return sentinel
        result = self.m.run_direct(owner, argv, stage_helper=Stage, **expected_kwargs)
        self.assertIs(result, sentinel)
        self.assertEqual(calls, [(owner, argv, expected_kwargs)])
        self.assertIs(calls[0][1], argv)
        class InterruptedStage:
            @staticmethod
            def _run_owned_stage(actual_owner, actual_argv, **kwargs):
                raise InterruptedError("explicit unqualified routing interruption")
        with self.assertRaisesRegex(InterruptedError, "explicit unqualified routing interruption"):
            self.m.run_direct(owner, argv, stage_helper=InterruptedStage, **expected_kwargs)
        self.assertFalse(logpath.exists())
        self.no_publication()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("Independent controls require unoptimized Python")
    output = Path(args.output_directory)
    if not output.is_absolute() or output.exists() or output.is_symlink() or output == ROOT or ROOT in output.parents:
        raise RuntimeError("Fresh external absolute output required")
    paths = [Path(__file__), ORACLE, SUBJECT, ROOT / ".github/workflows/interval-workers-qualification.yml",
             *[ROOT / member for member in PINS]]
    before = {str(path): digest(path.read_bytes()) for path in paths}
    output.mkdir()
    started = time.monotonic_ns()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Workers17Controls)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = {str(path): digest(path.read_bytes()) for path in paths}
    complete = result.wasSuccessful() and not result.skipped and result.testsRun == 10 and before == after
    record = dict(schema="interval_workers_qualification_independent_controls_v1", complete=complete,
                  unittest_methods=result.testsRun, expected_methods=10, frozen_negative_rows=10,
                  failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                  source_before=before, source_after=after, source_unchanged=before == after,
                  started_ns=started, finished_ns=time.monotonic_ns(),
                  metadata_fixtures_are_actual_worker_or_native_evidence=False,
                  routing_sentinels_are_native_results=False,
                  numerical_or_native_launches_requested=0, qualification_outputs_requested=0,
                  source_qualified=False, software_qualified=False, scientific_qualified=False)
    (output / "RESULT.json").write_bytes(json_bytes(record))
    raise SystemExit(0 if complete else 1)

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
