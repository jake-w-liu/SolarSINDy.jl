"""Independent offline evidence probes. Inert fixtures never issue qualification."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import io
import inspect
import json
import os
import re
import sys
import tempfile
import time
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
SUBJECT = ROOT / "tools/actual_b_chain/materialize_source_qualification.py"
ORACLE = HERE / "sourceq_importer_oracles_v1.json"
ORACLE_SHA = "009fc49e5496ec840e5e014ebb60e8096e726921ae191caa18a936bf70c34743"
HEAD = "69a8f1106930eb1cd3b02d6c1c488e1da9d44841"
CI_RUN = 42797906679
NATIVE_RUN = 42797153766
BRANCH = "codex/acyclic-b-platform-20261109"
PROOF_SHA = "1c32f26c06c06c7d9e17850cb4b56bb9f089eed0b4395961109c8e7153f04fa1"
FIXED = {
    "component136_names_v1.json": "5ee28ac856702e60d36dc2efb22248bca12726618133c90f57607f4a0935789e",
    "native69a8-source-proof.json": PROOF_SHA,
    "native69a8-posix-macos.zip": "cb9775aeebc2acb0b82a9e5edbb31628821e6c79546849e41073b0378697c520",
    "native69a8-anchor-macos.zip": "3781e526631985414946abfbd02f8e99cefa4c02d7854fd4e59130f08998e778",
}
PACKAGE_STEPS = ("Run julia-actions/julia-runtest@v1", "Run the research experiments")
DOC_STEPS = ("Build the manual (Julia 1.12.6)", "Build the manual (Julia 1.13.1)")
EIGHT = {f"Julia {version} / {platform}-latest" for version in ("1.12.6", "1.13.1")
         for platform in ("ubuntu", "windows", "macos")} | {"Documentation build", "Julia 1.12 deployment images"}
FULL18 = EIGHT | {"Canonical development harness", "Full-scale cross-runtime agreement"} | {
    f"Full-scale {version} / {role} / {layout}" for version in ("1.12.6", "1.13.1")
    for role in ("primary", "independent") for layout in ("ground-only", "mixed")}
PREFIX = ["Julia executable found", "Project.toml is present and nonempty", "src/ contains Julia source",
          "test/runtests.jl exists", "CODE_NOTES.md records implementation correctness context",
          "TEST_REPORT.md records test coverage, tolerances, oracles, or anti-false-test checks"]
SUFFIX = ["tests contain @testset and test assertions", "no obvious tautological tests found",
          "randomness appears to use explicit seed/RNG", "no TODO/FIXME/HACK markers in source/tests",
          "Pkg.test() passed", "examples/experiments.jl executed"]
WARNINGS = ["loose tolerances found; confirm each is justified", "paper_dst_superlearner/data directory not found",
            "paper_dst_superlearner/figs directory not found"]

def sha(body):
    return hashlib.sha256(body).hexdigest()

def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")

def fixture(name):
    path = FIXTURES / name
    raw = path.read_bytes()
    if sha(raw) != FIXED[name]:
        raise RuntimeError("Independent inert fixture changed: " + name)
    return raw

def load_subject():
    spec = importlib.util.spec_from_file_location("independent_sourceq_materializer_subject", SUBJECT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def proof_pair():
    payloads = {"Project.toml": b'name = "UnqualifiedProbe"\n',
                "src/Probe.jl": b"module Probe\nend\n",
                "test/runtests.jl": b"using Test\n@test true\n"}
    raw = {name: sha(value) for name, value in payloads.items()}
    oids = {name: hashlib.sha1(b"blob " + str(len(value)).encode("ascii") + b"\0" + value).hexdigest()
            for name, value in payloads.items()}
    linux = dict(complete=True, head="a" * 40, tree="b" * 40, source_files=3, source_sha256=raw,
                 commit_object_ids=oids, all_clean_and_checkout_filters_match=True)
    host = copy.deepcopy(linux)
    host["source_sha256"]["test/runtests.jl"] = sha(payloads["test/runtests.jl"].replace(b"\n", b"\r\n"))
    return linux, host

def harness_bytes(proof):
    names = PREFIX + ["parse ok: " + name for name in sorted(proof["source_sha256"]) if name.endswith(".jl")] + SUFFIX
    lines = ["  [PASS] " + name for name in names] + ["  [WARN] " + name for name in WARNINGS]
    lines += ["PASS: " + str(len(names)), "WARN: 3", "FAIL: 0"]
    return ("\n".join(lines) + "\n").encode("utf-8"), names

def run_record():
    return dict(id=CI_RUN, workflow_id=338156592, head_sha=HEAD, head_branch=BRANCH,
                path=".github/workflows/ci.yml", event="workflow_dispatch", run_attempt=1,
                status="completed", conclusion="success",
                repository=dict(id=1274208466, full_name="jake-w-liu/SolarSINDy.jl"))

def hosted_job(name="Julia 1.12.6 / ubuntu-latest", steps=PACKAGE_STEPS):
    return dict(id=40001, run_id=CI_RUN, run_attempt=1, head_sha=HEAD, name=name,
                status="completed", conclusion="success",
                started_at="2026-11-10T10:00:00Z", completed_at="2026-11-10T10:02:00Z",
                steps=[dict(name=step, number=i + 1, status="completed", conclusion="success",
                            started_at=f"2026-11-10T10:00:{10*i:02d}Z",
                            completed_at=f"2026-11-10T10:00:{10*i+5:02d}Z")
                       for i, step in enumerate(steps)])

def job_inventory():
    return [dict(hosted_job(name=name), id=50000 + i) for i, name in enumerate(sorted(FULL18))]

def safe_zip(path, members=None):
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in (members or {"raw/job.log": b"unqualified inert evidence\n"}).items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (0o100644 << 16)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, raw)
    raw = path.read_bytes()
    return dict(id=77, name="unqualified-parser-probe", size_in_bytes=len(raw), digest="sha256:" + sha(raw),
                expired=False, workflow_run=dict(id=CI_RUN, head_sha=HEAD, head_branch=BRANCH),
                archive_download_url=f"https://api.github.com/repos/jake-w-liu/SolarSINDy.jl/actions/artifacts/77/zip")

def rewrite_owner(path, transform=None, omit=None):
    """Rewrite data in memory, not by extracting or executing any archive member."""
    with zipfile.ZipFile(io.BytesIO(fixture("native69a8-posix-macos.zip"))) as source:
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as output:
            for entry in source.infolist():
                if entry.filename == omit:
                    continue
                if entry.file_size > 4 * 1024**2:
                    raise RuntimeError("Independent fixture member exceeds bound")
                raw = source.read(entry)
                if transform is not None:
                    raw = transform(entry.filename, raw)
                output.writestr(entry, raw)

POLICY = "666f9e8112c4ee7747e2121e4cb3b5ed38cf081484468e4cc2399bea271b8c94"

def workload_projections():
    """Literal checked-summary grammar only; no raw execution is asserted."""
    reports = {}
    for runtime in ("1.12.6", "1.13.1"):
        for role in ("primary", "independent"):
            for layout in ("ground-only", "mixed"):
                name = f"full-scale-{runtime}-{role}-{layout}"
                reports[name] = dict(complete=True, mode="workload", runtime=runtime, role=role,
                                     layout=layout, head=HEAD, artifact_sha256=sha(("artifact " + name).encode()),
                                     terminal_sha256=sha(("terminal " + name).encode()), source_proof_sha256=PROOF_SHA)
    return reports

def agreement_projection():
    rows = []
    for layout in ("ground-only", "mixed"):
        for other in (2, 3, 4):
            for index in range(1, 9):
                counts = (dict(scheduled=525600, features_accepted=525569, labels_accepted=525570, supervised=525539)
                          if index in (1, 5) else
                          dict(scheduled=44640, features_accepted=44609, labels_accepted=44610, supervised=44579))
                result = dict(complete=True, policy_sha256=POLICY, **counts,
                              source_provenance_qualified=False, actual_source_authority_granted=False,
                              actual_fit_performed=False, maximum_fraction_of_bound=0.25)
                rows.append(dict(layout=layout, other=other, index=index, result=result))
    return dict(complete=True, policy_sha256=POLICY, table_comparisons=48, rows_compared=7914240,
                peak_native_process_RSS_bytes=1048576, actual_source_access=False, actual_fits=0, reports=rows)

def component_projection():
    names = json.loads(fixture("component136_names_v1.json"))["names"]
    original = json.loads(fixture("native69a8-source-proof.json"))
    sources = {name: original["source_sha256"][name] for name in names}
    counts = dict(PASS=697, WARN=3, FAIL=0)
    root_sha = sha(b"unqualified structural root document")
    harness_sha = sha(b"unqualified structural harness payload")
    def cell(name):
        return dict(name=name, conclusion="success", receipt_sha256=sha(("unqualified receipt " + name).encode()),
                    log_sha256=sha(("unqualified log " + name).encode()))
    document = dict(schema="b_reference_component_software_qualification_v1", complete=True,
                    evidence_mode="synthetic_only", head=HEAD, source_sha256=sources,
                    harness=dict(counts=counts, all17_independent_workers_complete=True,
                                 unfiltered_pkg_test_complete=True, experiments_entry_complete=True),
                    native_jobs=[cell(name) for name in sorted(EIGHT) if " / " in name],
                    docs_and_containers=[cell("Documentation build"), cell("Julia 1.12 deployment images")],
                    root_qualification_sha256=root_sha, source_proof_sha256=PROOF_SHA,
                    canonical_harness_sha256=harness_sha, software_qualification_only=True,
                    actual_panel_fit_or_outcome_qualified=False, serving_promotion=False)
    expected = dict(mode="synthetic_only", head=HEAD, component_sources=sources, root_sha=root_sha,
                    proof_sha=PROOF_SHA, harness_sha=harness_sha, harness_counts=counts)
    return document, expected


class MaterializerOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sha(ORACLE.read_bytes()) != ORACLE_SHA:
            raise RuntimeError("Frozen oracle changed")
        cls.oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
        if len(cls.oracle["negative_rows"]) != 20 or len(cls.oracle["positive_rows"]) != 3:
            raise RuntimeError("Frozen row inventory changed")
        cls.m = load_subject()
        cls.observed_proof = json.loads(fixture("native69a8-source-proof.json"))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="sourceq-independent-")
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def rejects(self, code, fn, *args, **kwargs):
        with self.assertRaises(self.m.SourceQualificationError) as raised:
            fn(*args, **kwargs)
        self.assertEqual(raised.exception.code, code, str(raised.exception))
        self.assertFalse((self.work / "qualifications").exists())
        return raised.exception

    def unqualified(self, value):
        for field in ("source_qualified", "software_qualified", "scientific_qualified"):
            self.assertIs(value[field], False)

    def owner_positive(self):
        result = self.m.verify_owner_artifact(FIXTURES / "native69a8-posix-macos.zip",
                                               platform="macos-26-intel", proof=self.observed_proof)
        self.unqualified(result)
        return result

    def test_n01_incomplete_inventory(self):
        valid = job_inventory()
        self.assertEqual(set(self.m.verify_job_inventory(valid, FULL18, CI_RUN)), FULL18)
        self.rejects("incomplete_inventory", self.m.verify_job_inventory, valid[:-1], FULL18, CI_RUN)

    def test_n02_duplicate_or_swapped_job(self):
        valid = job_inventory()
        self.assertEqual(len(self.m.verify_job_inventory(valid, FULL18, CI_RUN)), 18)
        bad = copy.deepcopy(valid)
        bad[1]["id"] = bad[0]["id"]
        self.rejects("duplicate_or_swapped_job", self.m.verify_job_inventory, bad, FULL18, CI_RUN)

    def test_n03_wrong_original_identity(self):
        kwargs = dict(expected_head=HEAD, run_id=CI_RUN, workflow_id=338156592,
                      event="workflow_dispatch", branch=BRANCH)
        good = run_record()
        self.assertEqual(self.m.verify_run_identity(good, **kwargs), good)
        for field, value in (("head_sha", "c" * 40), ("run_attempt", 2), ("event", "push")):
            with self.subTest(field=field):
                bad = dict(good, **{field: value})
                self.rejects("wrong_original_identity", self.m.verify_run_identity, bad, **kwargs)

    def test_n04_unsuccessful_hosted_command(self):
        good = hosted_job()
        self.unqualified(self.m.verify_hosted_job(run_record(), good, b"actual command transcript\n", PACKAGE_STEPS))
        bad = copy.deepcopy(good)
        bad["steps"][0]["conclusion"] = "failure"
        self.rejects("unsuccessful_hosted_command", self.m.verify_hosted_job,
                     run_record(), bad, b"command failed\n", PACKAGE_STEPS)

    def test_n05_nonpositive_owned_stage(self):
        self.owner_positive()
        def changed(name, raw):
            if name == "native-owner.json":
                data = json.loads(raw)
                data["native"]["native_exit"] = 1
                return json_bytes(data)
            return raw
        path = self.work / "nonzero-owner.zip"
        rewrite_owner(path, changed)
        self.rejects("nonpositive_owned_stage", self.m.verify_owner_artifact,
                     path, platform="macos-26-intel", proof=self.observed_proof)

    def test_n06_missing_second_strict_docs(self):
        good = hosted_job("Documentation build", DOC_STEPS)
        self.unqualified(self.m.verify_hosted_job(run_record(), good, b"both docs successful\n", DOC_STEPS))
        bad = copy.deepcopy(good)
        bad["steps"].pop()
        self.rejects("missing_second_strict_docs", self.m.verify_hosted_job,
                     run_record(), bad, b"one manual only\n", DOC_STEPS)

    def test_n07_filtered_or_incomplete_execution(self):
        good = hosted_job()
        self.unqualified(self.m.verify_hosted_job(run_record(), good, b"Pkg and experiments\n", PACKAGE_STEPS))
        bad = copy.deepcopy(good)
        bad["steps"].pop()
        self.rejects("filtered_or_incomplete_execution", self.m.verify_hosted_job,
                     run_record(), bad, b"partial Pkg-only transcript\n", PACKAGE_STEPS)

    def test_n08_wrong_harness_names(self):
        proof, _ = proof_pair()
        good, names = harness_bytes(proof)
        result = self.m.verify_harness_log(good, proof)
        self.assertEqual(result["counts"], dict(PASS=len(names), WARN=3, FAIL=0))
        self.unqualified(result)
        bad = good.replace(b"parse ok: src/Probe.jl", b"parse ok: src/Other.jl")
        self.rejects("wrong_harness_names", self.m.verify_harness_log, bad, proof)

    def test_n09_warning_or_counter_counterfeit(self):
        proof, _ = proof_pair()
        good, _ = harness_bytes(proof)
        self.m.verify_harness_log(good, proof)
        for bad in (good.replace(WARNINGS[1].encode(), b"unverified replacement warning"),
                    good.replace(b"FAIL: 0\n", b"FAIL: 0\nFAIL: 0\n")):
            with self.subTest(raw_sha=sha(bad)):
                self.rejects("harness_warning_or_counter_counterfeit", self.m.verify_harness_log, bad, proof)

    def test_n10_resource_workload_gap(self):
        workloads = workload_projections()
        self.unqualified(self.m.verify_workload_inventory(workloads))
        bad = copy.deepcopy(workloads)
        bad.pop(next(iter(bad)))
        self.rejects("resource_workload_gap", self.m.verify_workload_inventory, bad)

    def test_n11_agreement48_gap(self):
        good = agreement_projection()
        self.unqualified(self.m.verify_agreement_report(good, policy_sha256=POLICY))
        bad = copy.deepcopy(good)
        bad["reports"].pop()
        self.rejects("agreement48_gap", self.m.verify_agreement_report, bad, policy_sha256=POLICY)

    def test_n12_missing_current_owner_evidence(self):
        self.owner_positive()
        path = self.work / "missing-state.zip"
        rewrite_owner(path, omit="state/RESULT.json")
        self.rejects("missing_current_owner_evidence", self.m.verify_owner_artifact,
                     path, platform="macos-26-intel", proof=self.observed_proof)

    def test_n13_cross_host_source_proof_gap(self):
        linux, host = proof_pair()
        result = self.m.verify_proof_join(linux, host, None)
        self.assertIs(result["host_checkout_verified"], False)
        self.unqualified(result)
        bad = copy.deepcopy(host)
        bad["source_sha256"]["src/Probe.jl"] = sha(b"arbitrary rewritten source\n")
        self.rejects("cross_host_source_proof_gap", self.m.verify_proof_join, linux, bad, None)

    def test_n14_component_closure_mismatch(self):
        raw = (ROOT / "tools/actual_b_chain/BMatchedReference342.jl").read_bytes()
        names = tuple(json.loads(fixture("component136_names_v1.json"))["names"])
        self.assertEqual(self.m.parse_component_members(raw), names)
        for bad in (raw.replace(('    "' + names[0] + '",').encode(), b"", 1),
                    raw.replace(names[1].encode(), names[0].encode(), 1)):
            with self.subTest(source_sha=sha(bad)):
                self.assertNotEqual(bad, raw)
                self.rejects("component_closure_mismatch", self.m.parse_component_members, bad)

    def test_n15_archive_identity_or_namespace_failure(self):
        good = self.work / "ordinary.zip"
        api = safe_zip(good)
        self.unqualified(self.m.verify_archive(good, api_record=api, run_id=CI_RUN, expected_head=HEAD))
        for name in ("../escape.log", "raw/CON.txt", "raw/case.log"):
            with self.subTest(name=name):
                path = self.work / ("bad-" + sha(name.encode()) + ".zip")
                members = {name: b"inert\n"}
                if name == "raw/case.log":
                    members["raw/CASE.log"] = b"case alias\n"
                bad_api = safe_zip(path, members)
                self.rejects("archive_identity_or_namespace_failure", self.m.verify_archive,
                             path, api_record=bad_api, run_id=CI_RUN, expected_head=HEAD)

    def test_n16_host_local_path_or_chain_rewrite(self):
        source = self.work / "ordinary.bin"
        source.write_bytes(b"unchanged original evidence\n")
        copy_path = self.work / "good-copy.bin"
        self.m.copy_verified_file(source, copy_path, expected_sha256=sha(source.read_bytes()), expected_bytes=source.stat().st_size)
        self.assertEqual(copy_path.read_bytes(), source.read_bytes())
        alias = self.work / "hardlink.bin"
        os.link(source, alias)
        self.assertGreater(source.stat().st_nlink, 1)
        self.rejects("host_local_path_or_chain_rewrite", self.m.copy_verified_file,
                     source, self.work / "alias-copy.bin", expected_sha256=sha(source.read_bytes()), expected_bytes=source.stat().st_size)

    def test_n17_coherent_header_raw_counterfeit(self):
        self.owner_positive()
        def changed(name, raw):
            if name == "state/RESULT.json":
                data = json.loads(raw)
                data["complete"] = True
                data["rows"][0]["passed"] = False
                return json_bytes(data)
            return raw
        path = self.work / "counterfeit-state.zip"
        rewrite_owner(path, changed)
        self.rejects("coherent_header_raw_counterfeit", self.m.verify_owner_artifact,
                     path, platform="macos-26-intel", proof=self.observed_proof)

    def test_n18_fabricated_native_job_facts(self):
        good = hosted_job()
        baseline = self.m.verify_hosted_job(run_record(), good, b"successful hosted commands\n", PACKAGE_STEPS)
        self.unqualified(baseline)
        forged = dict(good, pid=12345, leader_reaped=True, ownership_boundary_closed=True)
        self.rejects("fabricated_native_job_facts", self.m.verify_hosted_job,
                     run_record(), forged, b"successful hosted commands\n", PACKAGE_STEPS)
        for key in ("pid", "leader_reaped", "ownership_boundary_closed", "native", "wait_observed"):
            self.assertNotIn(key, baseline)
        self.assertEqual(baseline["exit_projection"], dict(native_exit=0, timed_out=False,
                                                        basis="github_job_and_required_steps_success"))
        component, expected = component_projection()
        self.unqualified(self.m.verify_component_document(component, **expected))
        component["native_jobs"][0]["pid"] = 12345
        self.rejects("fabricated_native_job_facts", self.m.verify_component_document, component, **expected)

    def test_n19_mode_or_authority_escalation(self):
        doc, expected = component_projection()
        self.unqualified(self.m.verify_component_document(doc, **expected))
        wrong = copy.deepcopy(doc)
        wrong["evidence_mode"] = "software_qualified"
        self.rejects("mode_or_authority_escalation", self.m.verify_component_document, wrong, **expected)
        wrong = copy.deepcopy(doc)
        wrong["serving_promotion"] = True
        self.rejects("mode_or_authority_escalation", self.m.verify_component_document, wrong, **expected)

    def test_n20_partial_or_late_publication(self):
        source = self.work / "large-ordinary.bin"
        source.write_bytes(b"unqualified engineering bytes\n" * 10000)
        expected = sha(source.read_bytes())
        baseline_target = self.work / "complete-copy.bin"
        self.m.copy_verified_file(source, baseline_target, expected_sha256=expected, expected_bytes=source.stat().st_size)
        self.assertEqual(baseline_target.read_bytes(), source.read_bytes())
        target = self.work / "partial-copy.bin"
        real_open = Path.open
        observed = {"read_calls": 0}
        class ReadFault:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.stream.close()
            def read(self, size=-1):
                observed["read_calls"] += 1
                if observed["read_calls"] > 1:
                    raise OSError("independent read fault after one retained block")
                return self.stream.read(size)
            def __getattr__(self, name):
                return getattr(self.stream, name)
        def faulting_open(path, mode="r", *args, **kwargs):
            stream = real_open(path, mode, *args, **kwargs)
            if path == source and "r" in mode and "b" in mode:
                return ReadFault(stream)
            return stream
        with mock.patch.object(Path, "open", faulting_open):
            self.rejects("partial_or_late_publication", self.m.copy_verified_file,
                         source, target, expected_sha256=expected, expected_bytes=source.stat().st_size)
        self.assertGreaterEqual(observed["read_calls"], 2)
        self.assertFalse(target.exists())
        self.assertFalse((self.work / "qualifications").exists())

    def test_p01_unqualified_metadata_composition(self):
        parameters = inspect.signature(self.m.materialize).parameters
        self.assertNotIn("trusted", parameters)
        self.assertNotIn("verified", parameters)
        self.assertNotIn("allow_fixture", parameters)
        self.assertFalse(any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()))
        names = tuple(json.loads(fixture("component136_names_v1.json"))["names"])
        raw = (ROOT / "tools/actual_b_chain/BMatchedReference342.jl").read_bytes()
        self.assertEqual(self.m.parse_component_members(raw), names)
        self.unqualified(self.m.verify_hosted_job(run_record(), hosted_job(), b"ordinary raw bytes\n", PACKAGE_STEPS))
        proof, _ = proof_pair()
        text, expected = harness_bytes(proof)
        result = self.m.verify_harness_log(text, proof)
        self.assertEqual(result["counts"], dict(PASS=len(expected), WARN=3, FAIL=0))
        self.unqualified(result)

    def test_p02_portable_proof_join(self):
        linux, host = proof_pair()
        self.assertNotEqual(linux["source_sha256"]["test/runtests.jl"], host["source_sha256"]["test/runtests.jl"])
        result = self.m.verify_proof_join(linux, host, None)
        self.assertIs(result["host_checkout_verified"], False)
        self.unqualified(result)

    def test_p03_fresh_host_local_materialization(self):
        source = self.work / "original.log"
        source.write_bytes(b"original physical hosted raw bytes\r\n")
        target = self.work / "host-copy.log"
        result = self.m.copy_verified_file(source, target, expected_sha256=sha(source.read_bytes()), expected_bytes=source.stat().st_size)
        self.assertEqual(target.read_bytes(), source.read_bytes())
        self.assertEqual(set(result), {"path", "sha256", "bytes"})
        self.assertEqual(result["sha256"], sha(source.read_bytes()))
        original = dict(path=str(source), sha256=sha(source.read_bytes()))
        relocated = dict(path=str(target), sha256=sha(target.read_bytes()))
        self.assertEqual(original["sha256"], relocated["sha256"])
        self.assertNotEqual(sha(json_bytes(original)), sha(json_bytes(relocated)))
        self.assertFalse((self.work / "qualifications").exists())
        manifest = self.work / "unqualified-manifest.json"
        manifest.write_bytes(b"{}\n")
        output = self.work / "rejected-materialization"
        self.rejects("input_error", self.m.materialize, manifest,
                     manifest_sha256="0" * 64, evidence_root=self.work, expected_head=HEAD,
                     ci_run=CI_RUN, native_run=NATIVE_RUN, checkout=ROOT, output_directory=output)
        self.assertFalse((output / "qualifications").exists())

    def test_u01_actual17_jobs_plus_native_success_remains_unready(self):
        recorded = self.oracle["actual_unready_counterexample"]
        self.assertEqual((recorded["CI_jobs_present"], recorded["native_jobs_completed_success"]), (17, 3))
        jobs = [job for job in job_inventory() if job["name"] != recorded["missing_logical_job"]]
        self.assertEqual(len(jobs), 17)
        self.rejects("incomplete_inventory", self.m.verify_job_inventory, jobs, FULL18, CI_RUN)
        self.assertFalse((self.work / "qualifications").exists())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("Independent source qualification tests require unoptimized Python")
    output = Path(args.output_directory)
    if not output.is_absolute() or output.exists() or output.is_symlink() or ROOT == output or ROOT in output.parents:
        raise RuntimeError("Fresh external absolute output required")
    protected = [Path(__file__), ORACLE, SUBJECT, SUBJECT.with_name("sourceq_full_scale_verifier.py"), ROOT / "tools/actual_b_chain/BMatchedReference342.jl", *[FIXTURES / name for name in FIXED]]
    before = {str(path): sha(path.read_bytes()) for path in protected}
    output.mkdir()
    started = time.monotonic_ns()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MaterializerOracleTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = {str(path): sha(path.read_bytes()) for path in protected}
    complete = result.wasSuccessful() and not result.skipped and result.testsRun == 24 and before == after
    record = dict(schema="matched342_sourceq_independent_tests_v1", complete=complete,
                  unittest_methods=result.testsRun, expected_methods=24,
                  frozen_negative_rows=20, frozen_metadata_positive_rows=3,
                  frozen_actual_unready_rows=1, source_before=before, source_after=after,
                  source_unchanged=before == after, started_ns=started, finished_ns=time.monotonic_ns(),
                  failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                  tests_issued_qualification_outputs=0, numerical_or_POSIX_fixture_launches_requested=0,
                  original_inert_archives_executed_or_extracted=False,
                  source_qualified=False, software_qualified=False, scientific_qualified=False)
    (output / "RESULT.json").write_bytes(json_bytes(record))
    raise SystemExit(0 if complete else 1)

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
