"""Future repository-integrated native owner; this draft has not been executed.

Python handles process/evidence metadata only. All RNG, matrix writing and
independent numerical/byte verification are Julia code in the owned child.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
import tomllib

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
MEMORY_LIMIT = 8 * 1024**3
OWNER_RELATIVE = "research/dbdt_completed_source/native_stage.py"
OWNER_SHA = "15d2644ae5eb2e87ce31c9913f103f9fa7e940b67403b94f29ef2ed5a4d175b8"
MEMORY_RELATIVE = "tools/run_canonical_harness.py"
MEMORY_SHA = "e68725a111494996ab848aa5b1a56b4d840e0900480fe7354dcd6cb1577c9a64"
INPUT_SHA = "14c7d83be58729868e8c86f949cbdc541f54f68f7891e07c3c569da91f9ff760"
SOURCES = (
    "bootstrap_indices.jl", "bootstrap_entry.jl", "test_bootstrap_indices.jl",
    "test_native_generated_artifact.jl", "frozen_population_years.toml",
    "run_native_bootstrap.py",
)


def need(value, message):
    if not value:
        raise ValueError(message)


def ordinary(path, *, directory=False):
    info = path.lstat()
    need(not (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400),
         f"Reparse/link entry refused: {path}")
    need(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode),
         f"Unexpected file type: {path}")


def ordinary_ancestors(path):
    for entry in (path, *path.parents):
        ordinary(entry, directory=True)


def bounded(path, limit):
    ordinary(path)
    before = path.stat()
    need(before.st_size <= limit, f"Byte bound exceeded: {path.name}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    after = path.stat()
    need(len(data) == before.st_size == after.st_size and len(data) <= limit,
         "Input length changed or exceeded limit")
    need((before.st_mtime_ns, before.st_ctime_ns) == (after.st_mtime_ns, after.st_ctime_ns),
         "Input metadata changed during read")
    return data


def digest(data):
    return hashlib.sha256(data).hexdigest()


def record(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
    need(bounded(path, len(data)) == data, "Receipt readback mismatch")


def git(root, *arguments):
    result = subprocess.run(["git", "-C", str(root), *arguments], check=True,
                            capture_output=True, timeout=30)
    need(len(result.stdout) <= 4 * 1024**2, "Git metadata exceeded bound")
    return result.stdout


def source_snapshot(root, expected_head):
    need(git(root, "rev-parse", "HEAD").decode().strip() == expected_head,
         "Checkout HEAD differs from requested source")
    need(not git(root, "status", "--porcelain=v1", "--untracked-files=normal"),
         "Integrated checkout must be clean")
    files = [HERE / name for name in SOURCES] + [root / OWNER_RELATIVE, root / MEMORY_RELATIVE]
    snapshot = {}
    for path in files:
        ordinary_ancestors(path.parent)
        relative = path.relative_to(root).as_posix()
        content = bounded(path, 1024**2)
        need(git(root, "cat-file", "blob", f"HEAD:{relative}") == content,
             f"Integrated source is not HEAD bytes: {relative}")
        snapshot[relative] = digest(content)
    need(snapshot[OWNER_RELATIVE] == OWNER_SHA and snapshot[MEMORY_RELATIVE] == MEMORY_SHA,
         "Repository-native owner/memory helper pins differ; independent re-review required")
    need(digest(bounded(HERE / "frozen_population_years.toml", 262144)) == INPUT_SHA,
         "Frozen551 event/year derivative differs")
    return snapshot


def load_pinned(name, path, expected_sha):
    need(digest(bounded(path, 1024**2)) == expected_sha, f"Pinned helper changed: {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, "Cannot load pinned helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def primary_reference(path, source_head, generator_sha):
    ordinary_ancestors(path)
    terminal = json.loads(bounded(path / "terminal.json", 65536))
    need(terminal.get("complete") is True and terminal.get("native_exit") == 0
         and terminal.get("timed_out") is False and terminal.get("role") == "primary"
         and terminal.get("runtime") == "1.12.6"
         and terminal.get("source_head") == source_head, "Primary terminal is not successful/pinned")
    result = path / "result"
    ordinary_ancestors(result)
    manifest_bytes = bounded(result / "INDEX_MANIFEST.toml", 65536)
    manifest = tomllib.loads(manifest_bytes.decode("utf-8"))
    matrix = bounded(result / "BOOTSTRAP_INDICES.csv", 2 * 1024**2)
    need(digest(manifest_bytes) == terminal.get("manifest_sha256")
         and digest(matrix) == terminal.get("matrix_sha256"), "Primary receipt byte pins differ")
    need(manifest.get("generator_source_sha256") == generator_sha
         and manifest.get("claimed_source_head") == source_head
         and manifest.get("role") == "primary" and manifest.get("runtime") == "1.12.6",
         "Primary source/role/runtime differs")
    need(manifest.get("source_authority_granted") is False
         and manifest.get("scientific_qualification") is False
         and manifest.get("actual_GSM_values") == 0, "Primary authority flags differ")
    return result, terminal["matrix_sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--runtime", choices=("1.12.6", "1.13.1"), required=True)
    parser.add_argument("--role", choices=("primary", "control"), required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--primary-evidence-dir", type=Path)
    args = parser.parse_args()
    need(re.fullmatch(r"[0-9a-f]{40}", args.source_head), "Source HEAD must be40hex")
    need((args.role == "control") == (args.primary_evidence_dir is not None),
         "Exactly controls require a primary evidence directory")
    need(args.role != "primary" or args.runtime == "1.12.6", "Primary runtime is frozen")
    root = args.project_root.absolute()
    ordinary_ancestors(root)
    need(root.resolve() == root and HERE.is_relative_to(root), "Use canonical integrated repository paths")
    out = args.evidence_dir.absolute()
    need(not out.is_relative_to(root) and not out.exists() and not out.is_symlink(),
         "Evidence must be fresh and outside the checkout")
    ordinary_ancestors(out.parent)
    need(out.parent.resolve() == out.parent, "Evidence parent must be canonical")
    out.mkdir()
    started = time.time()
    terminal = dict(schema="cme_gsm_bootstrap_native_owner_v1", complete=False,
                    source_head=args.source_head, role=args.role, runtime=args.runtime,
                    native_timeout_seconds=900, physical_admission_bytes=MEMORY_LIMIT,
                    actual_GSM_values=0, source_authority_granted=False,
                    scientific_qualification=False, software_qualification=False,
                    actual_primary_RNG_hash_predeclared=False)
    try:
        before = source_snapshot(root, args.source_head)
        record(out / "source-before.json", before)
        native = load_pinned("cme_bootstrap_owned_native", root / OWNER_RELATIVE, OWNER_SHA)
        memory = load_pinned("cme_bootstrap_physical_memory", root / MEMORY_RELATIVE, MEMORY_SHA)
        available, evidence = memory.available_memory()
        record(out / "memory-admission.json", dict(available_physical_bytes=available,
               required_bytes=MEMORY_LIMIT, source=evidence, admitted=available >= MEMORY_LIMIT))
        need(available >= MEMORY_LIMIT, "Available physical RAM is below8GiB; Julia was not launched")
        julia = shutil.which("julia")
        need(julia is not None, "Native Julia executable missing")
        primary = None
        if args.primary_evidence_dir is not None:
            primary, primary_sha = primary_reference(args.primary_evidence_dir.absolute(), args.source_head,
                digest(bounded(HERE / "bootstrap_indices.jl", 1024**2)))
            terminal["primary_reference_matrix_sha256"] = primary_sha
        owned_temp = out / "owned-tmp"
        owned_temp.mkdir()
        environment = os.environ.copy()
        environment.update(JULIA_LOAD_PATH="@stdlib", JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                           OMP_NUM_THREADS="1", TMPDIR=str(owned_temp), TMP=str(owned_temp), TEMP=str(owned_temp))
        command = [julia, "--startup-file=no", "--history-file=no", "--pkgimages=no", "--threads=1",
                   str(HERE / "bootstrap_entry.jl"), "--output", str(out / "result"),
                   "--role", args.role, "--source-head", args.source_head, "--runtime", args.runtime]
        if primary is not None:
            command += ["--primary", str(primary)]
        record(out / "command.json", dict(argv=command, timeout_seconds=900,
               ownership_helper_sha256=OWNER_SHA, no_Pkg_install=True, Python_RNG_generation=False))
        outcome = native.run_owned(command, cwd=root, environment=environment,
                    logpath=out / "native.log", timeout=900,
                    on_launch=lambda pid: record(out / "launch.json", dict(pid=pid, owner_pid=os.getpid())))
        record(out / "native.json", outcome)
        terminal.update(outcome)
        raw = bounded(out / "native.log", 8 * 1024**2)
        terminal["native_raw_sha256"] = digest(raw)
        need(outcome["native_exit"] == 0 and outcome["timed_out"] is False,
             "Owned native Julia failed or timed out")
        for marker in (b"BOOTSTRAP_LITERAL_AND_POPULATION_ORACLES_PASSED;ACTUAL_RNG_HASH_NOT_PREDECLARED",
                       b"BOOTSTRAP_NATIVE_RNG_AND_ARTIFACT_TAMPER_ORACLES_PASSED",
                       b"BOOTSTRAP_NATIVE_GENERATION_COMPLETE;"):
            need(raw.count(marker) == 1, "Expected native completion/oracle marker missing or duplicate")
        manifest_bytes = bounded(out / "result" / "INDEX_MANIFEST.toml", 65536)
        manifest = tomllib.loads(manifest_bytes.decode("utf-8"))
        matrix = bounded(out / "result" / "BOOTSTRAP_INDICES.csv", 2 * 1024**2)
        need(manifest["matrix_sha256"] == digest(matrix) and manifest["runtime"] == args.runtime
             and manifest["role"] == args.role and manifest["claimed_source_head"] == args.source_head,
             "Native generated artifact identities differ")
        resource = tomllib.loads(bounded(out / "result.resource.toml", 4096).decode("utf-8"))
        need(resource["runtime"] == args.runtime and resource["RSS_limit_bytes"] == MEMORY_LIMIT
             and 0 < resource["peak_Julia_process_RSS_bytes"] <= MEMORY_LIMIT,
             "Native Julia measured peak RSS is missing or exceeds8GiB")
        if primary is not None:
            comparison = tomllib.loads(bounded(out / "result.comparison.toml", 4096).decode("utf-8"))
            need(comparison["same_bytes"] is True and comparison["primary_replaced"] is False
                 and comparison["primary_matrix_sha256"] == terminal["primary_reference_matrix_sha256"]
                 and comparison["control_matrix_sha256"] == digest(matrix), "Control byte agreement failed")
        terminal.update(matrix_sha256=digest(matrix), manifest_sha256=digest(manifest_bytes),
                        peak_Julia_process_RSS_bytes=resource["peak_Julia_process_RSS_bytes"])
        terminal["complete"] = True
    except Exception as error:
        terminal["error_type"] = type(error).__name__
        terminal["error"] = str(error)
    finally:
        try:
            after = source_snapshot(root, args.source_head)
            record(out / "source-after.json", after)
            need("before" in locals() and after == before, "Integrated source changed during native stage")
        except Exception as error:
            terminal["complete"] = False
            terminal["source_closure_error"] = f"{type(error).__name__}: {error}"
        terminal["elapsed_seconds"] = time.time() - started
        terminal["partial_outputs_retained"] = True
        terminal["peak_RSS_is_post_execution_check_not_OS_memory_cap"] = True
        record(out / "terminal.json", terminal)
    print(json.dumps(terminal, sort_keys=True))
    return 0 if terminal["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
