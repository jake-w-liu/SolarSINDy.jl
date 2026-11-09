"""Fixed bounded B1 generated controls; no Julia or actual source admission."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).with_name('test_driver.py')
OWNER = ROOT / 'research/dbdt_completed_source/candidate_native_owner.py'
OWNER_SHA = 'ccbf1cbd0709e50d0b5080bdbe4977ae80ebd1649237e016588d24800ddf2e21'
SCRIPT_SHA = '329d86a7f35c245d0ba096b85e2bb5f3c15c08a4b00f9c0d9523b389c07179eb'
SECONDS = 180
METHODS = 24
NAMES = (
    'test/b_initial_native/fixed_child.py', 'test/b_initial_native/literal_oracles_v1.json',
    'test/b_initial_native/api_oracles_v2.json', 'test/b_initial_native/closure_oracles_v3.json',
    'test/b_initial_native_tests.jl',
    'examples/b_initial_l0_native_synthetic.jl',
    'tools/actual_b_chain/run_initial_l0_native.py',
    'tools/actual_b_chain/run_initial_l0_native.jl',
    'tools/actual_b_chain/BInitialNativePermitV2.jl',
    'tools/actual_b_chain/BInitialPointFits.jl', 'src/duration_point_training.jl',
    'research/dbdt_completed_source/native_stage.py',
    'research/dbdt_completed_source/run_candidate_stage.py',
    'tools/run_canonical_harness.py', 'examples/support/b_point_generated_fixture.jl',
    'test/b_point_training/tree_inventory_oracles_v1.json',
    'test/b_point_training/independent_literals_v1.json')


def ordinary(path, directory=False):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
        raise ValueError('ordinary controls required; links/reparse forbidden')
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError('ordinary control kind differs')


def pin(path):
    ordinary(path)
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, record):
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def validate_summary(raw):
    if not isinstance(raw, bytes) or len(raw) > 4 * 1024**2:
        raise ValueError('bounded native transcript required')
    if re.findall(rb'^Ran (\d+) tests in [^\r\n]+\r?$', raw, re.M) != [b'24'] or \
            re.findall(rb'^(OK)\r?$', raw, re.M) != [b'OK']:
        raise ValueError('exact24-method LF/CRLF transcript required')
    if re.search(rb'^(?:FAILED\b|FAIL:|ERROR:|SKIPPED\b)', raw, re.M):
        raise ValueError('failed/skipped native controls rejected')


def main(output):
    if sys.flags.optimize or sys.version_info < (3, 11):
        raise RuntimeError('unoptimized Python>=3.11 required')
    if sys.platform == 'darwin' and sys.version_info < (3, 14):
        raise RuntimeError('macOS Python>=3.14 required; no ownership skip')
    if not output.is_absolute() or os.path.lexists(output):
        raise ValueError('fresh absolute output required')
    if output.parent.resolve(strict=True) != output.parent:
        raise ValueError('canonical output parent required')
    if output.is_relative_to(ROOT) or ROOT.is_relative_to(output):
        raise ValueError('output/source overlap forbidden')
    for path in (ROOT, *ROOT.parents, output.parent, *output.parent.parents):
        ordinary(path, directory=True)
    if pin(SCRIPT) != SCRIPT_SHA or pin(OWNER) != OWNER_SHA:
        raise RuntimeError('fixed test/owner source changed')
    output.mkdir()
    fixtures = output / 'fixtures'
    fixtures.mkdir()
    executable = Path(sys.executable).resolve(strict=True)
    controls = {str(path): pin(path) for path in
                (Path(__file__), SCRIPT, OWNER, executable, *(ROOT / n for n in NAMES))}
    record = dict(schema='b1_generated_native_controls_v1', complete=False,
                  methods_expected=24, controls_before_sha256=controls,
                  Julia_executed=False, actual_observations=0,
                  source_authority_granted=False, scientific_qualification=False,
                  raw_positive_acceptance_bound_bytes=4 * 1024**2,
                  raw_bound_is_live_quota=False,
                  fixture_scope='Generated metadata and fixed owned Python children only')
    try:
        spec = importlib.util.spec_from_file_location('b1_controls_genuine_native_owner', OWNER)
        owner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(owner)
        if os.name == 'posix':
            owner.posix_features()
        native = owner.run_owned(
            [sys.executable, '-B', '-X', 'utf8', str(SCRIPT)], cwd=str(ROOT),
            environment=dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                             TMP=str(fixtures), TEMP=str(fixtures), TMPDIR=str(fixtures)),
            logpath=output / 'native.log', timeout=SECONDS)
        record.update(native=native, raw_sha256=pin(output / 'native.log'))
        if type(native['native_exit']) is not int or native['native_exit'] != 0 or \
                native['timed_out'] is not False or native.get('leader_reaped') is not True or \
                native.get('ownership_boundary_closed') is not True:
            raise RuntimeError('B1 generated native controls failed or timed out')
        with (output / 'native.log').open('rb') as stream:
            raw = stream.read(4 * 1024**2 + 1)
        validate_summary(raw)
        record.update(complete=True, methods=24)
    except BaseException as error:
        record['failure'] = dict(type=type(error).__name__, message=str(error),
                                 traceback=traceback.format_exc())
    finally:
        after = {}
        for path in controls:
            try:
                after[path] = pin(Path(path))
            except BaseException:
                after[path] = None
        changed = [p for p, sha in controls.items() if after[p] != sha]
        record.update(controls_after_sha256=after, changed_controls=changed)
        record['complete'] = record['complete'] and not changed
        write(output / 'TERMINAL.json', record)
        print(json.dumps({k: v for k, v in record.items() if k not in
                          ('controls_before_sha256', 'controls_after_sha256')},
                         sort_keys=True), flush=True)
    return 0 if record['complete'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(main(parser.parse_args().output))
