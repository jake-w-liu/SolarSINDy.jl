"""Bounded current-source V3 engineering controls; no Julia or scientific authority."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import signal
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'test/posix_owner_v3'
OWNER = ROOT / 'research/dbdt_completed_source/candidate_native_owner.py'
OWNER_PINS = {
    'tools/initial_reference_audit/owned_process.py': '8799458af4fe532c62726df503bb88a18385fe670485363290e45447737cc836',
    'research/dbdt_completed_source/native_stage.py': '8799458af4fe532c62726df503bb88a18385fe670485363290e45447737cc836',
    'research/dbdt_completed_source/candidate_native_owner.py': '3c1a2838a3381afe22ae69b95afc1d3b03523b569f34034604390a7122cccac1',
}
WHOLE_SECONDS = 660
RAW_LIMIT = 4 * 1024**2
META_LIMIT = 2 * 1024**2


def need(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    need(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= META_LIMIT,
         'bounded ordinary result required: ' + str(path))
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'duplicate result key')
            result[key] = value
        return result
    with path.open('rb') as stream:
        body = stream.read(META_LIMIT + 1)
    need(0 < len(body) <= META_LIMIT, 'result grew beyond bound')
    return json.loads(body, object_pairs_hook=unique, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write_new(path, value):
    body = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    need(len(body) <= META_LIMIT, 'controller metadata exceeds bound')
    with path.open('xb') as stream:
        need(stream.write(body) == len(body), 'short controller metadata write')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    started = time.monotonic_ns()
    deadline = started + WHOLE_SECONDS * 10**9
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_directory
    need(not sys.flags.optimize and output.is_absolute(), 'unoptimized file-based absolute output required')
    need(not output.exists() and output.parent.is_dir(), 'fresh external evidence root required')
    parent = output.parent.resolve(strict=True)
    need(parent != ROOT and not parent.is_relative_to(ROOT), 'evidence must be outside checkout')
    output = parent / output.name
    memory_path = ROOT / 'tools/run_canonical_harness.py'
    dependencies = [Path(__file__).resolve(), memory_path, *(ROOT / name for name in OWNER_PINS)]
    need(all(path.is_file() and not path.is_symlink() for path in dependencies), 'ordinary source dependencies required')
    protected = {str(path): digest(path) for path in dependencies}
    need(all(digest(ROOT / name) == pin for name, pin in OWNER_PINS.items()), 'reviewed owner pin changed')
    sys.path.insert(0, str(TESTS))
    observations = load(TESTS / 'observations.py', 'registered_posix_v3_observations')
    need(observations.source_pins() == OWNER_PINS, 'independent current owner bindings differ')
    test_sources = observations.test_inventory()
    memory = load(memory_path, 'registered_posix_v3_resources')

    def unchanged():
        return (all(digest(path) == pin for path, pin in protected.items()) and
                observations.source_pins() == OWNER_PINS and observations.test_inventory() == test_sources)

    def admit():
        available, method = memory.available_memory()
        free = shutil.disk_usage(parent).free
        need(available >= 512 * 1024**2 and free >= 8 * 1024**3, 'engineering512MiB/8GiB admission failed')
        return dict(available_physical_bytes=available, memory_method=method, free_disk_bytes=free)

    admission = admit()
    need(unchanged() and time.monotonic_ns() < deadline, 'source changed or controller deadline exhausted')
    output.mkdir()
    record = dict(schema='owned_posix_v3_registered_controls_v1', complete=False, started_ns=started,
        deadline_ns=deadline, admission=admission, source_pins=OWNER_PINS, test_sources=test_sources,
        Julia_launched=False, actual_archive_exposed=False, source_qualified=False, scientific_qualified=False,
        native_POSIX_applicable=os.name == 'posix' and (sys.platform.startswith('linux') or sys.platform == 'darwin'))
    write_new(output / 'PREEXECUTION.json', record)
    def stage(name, filename, seconds):
        need(unchanged(), 'source changed before stage')
        resource = admit()
        remaining = (deadline - time.monotonic_ns()) / 1e9 - 20
        allowance = min(seconds, remaining)
        need(allowance > 0, 'no budget remains for owned stage')
        raw = output / (name + '.log')
        target = output / name
        launched = []
        def on_launch(pid):
            need(type(pid) is int and pid > 0 and not launched, 'one genuine native launch required')
            launched.append(pid)
            write_new(output / (name + '-launch.json'), dict(pid=pid, monotonic_ns=time.monotonic_ns()))
        native = owner.run_owned([sys.executable, '-B', '-X', 'utf8', str(TESTS / filename),
            '--output-directory', str(target)], cwd=str(ROOT), environment=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
            logpath=raw, timeout=allowance, on_launch=on_launch)
        # Keep the actual receipt even if a later semantic or file check fails.
        receipt = dict(native=native, launch_pid=launched[0] if launched else None,
            allowance_seconds=allowance, admission=resource, raw_sha256=digest(raw), raw_bytes=raw.stat().st_size)
        write_new(output / (name + '-owner.json'), receipt)
        need(len(launched) == 1 and native['pid'] == launched[0], 'native result does not match actual launch')
        need(type(native['native_exit']) is int and native['native_exit'] == 0 and native['timed_out'] is False,
             'native stage failed or timed out')
        need(native['leader_reaped'] is True and native['ownership_boundary_closed'] is True and
             native['all_descendants_os_reaped'] is False, 'truthful native closure/reaping required')
        wall = native['wall_seconds']
        need(type(wall) in (int, float) and math.isfinite(wall) and 0 <= wall < allowance,
             'native wall exceeds stage allowance')
        need(receipt['raw_bytes'] <= RAW_LIMIT and unchanged() and time.monotonic_ns() < deadline,
             'raw bound, source or whole deadline failed')
        result = read_json(target / 'RESULT.json')
        need(result['complete'] is True, 'independent control result failed')
        receipt['result_sha256'] = digest(target / 'RESULT.json')
        return result, receipt

    previous_term = None
    def interrupted(signum, frame):
        # Unwind through run_owned's cleanup; default SIGTERM skips its finally.
        raise InterruptedError('V3 controller received termination signal')
    try:
        if os.name == 'posix':
            previous_term = signal.signal(signal.SIGTERM, interrupted)
        owner = load(OWNER, 'registered_posix_v3_current_owner')
        state, state_receipt = stage('state', 'test_state.py', 90)
        need(type(state['unittest_methods']) is int and state['unittest_methods'] == 5 and
             type(state['simulated_rows']) is int and state['simulated_rows'] == state['expected_simulated_rows'] == 85,
             'all five methods and85 independent simulated rows required')
        need(state['source_pins'] == OWNER_PINS and state['source_pins_unchanged'] is True and
             state['actual_POSIX_processes_started'] == state['actual_signals_sent'] == 0 and
             state['native_Linux_or_Darwin_qualified'] is False, 'simulated scope must remain explicit')
        need(len(state['rows']) == 85 and all(row['passed'] is True for row in state['rows']), 'simulated row failure')
        need(state['source_qualified'] is False and state['scientific_qualified'] is False,
             'simulated result cannot grant qualification')
        record.update(state=state_receipt, state_methods=5, simulated_rows=85)
        if record['native_POSIX_applicable']:
            native, native_receipt = stage('native', 'run_native.py', 500)
            historical = 3 if sys.platform == 'darwin' else 0
            need(native['actual_current_cases'] == 21 and native['principal_current_cases'] == 15 and
                 native['extra_current_cases'] == 6 and native['historical_stop_negative_cases'] == historical,
                 'all independent native current and historical cases required')
            need(len(native['case_passes']) == 21 and all(value is True for value in native['case_passes']) and
                 len(native['historical_negative_passes']) == historical and
                 all(value is True for value in native['historical_negative_passes']), 'native case failure')
            need(native['source_pins'] == OWNER_PINS and native['source_pins_unchanged'] is True and
                 native['test_sources_unchanged'] is True and native['test_source_sha256'] == test_sources and
                 type(native['elapsed_ns']) is int and 0 <= native['elapsed_ns'] < 480 * 10**9 and
                 native['individual_descendant_OS_reaping_observed'] is False and
                 native['source_qualified'] is False and native['scientific_qualified'] is False,
                 'native integrity/deadline evidence required')
            record.update(native=native_receipt, actual_current_native_cases=21, historical_negative_cases=historical)
        else:
            need(os.name == 'nt', 'unsupported native platform')
            record.update(actual_current_native_cases=0, historical_negative_cases=0,
                native_nonapplicability='Windows state controls and genuine outer Job receipt; POSIX native cases require Linux/Darwin')
        record['complete'] = True
    except BaseException as error:
        record.update(complete=False, failure=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
    finally:
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)
        try:
            record['source_unchanged'] = unchanged()
        except BaseException as error:
            record.update(source_unchanged=False, source_guard_error=repr(error))
        record.update(finished_ns=time.monotonic_ns())
        record['complete'] = record['complete'] and record['source_unchanged'] and record['finished_ns'] < deadline
        write_new(output / 'TERMINAL.json', record)
        if time.monotonic_ns() >= deadline:
            (output / 'TERMINAL.json').rename(output / 'TERMINAL_LATE_UNQUALIFIED.json')
            write_new(output / 'LATE_OUTPUT.json', dict(complete=False, deadline_ns=deadline, observed_ns=time.monotonic_ns()))
            return 1
    print(json.dumps(dict(complete=record['complete'], evidence=str(output),
        terminal_sha256=digest(output / 'TERMINAL.json'), failure=record.get('failure')), sort_keys=True), flush=True)
    if time.monotonic_ns() >= deadline:
        (output / 'TERMINAL.json').rename(output / 'TERMINAL_LATE_UNQUALIFIED.json')
        write_new(output / 'LATE_OUTPUT.json', dict(complete=False, deadline_ns=deadline, observed_ns=time.monotonic_ns()))
        return 1
    return 0 if record['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
