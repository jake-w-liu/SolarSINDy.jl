"""Bounded mandatory Darwin-anchor controls; no Julia or scientific authority."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'test/darwin_anchor_v1'
OWNER = ROOT / 'research/dbdt_completed_source/candidate_native_owner.py'
MEMORY = ROOT / 'tools/run_canonical_harness.py'
MEMORY_SHA = 'defbb2a90690acd83ff0dc8eae05fd8622543871ecfa6c7a595b04069d44b331'
OWNER_PINS = {
    'tools/initial_reference_audit/owned_process.py': '30c7fc96042c3d84814f0ea728c0415d657d90c02b5de4cf516da517c7f9a21c',
    'research/dbdt_completed_source/native_stage.py': '30c7fc96042c3d84814f0ea728c0415d657d90c02b5de4cf516da517c7f9a21c',
    'research/dbdt_completed_source/candidate_native_owner.py': 'c3575532cb67b4bf52d2db8a20e01ca7e8d97cfb0cb1054ded66467823e40c00',
}
TEST_PINS = {
    'oracles_v1.json': '51b90716f70fc35fcd5541ab995e49695957a17378b2de44012617c4e6e25e97',
    'test_anchor_state.py': '99b72148874d2a103bdc003f748be7b9fd3a0f310a3de02d88b7493fd8e455fd',
    'run_anchor_native.py': 'f6a4d1a9e5667284db79bde25e2850d0be00787e14235ea6f3e5bcde45802cfb',
    'fixture.py': 'af87432ffe060ba181766c748297cc4fdedd16f8bb8fb8c6698a1f8d9b65fa0c',
    'frozen/native_stage.py': '8799458af4fe532c62726df503bb88a18385fe670485363290e45447737cc836',
    'frozen/candidate_native_owner.py': '3c1a2838a3381afe22ae69b95afc1d3b03523b569f34034604390a7122cccac1',
    'frozen/before_waitid_identity_v2.py': '97b432ace38c70c4a87ffe1ff8d6e3818e8661bd48c60705ca102468c27aee03',
}
WHOLE_SECONDS = 660
RAW_LIMIT = 4 * 1024**2
META_LIMIT = 2 * 1024**2
SOURCE_LIMIT = 256 * 1024


def need(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_pinned(path, name, expected):
    need(path.is_file() and not path.is_symlink(), 'ordinary pinned module required')
    with path.open('rb') as stream:
        body = stream.read(SOURCE_LIMIT + 1)
    need(0 < len(body) <= SOURCE_LIMIT and hashlib.sha256(body).hexdigest() == expected,
         'pinned module changed or exceeds source bound')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    need(digest(path) == expected, 'pinned module changed during import')
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
    need(0 < len(body) <= META_LIMIT, 'result grew beyond metadata bound')
    return json.loads(body, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write_new(path, value):
    body = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    need(len(body) <= META_LIMIT, 'controller metadata exceeds bound')
    with path.open('xb') as stream:
        need(stream.write(body) == len(body), 'short controller metadata write')
        stream.flush()
        os.fsync(stream.fileno())


def worker_clocks(value, seconds):
    need(all(type(value.get(key)) is int for key in ('started_ns', 'finished_ns', 'deadline_ns')),
         'native worker clocks must be integers')
    need(0 <= value['started_ns'] <= value['finished_ns'] < value['deadline_ns'] and
         value['deadline_ns'] == value['started_ns'] + seconds * 10**9,
         'original absolute worker deadline differs or expired')


def worker_admission(value):
    need(type(value.get('available_physical_bytes')) is int and
         value['available_physical_bytes'] >= 512 * 1024**2 and
         type(value.get('free_disk_bytes')) is int and value['free_disk_bytes'] >= 8 * 1024**3 and
         isinstance(value.get('memory_method'), str) and bool(value['memory_method']) and
         type(value.get('resource_probe_subprocesses')) is int and
         value['resource_probe_subprocesses'] == (1 if sys.platform == 'darwin' else 0),
         'actual worker engineering admission differs')


def main():
    started = time.monotonic_ns()
    deadline = started + WHOLE_SECONDS * 10**9
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_directory
    need(not sys.flags.optimize and sys.version_info >= (3, 11) and output.is_absolute(),
         'unoptimized Python>=3.11 and absolute evidence path required')
    need(os.name == 'nt' or sys.platform == 'darwin' or sys.platform.startswith('linux'),
         'mandatory controls require Windows, Linux or Darwin')
    need(not output.exists() and output.parent.is_dir(), 'fresh external evidence root required')
    parent = output.parent.resolve(strict=True)
    need(parent != ROOT and not parent.is_relative_to(ROOT), 'evidence must be outside checkout')
    output = parent / output.name
    python = Path(sys.executable).resolve(strict=True)
    metadata = load_pinned(TESTS / 'test_anchor_state.py', 'registered_anchor_metadata',
                           TEST_PINS['test_anchor_state.py'])
    configuration = metadata.config()
    sources = metadata.source_pins(configuration)
    expected_sources = {str(ROOT / name) for name in OWNER_PINS} | {
        str(TESTS / 'frozen/native_stage.py'), str(TESTS / 'frozen/candidate_native_owner.py'),
        str(TESTS / 'frozen/before_waitid_identity_v2.py'),
        str(ROOT / 'test/posix_owner_v3/observations.py'),
        str(ROOT / 'test/initial_reference/diagnose_darwin_groups.py'),
    }
    need(type(sources) is dict and set(sources) == expected_sources and len(sources) == 8 and
         all(Path(path).is_absolute() and isinstance(pin, str) and
             re.fullmatch('[0-9a-f]{64}', pin) is not None for path, pin in sources.items()),
         'closed eight-source anchor binding differs')
    need(all(sources[str(ROOT / name)] == pin for name, pin in OWNER_PINS.items()),
         'current V5 owner binding differs')
    need(metadata.inventory() == TEST_PINS, 'final seven-file independent anchor inventory differs')
    test_sources = dict(TEST_PINS)
    paths = [Path(__file__).resolve(), MEMORY, python, *(Path(path) for path in sources)]
    need(all(path.is_file() and not path.is_symlink() for path in paths), 'ordinary protected sources required')
    protected = {str(path): digest(path) for path in paths}
    need(protected[str(MEMORY)] == MEMORY_SHA, 'current resource reader changed')
    memory = load_pinned(MEMORY, 'registered_anchor_resources', MEMORY_SHA)

    def unchanged():
        return (all(path.is_file() and not path.is_symlink() and digest(path) == pin
                    for path, pin in ((Path(name), pin) for name, pin in protected.items())) and
                metadata.config() == configuration and metadata.source_pins(configuration) == sources and
                metadata.inventory() == test_sources)

    def admit():
        available, method = memory.available_memory()
        free = shutil.disk_usage(parent).free
        need(type(available) is int and available >= 512 * 1024**2 and free >= 8 * 1024**3,
             'actual engineering512MiB/8GiB admission failed')
        return dict(available_physical_bytes=available, memory_method=method, free_disk_bytes=free)

    admission = admit()
    need(unchanged() and time.monotonic_ns() < deadline, 'source changed or original controller clock expired')
    output.mkdir()
    record = dict(schema='independent_darwin_anchor_registered_controls_v1', complete=False,
        started_ns=started, deadline_ns=deadline, admission=admission,
        source_pins=sources, test_sources=test_sources, protected_source_sha256=protected,
        native_Darwin_applicable=sys.platform == 'darwin', Julia_launched=False,
        actual_data_exposed=False, native_Darwin_qualified=False,
        source_qualified=False, scientific_qualified=False)
    write_new(output / 'PREEXECUTION.json', record)

    def stage(name, filename, seconds, worker_seconds):
        need(unchanged(), 'source changed before owned stage')
        resource = admit()
        need(unchanged(), 'source changed during admission')
        allowance = min(seconds, (deadline - time.monotonic_ns()) / 1e9 - 20)
        need(allowance > 0, 'no original controller budget remains for owned stage')
        raw = output / (name + '.log')
        target = output / name
        launched = []
        def on_launch(pid):
            need(type(pid) is int and pid > 0 and not launched, 'one genuine stage launch required')
            launched.append(pid)
            write_new(output / (name + '-launch.json'), dict(pid=pid, monotonic_ns=time.monotonic_ns()))
        try:
            native = owner.run_owned([str(python), '-I', '-B', '-X', 'utf8', str(TESTS / filename),
                '--output-directory', str(target)], cwd=str(ROOT),
                environment=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                logpath=raw, timeout=allowance, on_launch=on_launch)
        except BaseException as error:
            write_new(output / (name + '-owner-exception.json'), dict(native_result_returned=False,
                observed_launch_pids=launched, allowance_seconds=allowance, admission=resource,
                error=dict(type=type(error).__name__, errno=getattr(error, 'errno', None), message=str(error))))
            raise
        receipt = dict(native=native, launch_pid=launched[0] if launched else None,
                       allowance_seconds=allowance, admission=resource)
        try:
            need(raw.is_file() and not raw.is_symlink(), 'ordinary native raw log required')
            receipt.update(raw_bytes=raw.stat().st_size)
            need(0 <= receipt['raw_bytes'] <= RAW_LIMIT, 'native raw transcript exceeds4MiB')
            receipt['raw_sha256'] = digest(raw)
        except BaseException as error:
            receipt['raw_error'] = dict(type=type(error).__name__, message=str(error))
            write_new(output / (name + '-owner.json'), receipt)
            raise
        write_new(output / (name + '-owner.json'), receipt)
        need(len(launched) == 1 and type(native['pid']) is int and native['pid'] == launched[0],
             'native PID does not match actual launch')
        need(type(native['native_exit']) is int and native['native_exit'] == 0 and native['timed_out'] is False and
             native['leader_reaped'] is True and native['ownership_boundary_closed'] is True and
             native['all_descendants_os_reaped'] is False, 'native stage failed, timed out or lacks observed boundary')
        need(type(native['wall_seconds']) in (int, float) and math.isfinite(native['wall_seconds']) and
             0 <= native['wall_seconds'] < allowance, 'native stage wall exceeds original allowance')
        need(unchanged() and time.monotonic_ns() < deadline, 'source or controller deadline failed after native wait')
        result = read_json(target / 'RESULT.json')
        need(result.get('complete') is True and result.get('source_pins') == sources and
             result.get('test_sources') == test_sources and result.get('source_unchanged') is True,
             'independent worker result/source identity failed')
        worker_clocks(result, worker_seconds)
        pre = read_json(target / 'PREEXECUTION.json')
        need(pre.get('source_pins') == sources and pre.get('test_sources') == test_sources and
             pre.get('started_ns') == result['started_ns'] and pre.get('deadline_ns') == result['deadline_ns'],
             'worker prelaunch/source/absolute clock identity differs')
        worker_admission(pre['admission'] if name == 'native' else pre)
        if name == 'native':
            need(result.get('admission') == pre['admission'], 'worker admission facts changed')
        receipt['result_sha256'] = digest(target / 'RESULT.json')
        need(unchanged() and time.monotonic_ns() < deadline, 'worker verification exceeded original controller clock')
        return result, receipt, raw

    previous_term = None
    def interrupted(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise InterruptedError('Darwin anchor controller received termination signal')
    try:
        if os.name == 'posix':
            previous_term = signal.signal(signal.SIGTERM, interrupted)
        owner = load_pinned(OWNER, 'registered_anchor_current_owner', OWNER_PINS['research/dbdt_completed_source/candidate_native_owner.py'])
        state, state_receipt, state_raw = stage('state', 'test_anchor_state.py', 90, 90)
        need(state.get('schema') == 'independent_darwin_anchor_literal_result_v1' and
             type(state.get('unittest_methods')) is int and type(state.get('expected_methods')) is int and
             state['unittest_methods'] == state['expected_methods'] == 3 and
             type(state.get('literal_rows')) is int and type(state.get('expected_rows')) is int and
             state['literal_rows'] == state['expected_rows'] == 72 and
             isinstance(state.get('rows'), list) and len(state['rows']) == 72 and
             all(row.get('passed') is True for row in state['rows']), 'all72 literals and3 methods required')
        with state_raw.open('rb') as stream:
            transcript = stream.read(RAW_LIMIT + 1)
        need(len(transcript) <= RAW_LIMIT and
             re.findall(rb'^Ran (\d+) tests in [^\r\n]+\r?$', transcript, re.M) == [b'3'] and
             re.findall(rb'^(OK)\r?$', transcript, re.M) == [b'OK'] and
             re.search(rb'^(?:FAILED\b|FAIL:|ERROR:|SKIPPED\b)', transcript, re.M) is None,
             'exact three-method successful native unittest transcript required')
        need(type(state.get('actual_POSIX_test_processes_started')) is int and state['actual_POSIX_test_processes_started'] == 0 and
             type(state.get('actual_signals_sent')) is int and state['actual_signals_sent'] == 0 and
             all(state.get(key) is False for key in ('native_Darwin_qualified', 'Julia_launched',
                 'actual_data_exposed', 'source_qualified', 'scientific_qualified')),
             'literal controls must not claim native or scientific qualification')
        record.update(state=state_receipt, state_methods=3, literal_rows=72)
        if record['native_Darwin_applicable']:
            native, native_receipt, _ = stage('native', 'run_anchor_native.py', 500, 480)
            need(native.get('schema') == 'independent_darwin_anchor_native_result_v1' and
                 type(native.get('actual_current_cases')) is int and native['actual_current_cases'] == 12 and
                 type(native.get('actual_historical_negative_cases')) is int and native['actual_historical_negative_cases'] == 3 and
                 type(native.get('actual_failure_cleanup_negative_cases')) is int and native['actual_failure_cleanup_negative_cases'] == 1 and
                 type(native.get('actual_native_cases')) is int and native['actual_native_cases'] == 16 and
                 len(native['current_case_passes']) == 12 and all(value is True for value in native['current_case_passes']) and
                 len(native['historical_negative_passes']) == 3 and all(value is True for value in native['historical_negative_passes']) and
                 len(native['failure_cleanup_negative_passes']) == 1 and all(value is True for value in native['failure_cleanup_negative_passes']),
                 'all12 current,3 historical and1 failure-cleanup native Darwin cases required')
            runtime = native['runtime']
            need(runtime.get('python_version') == sys.version and runtime.get('python_binary_sha256') == protected[str(python)] and
                 isinstance(runtime.get('uname'), list) and len(runtime['uname']) == 5 and runtime['uname'][0] == 'Darwin' and
                 all(isinstance(value, str) for value in runtime['uname']), 'actual native Python/kernel identity required')
            need(all(native.get(key) is False for key in ('individual_descendant_OS_reaping_observed',
                 'native_Darwin_qualified', 'Julia_launched', 'actual_data_exposed', 'source_qualified', 'scientific_qualified')),
                 'native controls cannot grant broader qualification')
            record.update(native=native_receipt, actual_current_native_cases=12,
                          historical_negative_cases=3, failure_cleanup_negative_cases=1, actual_native_cases=16)
        else:
            record.update(actual_current_native_cases=0, historical_negative_cases=0,
                failure_cleanup_negative_cases=0, actual_native_cases=0,
                native_nonapplicability='Native Darwin cases require Darwin; all72 literals remain mandatory on Windows/Linux')
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
        record['finished_ns'] = time.monotonic_ns()
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
