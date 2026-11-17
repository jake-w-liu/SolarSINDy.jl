"""Run the seven fixed offline SourceQ/interval control CLIs; grants no authority."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SOURCEQ = 'test/source_qualification_materializer/'
INTERVAL = 'test/interval_workers_qualification/'
MATERIALIZER = 'tools/actual_b_chain/materialize_source_qualification.py'
VERIFIER = 'tools/actual_b_chain/sourceq_full_scale_verifier.py'
RUNNER = 'tools/actual_b_chain/run_interval_workers_qualification.py'
STAGE_HELPER = 'tools/actual_b_chain/run_matched342_native.py'
OWNER = 'research/dbdt_completed_source/candidate_native_owner.py'
MEMORY = 'tools/run_canonical_harness.py'
WHOLE_SECONDS, STAGE_SECONDS, CLEANUP_SECONDS, POSTCHECK_SECONDS = 900, 120, 20, 5
RAW_LIMIT, META_LIMIT, INPUT_LIMIT, EXECUTABLE_LIMIT = 4 * 1024**2, 2 * 1024**2, 4 * 1024**2, 128 * 1024**2
FIXED = {
    MATERIALIZER: '2a7d4ae5e93b7b3436ae95dc9ae05e153268fd37cc7b4620193c769c4ef5dd6f',
    VERIFIER: '6859faa129f73759f3772d4f8a790641a831a0bc519596a97f303c9437034011',
    RUNNER: '9dc46a4a0df7f5a4ff13818be963c91db6d42b74fd10c436bbe3d984cca50e63',
    STAGE_HELPER: 'ea2f94ec709b5e58086b69fbd142f902499d9a2d2f5dabc8e0c871c452b4533b',
    OWNER: 'c3575532cb67b4bf52d2db8a20e01ca7e8d97cfb0cb1054ded66467823e40c00',
    'research/dbdt_completed_source/native_stage.py': '30c7fc96042c3d84814f0ea728c0415d657d90c02b5de4cf516da517c7f9a21c',
    MEMORY: 'defbb2a90690acd83ff0dc8eae05fd8622543871ecfa6c7a595b04069d44b331',
    SOURCEQ + 'test_materializer.py': '3865c24a223012783707c7e5595b73fa74ac48ded094a8c937fd2e31e926cfbc',
    INTERVAL + 'test_controls.py': 'b3277bc44a2379cd00521d09b4c89fdc01a6845750b51837cb3b04df166e6486',
    SOURCEQ + 'test_canonical_native_boundary.py': '7edfbb712ed7f65d2857cac0f33bfbc8db935cc163f80ed2b90fde41e466e533',
    SOURCEQ + 'test_copy_boundary.py': '9d8b5e952cdd90001d3cf70a3d38a3e42b99f680cf246c62745f622c7dad66b9',
    SOURCEQ + 'test_cli_publication.py': 'c6c80246685d61a84667efd019f68557100fc0da468c410fac41a20da2a1193e',
    SOURCEQ + 'test_evidence_gaps.py': '73d1e141f0e4260d0a6df3f0ba8a21e875e7ffe2f3ce381ac0bc924622807021',
    SOURCEQ + 'test_lifecycle_event_predicate.py': 'd3a48892a3503acf2e94fd3204918ea794a22b2267e99550e0d773a3ff4b5fc0',
    SOURCEQ + 'sourceq_importer_oracles_v1.json': '009fc49e5496ec840e5e014ebb60e8096e726921ae191caa18a936bf70c34743',
    SOURCEQ + 'canonical_native_boundary_oracles_v1.json': '3e1957e74d0c29a49d59e3e3f2f0b19583bbfd9a74cf532feb12b59f91f01114',
    SOURCEQ + 'copy_boundary_oracles_v1.json': '3d04346734aca03ba1a4b254da8f8aa955e36e3f0569fdcb61f3ba129c789eb9',
    SOURCEQ + 'cli_publication_oracles_v1.json': '8f82b0b80afa97c3683b265dcd20aab9de2d6cae7c5cc1cb6107d105b5ff5e42',
    SOURCEQ + 'evidence_gap_oracles_v1.json': '2ca2cb56576bb8aac805eb4acaac5d5c17686f546d961b7426b15cc55e939bf7',
    SOURCEQ + 'stage_elapsed_oracles_v1.json': '611d707f47368286338ed8fef9eb7e425239bc83bff60620ca8b9af0941f3d42',
    SOURCEQ + 'lifecycle_event_oracles_v1.json': '59b1af3cd6cc1c1feeeb51f35b28f5656c1befeffbc8d2cee2e1a0e033b835d0',
    SOURCEQ + 'fixtures/component136_names_v1.json': '5ee28ac856702e60d36dc2efb22248bca12726618133c90f57607f4a0935789e',
    SOURCEQ + 'fixtures/native69a8-source-proof.json': '1c32f26c06c06c7d9e17850cb4b56bb9f089eed0b4395961109c8e7153f04fa1',
    SOURCEQ + 'fixtures/native69a8-posix-macos.zip': 'cb9775aeebc2acb0b82a9e5edbb31628821e6c79546849e41073b0378697c520',
    SOURCEQ + 'fixtures/native69a8-anchor-macos.zip': '3781e526631985414946abfbd02f8e99cefa4c02d7854fd4e59130f08998e778',
    INTERVAL + 'workers17_oracles_v1.json': '94331d0a6ea299803414e3b9243cc1777468da2971612d9bf22d9839466f7058',
    'test/test_interval_mean_forecaster.jl': '72dd65b36d313b39703c93a60a4a92a91146c786aa4e4069fac4e4aa99d1fbcd',
    'test/interval_mean_forecaster/worker.jl': 'e3cd1ca92c85c544b812b8debf56920c47ba7fba49cd156dd9387e4b9df56842',
    'test/interval_mean_forecaster/bindings.jl': '5cbef72d434cae309d382c110f832566aefe068f1661b31c7fd9464f11c7c939',
    '.github/workflows/interval-workers-qualification.yml': '4d8c19c2bb5abc9b30b3240207609f7674e59080ffe33521b52df353f6e4fe31',
    'tools/actual_b_chain/BMatchedReference342.jl': 'f6f89e7c9fd81f02b591ad7d590466df3035e4c0171170116ea7f03d32874edb',
    'test/matched_initial342/test_native_owner_lifecycle.py': '14fcd2b2331e4e105507cbe1ccdb8bd9c8ce2334fa0c6bb032b4349cf4cb2f73',
}
CASES = (
    ('materializer24', SOURCEQ + 'test_materializer.py', 24, 'matched342_sourceq_independent_tests_v1',
     (SOURCEQ + 'test_materializer.py', SOURCEQ + 'sourceq_importer_oracles_v1.json', MATERIALIZER, VERIFIER,
      'tools/actual_b_chain/BMatchedReference342.jl', SOURCEQ + 'fixtures/component136_names_v1.json',
      SOURCEQ + 'fixtures/native69a8-source-proof.json', SOURCEQ + 'fixtures/native69a8-posix-macos.zip',
      SOURCEQ + 'fixtures/native69a8-anchor-macos.zip'),
     ('tests_issued_qualification_outputs', 'numerical_or_POSIX_fixture_launches_requested'),
     ('original_inert_archives_executed_or_extracted',)),
    ('workers10', INTERVAL + 'test_controls.py', 10, 'interval_workers_qualification_independent_controls_v1',
     (INTERVAL + 'test_controls.py', INTERVAL + 'workers17_oracles_v1.json', RUNNER,
      '.github/workflows/interval-workers-qualification.yml', 'test/test_interval_mean_forecaster.jl',
      'test/interval_mean_forecaster/worker.jl', 'test/interval_mean_forecaster/bindings.jl'),
     ('numerical_or_native_launches_requested', 'qualification_outputs_requested'),
     ('metadata_fixtures_are_actual_worker_or_native_evidence', 'routing_sentinels_are_native_results')),
    ('canonical3', SOURCEQ + 'test_canonical_native_boundary.py', 3,
     'sourceq_canonical_native_boundary_independent_tests_v1',
     (SOURCEQ + 'test_canonical_native_boundary.py', SOURCEQ + 'canonical_native_boundary_oracles_v1.json', MATERIALIZER, VERIFIER),
     ('actual_numerical_or_owner_launches_requested',), ('metadata_fixtures_are_native_evidence',)),
    ('copy2', SOURCEQ + 'test_copy_boundary.py', 2, 'sourceq_copy_boundary_independent_tests_v1',
     (SOURCEQ + 'test_copy_boundary.py', SOURCEQ + 'copy_boundary_oracles_v1.json', MATERIALIZER, VERIFIER),
     ('actual_numerical_or_owner_launches_requested',), ('verification_or_authority_mocked', 'full_qualification_fixture_issued')),
    ('publication5', SOURCEQ + 'test_cli_publication.py', 5,
     'sourceq_and_interval_cli_publication_independent_tests_v1',
     (SOURCEQ + 'test_cli_publication.py', SOURCEQ + 'cli_publication_oracles_v1.json', MATERIALIZER, RUNNER),
     ('numerical_or_owner_launches_requested',), ('full_qualification_or_actual_owner_fixture_supplied', 'authority_or_evidence_verifier_mocked')),
    ('evidence5', SOURCEQ + 'test_evidence_gaps.py', 5, 'sourceq_finite_evidence_gaps_independent_tests_v1',
     (SOURCEQ + 'test_evidence_gaps.py', SOURCEQ + 'evidence_gap_oracles_v1.json', SOURCEQ + 'stage_elapsed_oracles_v1.json',
      MATERIALIZER, VERIFIER, SOURCEQ + 'fixtures/native69a8-source-proof.json',
      SOURCEQ + 'fixtures/native69a8-posix-macos.zip', SOURCEQ + 'fixtures/native69a8-anchor-macos.zip'),
     ('actual_numerical_or_owner_launches_requested',),
     ('metadata_native_and_context_fixtures_are_actual_collector_evidence', 'full_qualification_fixture_or_bypass')),
    ('lifecycle2', SOURCEQ + 'test_lifecycle_event_predicate.py', 2, 'sourceq_lifecycle_event_independent_tests_v1',
     (SOURCEQ + 'test_lifecycle_event_predicate.py', SOURCEQ + 'lifecycle_event_oracles_v1.json', MATERIALIZER, VERIFIER,
      'test/matched_initial342/test_native_owner_lifecycle.py'),
     ('actual_OS_observations_or_owner_launches', 'Windows_native_cases', 'POSIX_native_cases'), ('qualification_issued',)),
)


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def clock(deadline):
    need(time.monotonic_ns() < deadline, 'original900s controller deadline exhausted')


def ordinary(path, directory=False):
    path = Path(path)
    need(path.is_absolute() and path == path.resolve(strict=True), 'canonical ordinary path required')
    for member in (path, *path.parents):
        info = member.lstat()
        need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
             'symlink/reparse input rejected')
    need((path.is_dir() if directory else path.is_file()), 'ordinary file/directory required')
    return path


def read_bytes(path, limit):
    path = ordinary(path)
    before = path.stat()
    stamp = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    need(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= limit, 'bounded ordinary file required')
    with path.open('rb') as stream:
        need(stamp(os.fstat(stream.fileno())) == stamp(before), 'file changed before read')
        body = stream.read(limit + 1)
        need(stamp(os.fstat(stream.fileno())) == stamp(before), 'file changed during read')
    need(len(body) == before.st_size and len(body) <= limit and stamp(path.stat()) == stamp(before), 'file changed after read')
    return body


def digest(path, limit=INPUT_LIMIT):
    return hashlib.sha256(read_bytes(path, limit)).hexdigest()


def read_json(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            need(key not in value, 'duplicate result key')
            value[key] = item
        return value
    return json.loads(read_bytes(path, META_LIMIT), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write_new(path, value):
    body = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf8')
    need(len(body) <= META_LIMIT, 'metadata exceeds2MiB')
    with path.open('xb') as stream:
        need(stream.write(body) == len(body), 'short metadata write')
        stream.flush()
        os.fsync(stream.fileno())


def load_pinned(member, name):
    path = ROOT / member
    need(digest(path) == FIXED[member], 'pinned module changed before import')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    need(digest(path) == FIXED[member], 'pinned module changed during import')
    return module


def protected_sources(python):
    values = {}
    for member, expected in FIXED.items():
        values[str(ROOT / member)] = digest(ROOT / member)
        need(values[str(ROOT / member)] == expected, 'fixed control input changed: ' + member)
    values[str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
    values[str(python)] = digest(python, EXECUTABLE_LIMIT)
    return values


def native_positive(native, launched):
    common = {'pid', 'native_exit', 'timed_out', 'wall_seconds', 'ownership', 'descendant_cleanup',
              'leader_reaped', 'ownership_boundary_closed', 'all_descendants_os_reaped', 'descendant_evidence'}
    variant = 'wrapper_observed_wall_seconds' if os.name == 'nt' else 'observation_method'
    need(type(native) is dict and set(native) == common | {variant}, 'current closed native receipt required')
    need(type(native['pid']) is int and len(launched) == 1 and native['pid'] == launched[0]['pid'], 'actual native callback PID mismatch')
    need(type(native['native_exit']) is int and native['native_exit'] == 0 and native['timed_out'] is False and
         native['leader_reaped'] is True and native['ownership_boundary_closed'] is True and
         native['all_descendants_os_reaped'] is False, 'untimed native0/reaped/closed boundary required')
    pair = (('CREATE_SUSPENDED -> AssignProcessToJobObject -> ResumeThread', 'private Windows Job Object closed with KILL_ON_JOB_CLOSE')
            if os.name == 'nt' else ('start_new_session=True', 'owned POSIX session has no live members after stage'))
    need((native['ownership'], native['descendant_cleanup']) == pair and type(native['descendant_evidence']) is str and
         bool(native['descendant_evidence']), 'current platform ownership pair differs')
    for wall in ([native['wall_seconds'], native[variant]] if os.name == 'nt' else [native['wall_seconds']]):
        need(type(wall) in (int, float) and math.isfinite(wall) and 0 <= wall < STAGE_SECONDS, 'native stage wall exceeds120s')
    if os.name != 'nt':
        need(native[variant] == 'terminal-code waitid with WNOWAIT; Darwin stop-only consumption; original-group signal; zero-live-SID census; then terminal wait',
             'current POSIX observation method differs')


def verify_result(case, result, transcript, protected, launch, returned_ns):
    name, script, methods, schema, members, zero_fields, false_fields = case
    need(type(result) is dict and result.get('schema') == schema and result.get('complete') is True and
         type(result.get('unittest_methods')) is int and type(result.get('expected_methods')) is int and
         result['unittest_methods'] == result['expected_methods'] == methods and
         all(type(result.get(key)) is int and result[key] == 0 for key in ('failures', 'errors', 'skipped')),
         'exact complete test schema/counts required: ' + name)
    expected = {str(ROOT / member): protected[str(ROOT / member)] for member in members}
    need(result.get('source_before') == expected and result.get('source_after') == expected and
         result.get('source_unchanged') is True, 'case physical/source-report join differs: ' + name)
    need(all(result.get(key) is False for key in ('source_qualified', 'software_qualified', 'scientific_qualified', *false_fields)) and
         all(type(result.get(key)) is int and result[key] == 0 for key in zero_fields), 'metadata case claimed execution/authority: ' + name)
    need(all(type(result.get(key)) is int for key in ('started_ns', 'finished_ns')) and
         launch['monotonic_ns'] <= result['started_ns'] <= result['finished_ns'] <= returned_ns,
         'actual launch/test-result clock association differs: ' + name)
    need(re.findall(rb'^Ran (\d+) tests in [^\r\n]+\r?$', transcript, re.M) == [str(methods).encode()] and
         re.findall(rb'^(OK)\r?$', transcript, re.M) == [b'OK'] and
         re.search(rb'^(?:FAILED\b|FAIL:|ERROR:|SKIPPED\b)', transcript, re.M) is None,
         'exact raw unittest count/OK required: ' + name)
    names = re.findall(rb'^(test_[A-Za-z0-9_]+)\s+\(', transcript, re.M)
    need(len(names) == len(set(names)) == methods, 'unique verbose method names differ: ' + name)
    if name == 'materializer24':
        need((result.get('frozen_negative_rows'), result.get('frozen_metadata_positive_rows'), result.get('frozen_actual_unready_rows')) == (20, 3, 1),
             'original24 frozen rows differ')
    if name == 'workers10':
        need(type(result.get('frozen_negative_rows')) is int and result['frozen_negative_rows'] == 10, 'W01-W10 inventory differs')
    if name == 'lifecycle2':
        need(type(result.get('literal_rows')) is int and type(result.get('expected_rows')) is int and
             result['literal_rows'] == result['expected_rows'] == 15 and type(result.get('rows')) is list and
             [row.get('id') for row in result['rows']] == ['P01', 'P02', 'P03'] + ['N%02d' % i for i in range(1, 13)] and
             all(row.get('accepted') is True for row in result['rows'][:3]) and
             all(row.get('accepted') is False and row.get('code') == 'coherent_header_raw_counterfeit' for row in result['rows'][3:]) and
             result.get('fixtures_are_metadata_not_native_observations') is True and
             type(result.get('deadline_ns')) is int and result['deadline_ns'] == result['started_ns'] + 90 * 10**9 and
             result['finished_ns'] < result['deadline_ns'], 'frozen15 event-metadata predicates differ')


def main():
    started = time.monotonic_ns()
    deadline = started + WHOLE_SECONDS * 10**9
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    need(not sys.flags.optimize and sys.version_info >= (3, 11) and
         (os.name == 'nt' or sys.platform == 'darwin' or sys.platform.startswith('linux')), 'supported unoptimized Python>=3.11 required')
    output = args.output_directory
    need(output.is_absolute() and not os.path.lexists(output) and output.name not in ('', '.', '..'), 'fresh absolute output required')
    ordinary(output.parent, True)
    need(output.parent == output.parent.resolve(strict=True) and not output.is_relative_to(ROOT) and not ROOT.is_relative_to(output),
         'external disjoint nonalias output required')
    python = ordinary(Path(sys.executable).resolve(strict=True))
    memory = load_pinned(MEMORY, 'sourceq_interval_resources')

    def admit():
        clock(deadline)
        available, method = memory.available_memory()
        free = shutil.disk_usage(output.parent).free
        need(type(available) is int and available >= 536870912 and type(free) is int and free >= 8589934592 and
             type(method) is str and bool(method), 'actual engineering512MiB/8GiB admission failed')
        clock(deadline)
        return dict(available_physical_bytes=available, memory_method=method, free_disk_bytes=free, observed_ns=time.monotonic_ns(),
                    resource_probe_subprocesses=1 if sys.platform == 'darwin' else 0,
                    resource_probe_PID_observed=False)

    admission = admit()
    before = protected_sources(python)
    owner = load_pinned(OWNER, 'sourceq_interval_current_owner')
    helper = load_pinned(STAGE_HELPER, 'sourceq_interval_stage_helper')
    finalizer = load_pinned(RUNNER, 'sourceq_interval_terminal_protocol')
    need(before == protected_sources(python), 'source changed during controller imports')
    clock(deadline)
    output.mkdir()
    record = dict(schema='sourceq_interval_registered_controls_v1', complete=False,
                  started_ns=started, deadline_ns=deadline, admission=admission,
                  source_before=before, fixed_input_sha256=FIXED, case_receipts=[],
                  cases=0, expected_cases=7, unittest_methods=0, expected_methods=51,
                  stage_seconds=STAGE_SECONDS, owner_cleanup_seconds=CLEANUP_SECONDS,
                  postcheck_reserve_seconds=POSTCHECK_SECONDS,
                  runtime=dict(python_version=sys.version, python_executable=str(python), python_sha256=before[str(python)],
                               platform=sys.platform, os_name=os.name, uname=list(os.uname()) if os.name == 'posix' else None),
                  actual_Python_stage_launches=0, Julia_launched=False, actual_data_exposed=False,
                  native_Darwin_qualified=False, qualification_issued=False,
                  source_qualified=False, software_qualified=False, scientific_qualified=False)
    write_new(output / 'PREEXECUTION.json', record)
    try:
        for case in CASES:
            name, script, methods, schema, members, _, _ = case
            need(before == protected_sources(python), 'source changed before case')
            resource = admit()
            need(time.monotonic_ns() + (STAGE_SECONDS + CLEANUP_SECONDS + POSTCHECK_SECONDS) * 10**9 < deadline,
                 'full120s stage plus unchanged cleanup/postcheck no longer fits original900s clock')
            target, rawpath = output / name, output / (name + '.log')
            launched = []
            stage_started = time.monotonic_ns()
            def on_launch(pid):
                need(type(pid) is int and pid > 0 and not launched, 'one actual stage callback required')
                launched.append(dict(pid=pid, monotonic_ns=time.monotonic_ns()))
                record['actual_Python_stage_launches'] += 1
                write_new(output / (name + '-launch.json'), launched[0])
            try:
                native = helper._run_owned_stage(owner,
                    [str(python), '-I', '-B', '-X', 'utf8', str(ROOT / script), '--output-directory', str(target)],
                    cwd=str(ROOT), environment=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
                                                   TMP=str(output), TEMP=str(output), TMPDIR=str(output)),
                    logpath=rawpath, timeout=STAGE_SECONDS, on_launch=on_launch)
            except BaseException as error:
                failure = dict(native_result_returned=False,
                    observed_launches=launched, admission=resource, stage_started_ns=stage_started,
                    error=dict(type=type(error).__name__, message=str(error)), traceback=traceback.format_exc())
                try:
                    if rawpath.is_file():
                        body = read_bytes(rawpath, RAW_LIMIT)
                        failure['raw'] = dict(path=str(rawpath), bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
                except BaseException as raw_error:
                    failure['raw_error'] = repr(raw_error)
                write_new(output / (name + '-owner-exception.json'), failure)
                raise
            returned = time.monotonic_ns()
            receipt = dict(name=name, script=script, expected_methods=methods, expected_schema=schema,
                           admission=resource, launch=launched, native=native, stage_started_ns=stage_started,
                           returned_ns=returned, timeout_seconds=STAGE_SECONDS,
                           observed_stage_wall_seconds=(returned - stage_started) / 10**9)
            record['case_receipts'].append(receipt)
            try:
                transcript = read_bytes(rawpath, RAW_LIMIT)
                receipt['raw'] = dict(path=str(rawpath), bytes=len(transcript), sha256=hashlib.sha256(transcript).hexdigest())
            except BaseException as raw_error:
                receipt['raw_error'] = repr(raw_error)
                write_new(output / (name + '-owner.json'), receipt)
                raise
            write_new(output / (name + '-owner.json'), receipt)
            native_positive(native, launched)
            need(stage_started <= launched[0]['monotonic_ns'] <= returned and
                 0 <= returned - stage_started < STAGE_SECONDS * 10**9,
                 'observed native stage/callback exceeded original120s allowance')
            clock(deadline)
            resultpath = target / 'RESULT.json'
            result = read_json(resultpath)
            receipt['result'] = dict(path=str(resultpath), bytes=resultpath.stat().st_size, sha256=digest(resultpath, META_LIMIT))
            verify_result(case, result, transcript, before, launched[0], returned)
            need(before == protected_sources(python), 'source changed across case')
            clock(deadline)
            receipt['verified'] = True
            write_new(output / (name + '-verified.json'), receipt)
            record['cases'] += 1
            record['unittest_methods'] += methods
        need(record['cases'] == 7 and record['unittest_methods'] == 51 and record['actual_Python_stage_launches'] == 7,
             'all seven fixed CLIs/51 methods/actual launches required')
        record['complete'] = True
    except BaseException as error:
        record.update(complete=False, failure=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
    try:
        after = protected_sources(python)
        record.update(source_after=after, source_unchanged=before == after)
        record['complete'] = record['complete'] and record['source_unchanged'] and time.monotonic_ns() < deadline
    except BaseException as error:
        record.update(complete=False, source_unchanged=False, source_guard_error=repr(error))
    record['finished_ns'] = time.monotonic_ns()
    need(len(json.dumps(record, allow_nan=False).encode('utf8')) <= META_LIMIT, 'aggregate metadata exceeds2MiB')
    if not record['complete']:
        write_new(output / 'FAILURE.json', record)

    def emit_unaccepted():
        print(json.dumps(dict(accepted=False, case_verification_passed=record['complete'], cases=record['cases'],
                             unittest_methods=record['unittest_methods'], evidence=str(output)), sort_keys=True), flush=True)
        need(before == protected_sources(python), 'source changed during final output')
        clock(deadline)

    accepted = finalizer.finalize_interval_terminal(output / 'TERMINAL.json', record, deadline, emit=emit_unaccepted)
    return 0 if accepted else 1


if __name__ == '__main__':
    raise SystemExit(main())
