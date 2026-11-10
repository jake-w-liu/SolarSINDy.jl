"""Fixed, finite observation launcher; never issues an ownership qualification.

The pinned production candidate owner is exercised unchanged. Its exception,
including Darwin EPERM, remains a failure even when the diagnostic child exited
normally. The native direct-child wait is recorded separately from group/session
closure, which this launcher does not establish.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

OWNER_SHA = 'ccbf1cbd0709e50d0b5080bdbe4977ae80ebd1649237e016588d24800ddf2e21'
LEGACY_SHA = '15d2644ae5eb2e87ce31c9913f103f9fa7e940b67403b94f29ef2ed5a4d175b8'
DIAGNOSTIC_SHA = '3395df92110f61ac1deaae85f1fcfe93125865575e29619623a83737e95a2ee2'
OUTER_SECONDS = 180.
CHILD_SECONDS = 150.
RAW_CAP = 4 * 1024**2
MIN_RAM = 512 * 1024**2
MIN_DISK = 8 * 1024**3


def pin(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False) + '\n').encode('utf-8')
    if len(raw) > 2 * 1024**2:
        raise RuntimeError('launcher metadata byte cap')
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def physical_available():
    # Same conservative Darwin page domain as tools/run_canonical_harness.py:
    # free + speculative pages, excluding inactive/reclaimable estimates.
    result = subprocess.run(['/usr/bin/vm_stat'], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=5, check=True)
    if len(result.stdout) > 65536 or len(result.stderr) > 65536:
        raise RuntimeError('vm_stat output cap')
    text = result.stdout.decode('ascii', errors='strict')
    size = re.search(r'page size of (\d+) bytes', text)
    rows = dict(re.findall(r'^(Pages free|Pages speculative):\s+(\d+)\.$', text, re.M))
    if size is None or len(rows) != 2:
        raise RuntimeError('Darwin physical-page evidence unavailable')
    return sum(map(int, rows.values())) * int(size[1])


def failure(error):
    return dict(type=type(error).__name__, errno=getattr(error, 'errno', None),
                message=str(error)[:4096])


def main():
    started = time.monotonic()
    if len(sys.argv) != 1 or not __debug__:
        raise RuntimeError('fixed unoptimized file-based launcher accepts no arguments')
    checkout = Path(__file__).resolve(strict=True).parents[2]
    parent = Path(os.environ['RUNNER_TEMP']).resolve(strict=True)
    if parent == checkout or parent in checkout.parents or checkout in parent.parents:
        raise RuntimeError('fixed diagnostic evidence must be outside the checkout')
    output = parent / 'solarsindy-darwin-owner-diagnostic-v2'
    output.mkdir()  # A prior attempt is preserved; no overwrite or retry.
    paths = {
        'candidate_owner': checkout/'research/dbdt_completed_source/candidate_native_owner.py',
        'legacy_owner': checkout/'research/dbdt_completed_source/native_stage.py',
        'audit_owner': checkout/'tools/initial_reference_audit/owned_process.py',
        'diagnostic': Path(__file__).with_name('diagnose_darwin_groups.py'),
    }
    expected = {'candidate_owner': OWNER_SHA, 'legacy_owner': LEGACY_SHA,
                'audit_owner': LEGACY_SHA, 'diagnostic': DIAGNOSTIC_SHA}
    before = {}
    outcome = owner_error = launch_error = post_error = None
    launched = False
    native = {'started': False, 'pid': None, 'wait_returned': False,
              'native_exit': None, 'wait_seconds': None, 'reaped_observed': False}
    capture = {'bytes_seen': 0, 'bytes_retained': 0, 'cap_exceeded': False,
               'eof_observed': False, 'error': None, 'close_error': None,
               'thread_joined': False}
    group = {'attempted': False, 'signal_returned': False, 'error': None}
    process_holder = []
    reader_holder = []
    admission = None
    try:
        if platform.system() != 'Darwin' or os.name != 'posix':
            raise RuntimeError('Darwin diagnostic requires its native macOS runner')
        if sys.version_info[:3] != (3, 14, 10):
            raise RuntimeError('diagnostic requires exact Python 3.14.10')
        before = {key: pin(path) for key, path in paths.items()}
        if before != expected:
            raise RuntimeError('fixed diagnostic/production owner pins differ')
        memory = physical_available()
        disk = shutil.disk_usage(output).free
        admission = dict(available_physical_bytes=memory, free_disk_bytes=disk,
                         minimum_physical_bytes=MIN_RAM, minimum_disk_bytes=MIN_DISK,
                         method='Darwin vm_stat free plus speculative physical pages')
        write_new(output/'ADMISSION.json', admission)
        if memory < MIN_RAM or disk < MIN_DISK:
            raise RuntimeError('physical RAM/disk admission failed before Python child')
        if time.monotonic() - started >= 10:
            raise RuntimeError('prelaunch reserve exhausted')
        spec = importlib.util.spec_from_file_location('observed_fixed_candidate_owner', paths['candidate_owner'])
        owner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(owner)

        class ObservedProcess:
            def __init__(self, actual, reader):
                self.actual, self.reader = actual, reader

            def __getattr__(self, key):
                return getattr(self.actual, key)

            def kill(self):
                # Delegate unchanged. Popen may itself observe/reap an already
                # terminal leader here; the subsequent real wait is retained.
                return self.actual.kill()

            def wait(self, timeout=None):
                before_wait = time.monotonic()
                result = self.actual.wait(timeout=timeout)
                native.update(wait_returned=True, native_exit=result,
                              wait_seconds=time.monotonic()-before_wait,
                              reaped_observed=True)
                self.reader.join(timeout=min(2., max(0., OUTER_SECONDS-(time.monotonic()-started))))
                capture['thread_joined'] = not self.reader.is_alive()
                return result

        class ObservedSubprocess:
            STDOUT = subprocess.STDOUT

            @staticmethod
            def Popen(argv, **options):
                if process_holder or options.get('start_new_session') is not True:
                    raise RuntimeError('one fresh diagnostic child session required')
                sink = options['stdout']
                options['stdout'] = subprocess.PIPE
                actual = subprocess.Popen(argv, **options)
                native.update(started=True, pid=actual.pid)
                process_holder.append(actual)

                def drain():
                    try:
                        while True:
                            chunk = actual.stdout.read(65536)
                            if not chunk:
                                capture['eof_observed'] = True
                                break
                            capture['bytes_seen'] += len(chunk)
                            keep = chunk[:max(0, RAW_CAP-capture['bytes_retained'])]
                            if keep:
                                written = sink.write(keep)
                                if type(written) is not int or not 0 < written <= len(keep):
                                    raise RuntimeError('raw sink write made invalid/no progress')
                                capture['bytes_retained'] += written
                                if written != len(keep):
                                    raise RuntimeError('raw sink short write')
                            if capture['bytes_seen'] > RAW_CAP:
                                capture['cap_exceeded'] = True
                    except BaseException as error:
                        capture['error'] = failure(error)
                    finally:
                        try:
                            actual.stdout.close()
                        except BaseException as error:
                            capture['close_error'] = failure(error)
                            if capture['error'] is None:
                                capture['error'] = capture['close_error']

                reader = threading.Thread(target=drain, name='bounded-diagnostic-raw', daemon=True)
                reader_holder.append(reader)
                try:
                    reader.start()
                except BaseException:
                    # Popen already succeeded, but run_owned has not received
                    # the process object. Retain an actual native wait here.
                    group['attempted'] = True
                    try:
                        os.killpg(actual.pid, signal.SIGKILL)
                        group['signal_returned'] = True
                    except BaseException as error:
                        group['error'] = failure(error)
                        actual.kill()
                    finally:
                        before_wait = time.monotonic()
                        code = actual.wait(timeout=20)
                        native.update(wait_returned=True, native_exit=code,
                                      wait_seconds=time.monotonic()-before_wait,
                                      reaped_observed=True)
                        actual.stdout.close()
                    raise
                return ObservedProcess(actual, reader)

        class ObservedOS:
            def __getattr__(self, key):
                return getattr(os, key)

            def killpg(self, pid, number):
                if (len(process_holder) != 1 or pid != process_holder[0].pid or
                        number != signal.SIGKILL or native['reaped_observed']):
                    raise RuntimeError('unowned or post-wait group signal refused')
                group['attempted'] = True
                try:
                    os.killpg(pid, number)
                except BaseException as error:
                    group['error'] = failure(error)
                    raise  # EPERM remains the pinned owner's failure.
                group['signal_returned'] = True

        owner.subprocess, owner.os = ObservedSubprocess, ObservedOS()
        environment = dict(os.environ, RUNNER_TEMP=str(output),
                           PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')

        def on_launch(pid):
            nonlocal launched
            launched = True
            write_new(output/'LAUNCHED.json', dict(pid=pid, actual_native_start=True,
                      elapsed_seconds=time.monotonic()-started))

        try:
            outcome = owner.run_owned(
                [sys.executable, '-B', '-X', 'utf8', str(paths['diagnostic']), '--checkout', str(checkout)],
                cwd=str(output), environment=environment, logpath=output/'native.log',
                timeout=CHILD_SECONDS, on_launch=on_launch)
        except BaseException as error:
            owner_error = failure(error)
    except BaseException as error:
        launch_error = failure(error)
    after = {}
    try:
        after = {key: pin(path) for key, path in paths.items()}
        if before != expected or after != expected:
            raise RuntimeError('source pre/post pins not identical to fixed expected values')
        if any(reader.is_alive() for reader in reader_holder):
            raise RuntimeError('raw capture thread unresolved; capture cannot be accepted')
    except BaseException as error:
        post_error = failure(error)
    raw = output/'native.log'
    raw_ref = None if not raw.is_file() or any(reader.is_alive() for reader in reader_holder) else dict(
        path=str(raw), bytes=raw.stat().st_size, sha256=pin(raw))
    capture_snapshot = (dict(capture) if not any(reader.is_alive() for reader in reader_holder)
                        else dict(thread_joined=False, mutable_capture_state_accepted=False))
    observations = list(output.glob('solarsindy-darwin-owner-diag-*/OBSERVATIONS.json'))
    observation_refs = [dict(path=str(path), bytes=path.stat().st_size, sha256=pin(path)) for path in observations]
    python_sha = pin(sys.executable)
    actor_sha = pin(__file__)
    # Native kernel fields only; optional processor remains unqueried/unknown.
    system = dict(zip(('system','node','release','version','machine'), os.uname()))
    system['processor'] = ''
    raw_size_matches = (raw_ref is not None and capture_snapshot.get('thread_joined') is True and
                        raw_ref['bytes'] == capture_snapshot['bytes_retained'] <= RAW_CAP)
    # This sample follows every checked source/raw/observation/runtime hash.
    elapsed = time.monotonic()-started
    passed = (launch_error is None and owner_error is None and post_error is None and
              launched and native['wait_returned'] and native['native_exit'] == 0 and
              capture['thread_joined'] and capture['eof_observed'] and capture['error'] is None and
              not capture['cap_exceeded'] and raw_size_matches and
              len(observations) == 1 and elapsed < OUTER_SECONDS and
              outcome is not None and outcome.get('timed_out') is False)
    terminal = dict(schema='darwin_owner_diagnostic_native_observer_v2',
                    observation_launcher_passed=passed, native=native, capture=capture_snapshot,
                    owner_returned_claims_unqualified=outcome, owner_exception=owner_error,
                    launcher_exception=launch_error, postcheck_exception=post_error,
                    first_group_signal=group, admission=admission,
                    actual_wall_seconds=elapsed, fixed_outer_seconds=OUTER_SECONDS,
                    wall_scope='through all source/raw/observation/runtime hashes and system metadata; before terminal fsync',
                    child_owner_timeout_seconds=CHILD_SECONDS, deadline_satisfied=elapsed < OUTER_SECONDS,
                    source_before=before, source_after=after, raw=raw_ref,
                    raw_size_matches_retained_cap=raw_size_matches, observations=observation_refs,
                    python_version=sys.version, python_binary_sha256=python_sha,
                    actor_source_sha256=actor_sha, github_sha=os.environ.get('GITHUB_SHA'),
                    system=system,
                    ownership_boundary_closed=False, ownership_boundary_closure_observed=False,
                    all_descendants_os_reaped=False, production_owner_qualified=False,
                    original_CI_root_cause_proven=False, scientific_qualified=False)
    write_new(output/'TERMINAL.json', terminal)
    print(json.dumps(dict(output=str(output), terminal_sha256=pin(output/'TERMINAL.json'),
                          observation_launcher_passed=passed, production_owner_qualified=False)), flush=True)
    # Terminal fsync, hash, and console output are inside the acceptance clock.
    # A late terminal can never turn into a successful native actor exit.
    return 0 if passed and time.monotonic()-started < OUTER_SECONDS else 1


if __name__ == '__main__':
    raise SystemExit(main())
