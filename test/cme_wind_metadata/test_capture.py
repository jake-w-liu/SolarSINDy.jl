"""Independent transport fixtures; no real NASA responses or authority grants."""
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('wind_metadata_capture_under_test', ROOT/'tools/capture_wind_metadata.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class Response(io.BytesIO):
    def __init__(self, body=b'opaque metadata fixture\n', code=200, headers=None):
        super().__init__(body)
        self.headers = headers or {}
        self.code = code

    def getcode(self):
        return self.code


class WindMetadataCaptureTests(unittest.TestCase):
    def setUp(self):
        # Existing transport oracles exercise the network kernel in process;
        # mocks must never escape into a real request in a spawned worker.
        owner = patch.object(M, '_run_resource_owned',
            side_effect=lambda name, url, directory, deadline: M.fetch_one(name, url, directory))
        owner.start()
        self.addCleanup(owner.stop)

    def test_four_literal_resources_only_and_transport_is_not_authority(self):
        opener = Mock()
        opener.open.side_effect = lambda *args, **kwargs: Response()
        with tempfile.TemporaryDirectory() as raw, patch.object(M.urllib.request, 'build_opener', return_value=opener):
            root = Path(raw).resolve(strict=True)
            out = root/'capture'
            result = M.capture(out, 'a'*40)
            self.assertTrue(result['capture_complete'])
            self.assertTrue(result['master_and_at_least_one_info_HTTP200'])
            self.assertEqual(len(result['resources']), 4)
            self.assertEqual({call.args[0].full_url for call in opener.open.call_args_list}, {url for _, url in M.RESOURCES})
            for item in result['resources']:
                self.assertEqual(Path(item['body']['path']).read_bytes(), b'opaque metadata fixture\n')
                self.assertEqual(item['body']['sha256'], hashlib.sha256(b'opaque metadata fixture\n').hexdigest())
                self.assertFalse(item['actual_GSM_vectors_requested'])
                self.assertFalse(item['source_authority_granted'])
            self.assertFalse(result['source_authority_granted'])
            self.assertFalse(result['scientific_qualification_granted'])
            self.assertFalse(result['metadata_semantics_independently_admitted'])
            self.assertFalse(result['model_execution_performed'])
            for receipt in result['resource_receipts']:
                self.assertEqual(hashlib.sha256(Path(receipt['path']).read_bytes()).hexdigest(), receipt['sha256'])
            self.assertEqual(json.loads((out/'CAPTURE.json').read_bytes()), result)
            self.assertNotIn(b'\r', (out/'CAPTURE.json').read_bytes())
            with self.assertRaises(FileExistsError):
                M.capture(out)

    def test_non200_error_body_is_retained_without_transport_success(self):
        opener = Mock()
        def unavailable(request, **kwargs):
            raise urllib.error.HTTPError(request.full_url, 503, 'fixture unavailable', {}, io.BytesIO(b'fixture outage'))
        opener.open.side_effect = unavailable
        with tempfile.TemporaryDirectory() as raw, patch.object(M.urllib.request, 'build_opener', return_value=opener):
            result = M.capture(Path(raw).resolve()/'capture')
            self.assertFalse(result['master_and_at_least_one_info_HTTP200'])
            self.assertTrue(all(item['http_status']==503 for item in result['resources']))
            self.assertTrue(all(Path(item['body']['path']).read_bytes()==b'fixture outage' for item in result['resources']))

    def test_timeout_has_terminal_receipt_without_invented_body(self):
        opener = Mock()
        opener.open.side_effect = TimeoutError('fixture timeout')
        with tempfile.TemporaryDirectory() as raw, patch.object(M.urllib.request, 'build_opener', return_value=opener):
            result = M.capture(Path(raw).resolve()/'capture')
            self.assertFalse(result['master_and_at_least_one_info_HTTP200'])
            for item in result['resources']:
                self.assertFalse(item['response_complete'])
                self.assertNotIn('body', item)
                self.assertEqual(item['failure']['type'], 'TimeoutError')

    def test_oversized_stream_retains_only_bound_and_is_incomplete(self):
        opener = Mock()
        opener.open.return_value = Response(b'x'*33)
        with tempfile.TemporaryDirectory() as raw, patch.object(M.urllib.request, 'build_opener', return_value=opener), patch.object(M, 'MAX_BYTES', 32):
            directory = Path(raw).resolve()
            item, _ = M.fetch_one(*M.RESOURCES[0], directory)
            self.assertFalse(item['response_complete'])
            self.assertEqual(item['failure']['type'], 'ValueError')
            self.assertEqual(item['body']['bytes'], 32)
            self.assertEqual(Path(item['body']['path']).read_bytes(), b'x'*32)

    def test_declared_length_mismatch_preserves_incomplete_bytes(self):
        opener = Mock()
        opener.open.return_value = Response(b'abc', headers={'Content-Length':'4'})
        with tempfile.TemporaryDirectory() as raw, patch.object(M.urllib.request, 'build_opener', return_value=opener):
            item, _ = M.fetch_one(*M.RESOURCES[0], Path(raw).resolve())
            self.assertFalse(item['response_complete'])
            self.assertEqual(item['body']['bytes'], 3)

    def test_redirect_resource_and_credentials_are_strict(self):
        original = M.RESOURCES[1][1]
        M.validate_url(original.replace('cdaweb.', 'spdf.'), original)
        for changed in ('http://cdaweb.gsfc.nasa.gov/hapi/info?id=WI_H0_MFI%400',
                'https://example.org/hapi/info?id=WI_H0_MFI%400',
                'https://cdaweb.gsfc.nasa.gov/hapi/data?id=WI_H0_MFI%400',
                'https://cdaweb.gsfc.nasa.gov/hapi/info?id=WI_H0_MFI%401',
                'https://user:secret@cdaweb.gsfc.nasa.gov/hapi/info?id=WI_H0_MFI%400',
                original+'&id=WI_H0_MFI%400', original+'&id=', original+'&extra=', original+'#fragment'):
            with self.subTest(url=changed), self.assertRaises(ValueError):
                M.validate_url(changed, original)
        opener = Mock()
        redirect = urllib.error.HTTPError(original, 302, 'fixture redirect',
            {'Location':'https://cdaweb.gsfc.nasa.gov/hapi/data?id=WI_H0_MFI%400'}, io.BytesIO(b''))
        opener.open.side_effect = redirect
        with redirect, tempfile.TemporaryDirectory() as raw, patch.object(M.urllib.request, 'build_opener', return_value=opener):
            item, _ = M.fetch_one(*M.RESOURCES[1], Path(raw).resolve())
            self.assertFalse(item['response_complete'])
            self.assertNotIn('body', item)
            self.assertEqual(opener.open.call_count, 1)

    def test_output_and_claimed_head_rejection_precedes_requests(self):
        with tempfile.TemporaryDirectory() as raw, patch.object(M.urllib.request, 'build_opener') as opener:
            target = Path(raw).resolve()/'capture'
            for head in ('MAIN', 'a'*39, 'a'*41, 'A'*40, True, 7):
                with self.subTest(head=head), self.assertRaises(ValueError):
                    M.capture(target, head)
            with self.assertRaises(ValueError):
                M.capture(Path('relative'))
            self.assertFalse(target.exists())
            opener.assert_not_called()

    def test_empty_200_response_is_not_usable_metadata(self):
        opener = Mock()
        opener.open.side_effect = lambda *args, **kwargs: Response(b'')
        with tempfile.TemporaryDirectory() as raw, patch.object(M.urllib.request, 'build_opener', return_value=opener):
            result = M.capture(Path(raw).resolve()/'capture')
            self.assertFalse(result['master_and_at_least_one_info_HTTP200'])
            self.assertTrue(all(item['body']['bytes']==0 for item in result['resources']))

    def test_allowed_redirect_retains_same_group_without_credentials(self):
        original = M.RESOURCES[1][1]
        location = original.replace('cdaweb.', 'spdf.')
        redirect = urllib.error.HTTPError(original, 302, 'fixture redirect',
            {'Location':location}, io.BytesIO(b''))
        opener = Mock()
        opener.open.side_effect = [redirect, Response(b'fixed info fixture')]
        with redirect, tempfile.TemporaryDirectory() as raw, patch.object(M.urllib.request, 'build_opener', return_value=opener):
            item, _ = M.fetch_one(*M.RESOURCES[1], Path(raw).resolve())
            self.assertTrue(item['response_complete'])
            self.assertEqual(len(item['redirects']), 1)
            request = opener.open.call_args_list[-1].args[0]
            self.assertEqual(request.full_url, location)
            self.assertFalse(request.has_header('Authorization'))
            self.assertFalse(request.has_header('Cookie'))

    def test_received_hash_detects_body_rewrite_before_retention(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw).resolve()
            class Rewritten(Response):
                def __exit__(self, *args):
                    result = super().__exit__(*args)
                    (directory/'master.skt.body').write_bytes(b'changed after network receipt')
                    return result
            opener = Mock()
            opener.open.return_value = Rewritten(b'original received metadata')
            with patch.object(M.urllib.request, 'build_opener', return_value=opener):
                item, _ = M.fetch_one(*M.RESOURCES[0], directory)
            self.assertFalse(item['response_complete'])
            self.assertEqual(item['body_retention_failure']['type'], 'ValueError')
            self.assertEqual(Path(item['body']['path']).read_bytes(), b'changed after network receipt')

    def test_default_TLS_checks_and_header_whitelist_are_preserved(self):
        opener = Mock()
        opener.open.side_effect = lambda *args, **kwargs: Response(headers={
            'Content-Type':'application/json', 'Set-Cookie':'fixture-secret-cookie',
            'Authorization':'fixture-secret-authorization'})
        with tempfile.TemporaryDirectory() as raw, patch.object(M.urllib.request, 'build_opener', return_value=opener) as builder:
            result = M.capture(Path(raw).resolve()/'capture')
            for call in builder.call_args_list:
                handler = call.args[1]
                self.assertEqual(handler._context.verify_mode, ssl.CERT_REQUIRED)
                self.assertTrue(handler._context.check_hostname)
            encoded = json.dumps(result)
            self.assertNotIn('fixture-secret', encoded)
            self.assertTrue(all(item['response_headers']=={'Content-Type':'application/json'} for item in result['resources']))

    def test_late_EOF_is_rejected_with_literal_clock(self):
        clock = [0.0]
        literal = b'fixed metadata deadline fixture\n'
        class LateEOF(Response):
            calls = 0
            def read(self, amount):
                self.calls += 1
                clock[0] = 1.0 if self.calls == 1 else 46.0
                return literal if self.calls == 1 else b''
        opener = Mock()
        opener.open.return_value = LateEOF(literal)
        with tempfile.TemporaryDirectory() as raw, patch.object(M.urllib.request, 'build_opener', return_value=opener), patch.object(M.time, 'monotonic', side_effect=lambda: clock[0]):
            item, _ = M.fetch_one(*M.RESOURCES[0], Path(raw).resolve())
        self.assertFalse(item['response_complete'])
        self.assertEqual(item['failure']['type'], 'TimeoutError')
        self.assertEqual(item['elapsed_seconds'], 46.0)
        self.assertEqual(item['body']['sha256'], hashlib.sha256(literal).hexdigest())

    def test_close_exception_cannot_leave_a_successful_response(self):
        class BrokenClose(Response):
            def __exit__(self, *args):
                super().__exit__(*args)
                raise OSError('literal response-close failure')
        opener = Mock()
        opener.open.return_value = BrokenClose()
        with tempfile.TemporaryDirectory() as raw, patch.object(M.urllib.request, 'build_opener', return_value=opener):
            item, _ = M.fetch_one(*M.RESOURCES[0], Path(raw).resolve())
            self.assertFalse(item['response_complete'])
            self.assertEqual(item['failure']['type'], 'OSError')


class WindMetadataOwnershipTests(unittest.TestCase):
    fixture = Path(__file__).resolve().with_name('worker_fixture.py')

    def command(self, mode):
        return lambda index, directory: [sys.executable, '-B', '-X', 'utf8',
            str(self.fixture), mode, str(directory), str(index)]

    def test_real_blocked_child_is_killed_reaped_and_partial_evidence_retained(self):
        processes = []
        real_popen = subprocess.Popen
        def launch(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            self.assertIs(kwargs['stdout'], subprocess.DEVNULL)
            self.assertIs(kwargs['stderr'], subprocess.DEVNULL)
            return process
        with tempfile.TemporaryDirectory() as raw, patch.object(M, '_worker_command', side_effect=self.command('block')), patch.object(M, 'RESOURCE_SECONDS', 1.5), patch.object(M.subprocess, 'Popen', side_effect=launch):
            directory = Path(raw).resolve()
            began = time.monotonic()
            item, receipt = M._run_resource_owned(*M.RESOURCES[0], directory, began+30)
            elapsed = time.monotonic()-began
            self.assertFalse(item['response_complete'])
            self.assertEqual(item['failure']['type'], 'ResourceDeadlineExceeded')
            self.assertTrue(item['resource_process']['timed_out'])
            self.assertTrue(item['resource_process']['killed'])
            self.assertTrue(item['resource_process']['reaped'])
            self.assertIsNotNone(processes[0].poll())
            self.assertEqual(processes[0].returncode, item['resource_process']['native_exit'])
            self.assertEqual(Path(item['body']['path']).read_bytes(), b'literal persisted prefix\n')
            self.assertEqual(Path(item['partial_worker_receipt']['path']).read_bytes(), b'{"unfinished":')
            self.assertTrue(item['body_is_only_persisted_unadmitted_bytes'])
            self.assertLess(elapsed, 1.5+M.REAP_SECONDS+1)
            self.assertEqual(hashlib.sha256(Path(receipt['path']).read_bytes()).hexdigest(), receipt['sha256'])

    def test_real_valid_child_requires_literal_receipt_and_body_binding(self):
        with tempfile.TemporaryDirectory() as raw, patch.object(M, '_worker_command', side_effect=self.command('success')):
            directory = Path(raw).resolve()
            item, _ = M._run_resource_owned(*M.RESOURCES[1], directory, time.monotonic()+60)
            self.assertTrue(item['response_complete'])
            self.assertTrue(item['resource_process']['reaped'])
            self.assertEqual(item['resource_process']['native_exit'], 0)
            self.assertFalse(item['resource_process']['timed_out'])
            self.assertEqual(item['body']['sha256'], hashlib.sha256(b'literal complete metadata\n').hexdigest())
            self.assertIn('worker_receipt', item)

    def test_cancellation_reaps_real_child_before_propagating(self):
        processes = []
        real_popen = subprocess.Popen
        def launch(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            original_wait = process.wait
            calls = [0]
            def interrupted_wait(timeout=None):
                calls[0] += 1
                if calls[0] == 1:
                    raise KeyboardInterrupt('literal cancellation fixture')
                return original_wait(timeout=timeout)
            process.wait = interrupted_wait
            return process
        with tempfile.TemporaryDirectory() as raw, patch.object(M, '_worker_command', side_effect=self.command('block')), patch.object(M.subprocess, 'Popen', side_effect=launch):
            with self.assertRaises(KeyboardInterrupt):
                M._run_resource_owned(*M.RESOURCES[0], Path(raw).resolve(), time.monotonic()+60)
            self.assertEqual(len(processes), 1)
            self.assertIsNotNone(processes[0].poll())

    def test_late_exit_zero_is_rejected_by_owner(self):
        process = Mock(pid=12345, returncode=0)
        process.wait.return_value = 0
        process.poll.return_value = 0
        clock = iter([0.0, 46.0, 46.0, 46.0])
        with tempfile.TemporaryDirectory() as raw, patch.object(M.subprocess, 'Popen', return_value=process), patch.object(M.time, 'monotonic', side_effect=lambda: next(clock)):
            item, _ = M._run_resource_owned(*M.RESOURCES[0], Path(raw).resolve(), 120)
        self.assertFalse(item['response_complete'])
        self.assertEqual(item['failure']['type'], 'ResourceDeadlineExceeded')
        self.assertEqual(item['resource_process']['native_exit'], 0)
        self.assertTrue(item['resource_process']['timed_out'])
        process.kill.assert_not_called()

    def test_exit_zero_with_bad_or_missing_worker_evidence_fails_closed(self):
        for mode in ('missing', 'malformed', 'duplicate', 'wrong-resource', 'wrong-hash', 'nonfinite', 'failure-success'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as raw, patch.object(M, '_worker_command', side_effect=self.command(mode)):
                item, _ = M._run_resource_owned(*M.RESOURCES[0], Path(raw).resolve(), time.monotonic()+60)
                self.assertFalse(item['response_complete'])
                self.assertTrue(item['resource_process']['reaped'])
                self.assertEqual(item['resource_process']['native_exit'], 0)
                self.assertIn('failure', item)
        # A writer racing the first evidence check cannot turn the second
        # receipt read into an unbounded allocation before hash rejection.
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw).resolve()
            receipt = directory/'master.skt.worker-receipt.json'
            original_open = Path.open
            original_evidence = M.evidence
            reads = []
            grown = [False]
            class Observed:
                def __init__(self, stream): self.stream = stream
                def __enter__(self): self.stream.__enter__(); return self
                def __exit__(self, *args): return self.stream.__exit__(*args)
                def read(self, amount=-1): reads.append(amount); return self.stream.read(amount)
                def __getattr__(self, name): return getattr(self.stream, name)
            def observed_open(path, mode='r', *args, **kwargs):
                stream = original_open(path, mode, *args, **kwargs)
                return Observed(stream) if Path(path)==receipt and mode=='rb' else stream
            def growing_evidence(path, limit=1024**2):
                value = original_evidence(path, limit)
                if Path(path)==receipt and not grown[0]:
                    grown[0] = True
                    with original_open(receipt, 'ab') as stream:
                        stream.write(b'x'*(3*1024**2))
                return value
            with patch.object(M, '_worker_command', side_effect=self.command('success')), patch.object(Path, 'open', observed_open), patch.object(M, 'evidence', side_effect=growing_evidence):
                item, _ = M._run_resource_owned(*M.RESOURCES[0], directory, time.monotonic()+60)
            self.assertTrue(grown[0])
            self.assertFalse(item['response_complete'])
            self.assertEqual(reads, [1048577, 1048577])
            self.assertEqual(item['failure']['type'], 'EvidenceRetentionFailure')
            self.assertTrue(item['resource_process']['reaped'])

    def test_oversized_worker_output_retains_aggregate_failure_terminal(self):
        for mode in ('oversized-receipt', 'oversized-body'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as raw, patch.object(M, '_worker_command', side_effect=self.command(mode)):
                directory = Path(raw).resolve()/'capture'
                result = M.capture(directory)
                self.assertTrue(result['capture_complete'])
                self.assertFalse(result['master_and_at_least_one_info_HTTP200'])
                self.assertTrue(all(not row['response_complete'] for row in result['resources']))
                self.assertTrue(all('evidence_retention_failures' in row for row in result['resources']))
                self.assertTrue((directory/'CAPTURE.json').is_file())
                self.assertEqual(len(result['resource_receipts']), 4)

    def test_expired_capture_does_not_launch_a_worker(self):
        with tempfile.TemporaryDirectory() as raw, patch.object(M.subprocess, 'Popen') as launch:
            item, _ = M._run_resource_owned(*M.RESOURCES[0], Path(raw).resolve(), time.monotonic()-1)
            launch.assert_not_called()
            self.assertFalse(item['response_complete'])
            self.assertFalse(item['resource_process']['reaped'])
            self.assertIsNone(item['resource_process']['pid'])

    def test_full_capture_owns_four_fixed_children_with_at_most_two_unreaped(self):
        real_popen = subprocess.Popen
        lock = threading.Lock()
        active = set()
        maximum = [0]
        created = []
        def launch(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            created.append(process)
            with lock:
                active.add(process.pid)
                maximum[0] = max(maximum[0], len(active))
            original_wait = process.wait
            def tracked_wait(timeout=None):
                result = original_wait(timeout=timeout)
                with lock:
                    active.discard(process.pid)
                return result
            process.wait = tracked_wait
            return process
        with tempfile.TemporaryDirectory() as raw, patch.object(M, '_worker_command', side_effect=self.command('success')), patch.object(M.subprocess, 'Popen', side_effect=launch):
            result = M.capture(Path(raw).resolve()/'capture', 'a'*40)
            self.assertTrue(result['capture_complete'])
            self.assertTrue(result['master_and_at_least_one_info_HTTP200'])
            self.assertEqual(len(created), 4)
            self.assertLessEqual(maximum[0], 2)
            self.assertFalse(active)
            self.assertTrue(all(p.poll() is not None for p in created))
            self.assertEqual({row['url'] for row in result['resources']}, {url for _, url in M.RESOURCES})
            self.assertTrue(all(row['resource_process']['reaped'] for row in result['resources']))
            self.assertFalse(result['source_authority_granted'])

    def test_full_capture_spawn_failures_retain_all_four_terminal_receipts(self):
        with tempfile.TemporaryDirectory() as raw, patch.object(M.subprocess, 'Popen', side_effect=OSError('fixture launch failure')):
            directory = Path(raw).resolve()/'capture'
            result = M.capture(directory)
            self.assertTrue(result['capture_complete'])
            self.assertFalse(result['master_and_at_least_one_info_HTTP200'])
            self.assertEqual(len(result['resources']), 4)
            self.assertTrue(all(row['failure']['type']=='OSError' for row in result['resources']))
            self.assertTrue(all(not row['response_complete'] and 'body' not in row for row in result['resources']))
            self.assertTrue(all(Path(row['path']).is_file() for row in result['resource_receipts']))
            self.assertEqual(json.loads((directory/'CAPTURE.json').read_bytes()), result)

    def test_native_worker_entry_and_kernel_complete_all_four_fixed_resources(self):
        with tempfile.TemporaryDirectory() as raw, patch.object(M, '_worker_command', side_effect=self.command('kernel-entry')):
            directory = Path(raw).resolve()/'capture'
            result = M.capture(directory)
            self.assertTrue(result['capture_complete'])
            self.assertTrue(result['master_and_at_least_one_info_HTTP200'])
            self.assertEqual(len(result['resources']), 4)
            self.assertTrue(all(row['response_complete'] and row['resource_process']['reaped'] for row in result['resources']))
            self.assertTrue(all(Path(row['body']['path']).read_bytes()==b'literal complete metadata\n' for row in result['resources']))
            self.assertTrue(all('worker_receipt' in row and 'partial_worker_receipt' not in row for row in result['resources']))
            self.assertFalse(result['source_authority_granted'])

    def test_unreaped_child_is_a_hard_failure_without_success_receipt(self):
        process = Mock(pid=12345, returncode=None)
        process.wait.side_effect = subprocess.TimeoutExpired('fixture', 1)
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as raw, patch.object(M.subprocess, 'Popen', return_value=process):
            directory = Path(raw).resolve()
            with self.assertRaisesRegex(RuntimeError, 'could not be reaped'):
                M._run_resource_owned(*M.RESOURCES[0], directory, time.monotonic()+60)
            process.kill.assert_called_once()
            self.assertFalse((directory/'master.skt.receipt.json').exists())
        # An executor slot freed by an unreaped-child exception cannot be
        # reused to launch the third or fourth resource.
        for attempt in range(3):
            created = []
            indices = []
            def launch(arguments, **kwargs):
                failed = Mock(pid=90000+len(created), returncode=None)
                failed.poll.return_value = None
                failed.wait.side_effect = subprocess.TimeoutExpired('unreapable fixture', 1)
                created.append(failed)
                indices.append(arguments[arguments.index('--worker-resource')+1])
                return failed
            with self.subTest(attempt=attempt), tempfile.TemporaryDirectory() as raw, patch.object(M.subprocess, 'Popen', side_effect=launch):
                with self.assertRaisesRegex(RuntimeError, 'could not be reaped'):
                    M.capture(Path(raw).resolve()/'capture')
                self.assertLessEqual(len(created), 2)
                self.assertTrue(all(index in ('0', '1') for index in indices))
                self.assertTrue(all(failed.kill.call_count==1 for failed in created))

    def test_read_before_deadline_and_close_after_deadline_is_incomplete(self):
        clock = [0.0]
        class LateClose(Response):
            def __exit__(self, *args):
                result = super().__exit__(*args)
                clock[0] = 46.0
                return result
        opener = Mock()
        opener.open.return_value = LateClose()
        with tempfile.TemporaryDirectory() as raw, patch.object(M.urllib.request, 'build_opener', return_value=opener), patch.object(M.time, 'monotonic', side_effect=lambda: clock[0]):
            item, _ = M.fetch_one(*M.RESOURCES[0], Path(raw).resolve())
            self.assertFalse(item['response_complete'])
            self.assertEqual(item['failure']['type'], 'TimeoutError')



if __name__ == '__main__':
    unittest.main(verbosity=2)
