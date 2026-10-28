"""Independent transport fixtures; no real NASA responses or authority grants."""
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import ssl
import tempfile
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
