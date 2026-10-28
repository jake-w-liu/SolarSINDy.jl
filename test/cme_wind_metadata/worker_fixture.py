"""Literal native child fixtures; kernel-entry imports are network-free."""
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import os
import sys
from unittest.mock import Mock, patch

mode, raw, index_text = sys.argv[1:]
directory = Path(raw)
index = int(index_text)
resources = (
    ('master.skt', 'https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_h0_mfi_00000000_v01.skt'),
    ('hapi-0.json', 'https://cdaweb.gsfc.nasa.gov/hapi/info?id=WI_H0_MFI%400'),
    ('hapi-1.json', 'https://cdaweb.gsfc.nasa.gov/hapi/info?id=WI_H0_MFI%401'),
    ('hapi-2.json', 'https://cdaweb.gsfc.nasa.gov/hapi/info?id=WI_H0_MFI%402'),
)
name, url = resources[index]
if mode == 'kernel-entry':
    source = Path(__file__).resolve().parents[2]/'tools/capture_wind_metadata.py'
    spec = importlib.util.spec_from_file_location('native_kernel_entry_fixture', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class Response(io.BytesIO):
        headers = {}
        def getcode(self): return 200
    opener = Mock()
    opener.open.return_value = Response(b'literal complete metadata\n')
    with patch.object(module.urllib.request, 'build_opener', return_value=opener), patch.object(sys, 'argv', [str(source), '--worker-resource', str(index), '--output', str(directory)]):
        result = module.main()
    assert result == 0 and opener.open.call_count == 1
    assert opener.open.call_args.args[0].full_url == url
    raise SystemExit(0)
if mode == 'block':
    (directory/(name+'.body')).write_bytes(b'literal persisted prefix\n')
    (directory/(name+'.worker-receipt.partial.json')).write_bytes(b'{"unfinished":')
    # Fill more than a typical pipe; the owner must use DEVNULL, not an unread PIPE.
    sys.stdout.buffer.write(b'x' * 131072)
    sys.stdout.buffer.flush()
    reader, writer = os.pipe()
    try:
        os.read(reader, 1)
    finally:
        os.close(reader)
        os.close(writer)
    raise SystemExit(9)
if mode == 'missing':
    raise SystemExit(0)
body = b'literal complete metadata\n'
target = directory/(name+'.body')
target.write_bytes(body)
record = dict(schema='wind_metadata_resource_v2', resource=name, url=url,
    started_utc='2026-01-01T00:00:00+00:00', finished_utc='2026-01-01T00:00:00.010000+00:00',
    response_complete=True, redirects=[], request_method='GET',
    TLS_certificate_and_hostname_checks_enabled=True,
    actual_GSM_vectors_requested=False, source_authority_granted=False,
    elapsed_seconds=0.01, http_status=200,
    body=dict(path=str(target), bytes=len(body), sha256=hashlib.sha256(body).hexdigest()))
if mode == 'wrong-resource':
    record['resource'] = 'different-resource'
elif mode == 'wrong-hash':
    record['body']['sha256'] = '0'*64
elif mode == 'nonfinite':
    record['elapsed_seconds'] = float('nan')
elif mode == 'failure-success':
    record['failure'] = dict(type='OSError', reason_type='NoneType')
elif mode == 'oversized-body':
    target.write_bytes(b'x' * (2*1024**2+1))
text = json.dumps(record)
if mode == 'malformed':
    text = '{broken json'
elif mode == 'duplicate':
    text = '{"response_complete": false, '+text[1:]
elif mode == 'oversized-receipt':
    text = 'x' * (1024**2+1)
(directory/(name+'.worker-receipt.json')).write_text(text+'\n', encoding='utf8')
