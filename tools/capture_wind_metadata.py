"""Bounded NASA Wind metadata capture; never requests magnetic observations."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import math
import os
import re
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

MASTER_PATH = '/pub/software/cdawlib/0SKELTABLES/wi_h0_mfi_00000000_v01.skt'
RESOURCES = (('master.skt', 'https://cdaweb.gsfc.nasa.gov' + MASTER_PATH),) + tuple(
    (f'hapi-{index}.json', f'https://cdaweb.gsfc.nasa.gov/hapi/info?id=WI_H0_MFI%40{index}')
    for index in range(3))
MAX_BYTES = 2 * 1024**2
MAX_REDIRECTS = 3
SOCKET_SECONDS = 15
RESOURCE_SECONDS = 45
REAP_SECONDS = 5
CAPTURE_SECONDS = 120
HEADER_NAMES = ('Content-Type', 'Content-Length', 'Date', 'ETag', 'Last-Modified')


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def evidence(path, limit=1024**2):
    path = Path(path)
    before = path.lstat()
    if (path.is_symlink() or getattr(before, 'st_file_attributes', 0) & 0x400
            or not path.is_file() or before.st_size > limit):
        raise ValueError('metadata evidence is not an ordinary bounded file')
    with path.open('rb') as stream:
        body = stream.read(limit + 1)
    if len(body) != before.st_size or len(body) > limit:
        raise ValueError('metadata evidence exceeds its exact byte bound')
    digest = hashlib.sha256(body).hexdigest()
    after = path.lstat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                              value.st_mtime_ns, value.st_ctime_ns)
    if identity(before) != identity(after):
        raise ValueError('metadata evidence changed during hashing')
    return dict(path=str(path), bytes=after.st_size, sha256=digest)


def validate_url(url, original):
    """Redirects must retain the fixed metadata resource, including its group."""
    if original not in {resource[1] for resource in RESOURCES}:
        raise ValueError('unknown original metadata resource')
    value = urllib.parse.urlsplit(url)
    expected = urllib.parse.urlsplit(original)
    if (value.scheme != 'https' or value.hostname not in
            ('cdaweb.gsfc.nasa.gov', 'spdf.gsfc.nasa.gov') or
            value.username is not None or value.password is not None or
            value.port not in (None, 443) or value.fragment or value.path != expected.path):
        raise ValueError('redirect is outside the fixed HTTPS metadata resource')
    if sorted(urllib.parse.parse_qsl(value.query, strict_parsing=True, keep_blank_values=True)) != sorted(
            urllib.parse.parse_qsl(expected.query, strict_parsing=True, keep_blank_values=True)):
        raise ValueError('redirect changes the fixed metadata query')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, newurl):
        return None


def fetch_one(name, original, output):
    if (name, original) not in RESOURCES:
        raise ValueError('only fixed Wind metadata resources are supported')
    started = time.monotonic()
    record = dict(schema='wind_metadata_resource_v2', resource=name, url=original, started_utc=now(),
                  response_complete=False, redirects=[], request_method='GET',
                  TLS_certificate_and_hostname_checks_enabled=True,
                  actual_GSM_vectors_requested=False, source_authority_granted=False)
    target = output / (name + '.body')
    response = None
    received_digest = None
    try:
        opener = urllib.request.build_opener(
            NoRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        current = original
        for hop in range(MAX_REDIRECTS + 1):
            validate_url(current, original)
            if time.monotonic() - started >= RESOURCE_SECONDS:
                raise TimeoutError('metadata resource deadline exceeded')
            request = urllib.request.Request(current, headers={
                'User-Agent': 'SolarSINDy-Wind-metadata-capture', 'Accept-Encoding': 'identity'})
            try:
                response = opener.open(request, timeout=SOCKET_SECONDS)
            except urllib.error.HTTPError as error:
                if error.code not in (301, 302, 303, 307, 308):
                    response = error
                else:
                    with error:
                        location = error.headers.get('Location', '')
                        next_url = urllib.parse.urljoin(current, location)
                        validate_url(next_url, original)
                        if hop == MAX_REDIRECTS:
                            raise ValueError('metadata redirect limit exceeded')
                        parsed = urllib.parse.urlsplit(next_url)
                        record['redirects'].append(dict(status=error.code, host=parsed.hostname,
                            path=parsed.path, location_sha256=hashlib.sha256(location.encode()).hexdigest()))
                        current = next_url
                    continue
            if time.monotonic() - started >= RESOURCE_SECONDS:
                raise TimeoutError('metadata resource deadline exceeded')
            break
        if response is None:
            raise RuntimeError('metadata response is unavailable')
        with response:
            record['http_status'] = response.getcode()
            headers = {}
            for key in HEADER_NAMES:
                value = response.headers.get(key)
                if value is not None:
                    if len(value) > 4096:
                        raise ValueError('metadata header exceeds bound')
                    headers[key] = value
            record['response_headers'] = headers
            declared = headers.get('Content-Length')
            if declared is not None and (not declared.isdecimal() or int(declared) > MAX_BYTES):
                raise ValueError('metadata content length exceeds bound')
            size = 0
            received_digest = hashlib.sha256()
            with target.open('xb') as stream:
                while True:
                    if time.monotonic() - started >= RESOURCE_SECONDS:
                        raise TimeoutError('metadata resource deadline exceeded')
                    chunk = response.read(min(65536, MAX_BYTES - size + 1))
                    if time.monotonic() - started >= RESOURCE_SECONDS:
                        raise TimeoutError('metadata resource deadline exceeded')
                    if not chunk:
                        break
                    if size + len(chunk) > MAX_BYTES:
                        # Retain the bounded prefix, explicitly marked incomplete.
                        prefix = chunk[:MAX_BYTES - size]
                        stream.write(prefix)
                        received_digest.update(prefix)
                        raise ValueError('metadata body exceeds bound')
                    stream.write(chunk)
                    received_digest.update(chunk)
                    size += len(chunk)
            if declared is not None and size != int(declared):
                raise ValueError('metadata body differs from content length')
            record['response_complete'] = True
    except Exception as error:
        # Neither credentials nor arbitrary redirect URLs enter diagnostics.
        record['response_complete'] = False
        record['failure'] = dict(type=type(error).__name__,
            reason_type=type(getattr(error, 'reason', None)).__name__)
    finally:
        if response is not None:
            response.close()
        if target.exists():
            try:
                record['body'] = evidence(target, limit=MAX_BYTES)
                if received_digest is None or record['body']['sha256'] != received_digest.hexdigest():
                    raise ValueError('retained body differs from received metadata bytes')
            except Exception as error:
                record['response_complete'] = False
                record['body_retention_failure'] = dict(type=type(error).__name__)
        elif received_digest is not None:
            record['response_complete'] = False
            record['body_retention_failure'] = dict(type='MissingRetainedBody')
        record['elapsed_seconds'] = time.monotonic() - started
        if record['elapsed_seconds'] >= RESOURCE_SECONDS:
            record['response_complete'] = False
            record['failure'] = dict(type='TimeoutError', reason_type='NoneType')
        record['finished_utc'] = now()
        # A killed worker may leave the staging receipt. The owner retains it
        # as partial evidence and writes a separate authoritative receipt.
        receipt = output / (name + '.worker-receipt.json')
        staging = output / (name + '.worker-receipt.partial.json')
        if receipt.exists():
            raise FileExistsError('worker receipt already exists')
        write_new(staging, record)
        os.replace(staging, receipt)
    return record, evidence(receipt)


def _worker_command(index, directory):
    return [sys.executable, '-B', '-X', 'utf8', str(Path(__file__).resolve()),
            '--worker-resource', str(index), '--output', str(directory)]


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate worker receipt key')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('nonfinite worker receipt value')


def _read_worker(name, original, directory):
    path = directory / (name + '.worker-receipt.json')
    sealed = evidence(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sealed['sha256']:
        raise ValueError('worker receipt changed before admission')
    record = json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    if (not isinstance(record, dict) or record.get('schema') != 'wind_metadata_resource_v2'
            or record.get('resource') != name or record.get('url') != original
            or record.get('request_method') != 'GET'
            or record.get('TLS_certificate_and_hostname_checks_enabled') is not True
            or record.get('actual_GSM_vectors_requested') is not False
            or record.get('source_authority_granted') is not False
            or type(record.get('response_complete')) is not bool
            or type(record.get('elapsed_seconds')) not in (int, float)
            or not math.isfinite(record['elapsed_seconds']) or record['elapsed_seconds'] < 0):
        raise ValueError('worker receipt does not match the fixed metadata contract')
    target = directory / (name + '.body')
    if target.exists():
        if record.get('body') != evidence(target, limit=MAX_BYTES):
            raise ValueError('worker receipt does not match retained body')
    elif 'body' in record or record['response_complete']:
        raise ValueError('worker receipt lacks its retained body')
    if record['response_complete'] and record['elapsed_seconds'] >= RESOURCE_SECONDS:
        raise ValueError('worker completed after its resource deadline')
    if record['response_complete'] and any(key in record for key in ('failure', 'body_retention_failure')):
        raise ValueError('worker success receipt carries a failure state')
    return record, sealed


def _run_resource_owned(name, original, directory, capture_deadline):
    """Own one fixed-resource worker; late completion never grants success."""
    if (name, original) not in RESOURCES:
        raise ValueError('only fixed Wind metadata resources are supported')
    started = time.monotonic()
    deadline = min(started + RESOURCE_SECONDS, capture_deadline - REAP_SECONDS)
    record = dict(schema='wind_metadata_resource_v2', resource=name, url=original,
        started_utc=now(), response_complete=False, redirects=[], request_method='GET',
        TLS_certificate_and_hostname_checks_enabled=True,
        actual_GSM_vectors_requested=False, source_authority_granted=False)
    process = None
    timed_out = False
    killed = False
    exitcode = None
    failure = None
    try:
        if deadline <= started:
            raise TimeoutError('capture transport deadline already expired')
        index = RESOURCES.index((name, original))
        process = subprocess.Popen(_worker_command(index, directory),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            close_fds=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1'))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
        else:
            try:
                exitcode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
        timed_out = timed_out or time.monotonic() >= deadline
    except Exception as error:
        failure = dict(type=type(error).__name__, reason_type='NoneType')
    finally:
        # This executes for cancellation as well as ordinary errors. The fixed
        # worker is this module's transport kernel and creates no descendants.
        if process is not None:
            if process.poll() is None:
                process.kill()
                killed = True
            try:
                exitcode = process.wait(timeout=REAP_SECONDS)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError('metadata worker could not be reaped') from error
    worker_path = directory / (name + '.worker-receipt.json')
    partial_path = directory / (name + '.worker-receipt.partial.json')
    if not timed_out and failure is None and exitcode == 0:
        try:
            record, _ = _read_worker(name, original, directory)
        except Exception as error:
            failure = dict(type=type(error).__name__, reason_type='NoneType')
    elif failure is None:
        failure = dict(type='ResourceDeadlineExceeded' if timed_out else 'WorkerProcessFailure',
                       reason_type='NoneType')
    if failure is not None:
        record['response_complete'] = False
        record['failure'] = failure
    retention_failures = {}
    for label, path in (('worker_receipt', worker_path), ('partial_worker_receipt', partial_path)):
        try:
            if path.exists():
                record[label] = evidence(path)
        except Exception as error:
            retention_failures[label] = dict(type=type(error).__name__)
    target = directory / (name + '.body')
    try:
        if target.exists():
            record['body'] = evidence(target, limit=MAX_BYTES)
            if failure is not None:
                record['body_is_only_persisted_unadmitted_bytes'] = True
    except Exception as error:
        record.pop('body', None)
        retention_failures['body'] = dict(type=type(error).__name__)
    if retention_failures:
        record['response_complete'] = False
        record['failure'] = dict(type='EvidenceRetentionFailure', reason_type='NoneType')
        record['evidence_retention_failures'] = retention_failures
    record['resource_process'] = dict(pid=process.pid if process is not None else None,
        native_exit=exitcode, timed_out=timed_out, killed=killed,
        reaped=process is not None and process.returncode is not None,
        transport_deadline_seconds=RESOURCE_SECONDS, kill_reap_grace_seconds=REAP_SECONDS)
    record['owner_elapsed_seconds'] = time.monotonic() - started
    record['owner_finished_utc'] = now()
    receipt = directory / (name + '.receipt.json')
    write_new(receipt, record)
    return record, evidence(receipt)


def capture(output, claimed_head=None):
    if claimed_head is not None and (not isinstance(claimed_head, str)
                                    or not re.fullmatch('[0-9a-f]{40}', claimed_head)):
        raise ValueError('claimed source head must be forty lowercase hexadecimal characters')
    requested = Path(output)
    if not requested.is_absolute() or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', requested.name):
        raise ValueError('output must be an absolute path with a portable directory name')
    # Test-owned/system temporary roots can have aliases on macOS or Windows.
    # Canonicalize the existing parent, then create a new exclusive child.
    parent = requested.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError('metadata output parent must be an existing directory')
    directory = parent / requested.name
    directory.mkdir()
    started = now()
    deadline = time.monotonic() + CAPTURE_SECONDS
    with ThreadPoolExecutor(max_workers=2) as pool:
        items = list(pool.map(lambda resource: _run_resource_owned(*resource, directory, deadline), RESOURCES))
    records = [item[0] for item in items]
    usable = lambda item: (item['response_complete'] and item.get('http_status') == 200
                           and item.get('body', {}).get('bytes', 0) > 0)
    transport_usable = usable(records[0]) and any(usable(item) for item in records[1:])
    terminal = dict(schema='wind_metadata_transport_capture_v1', capture_complete=True,
        started_utc=started, finished_utc=now(), claimed_source_head=claimed_head,
        controller_sha256=evidence(Path(__file__).resolve())['sha256'],
        resources=records, resource_receipts=[item[1] for item in items],
        master_and_at_least_one_info_HTTP200=transport_usable,
        MAX_BYTES_per_resource=MAX_BYTES, maximum_concurrent_requests=2,
        per_resource_transport_deadline_seconds=RESOURCE_SECONDS,
        kill_reap_grace_seconds=REAP_SECONDS, capture_transport_deadline_seconds=CAPTURE_SECONDS,
        actual_GSM_vectors_requested=False, model_execution_performed=False,
        source_authority_granted=False, scientific_qualification_granted=False,
        metadata_semantics_independently_admitted=False)
    write_new(directory / 'CAPTURE.json', terminal)
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-head', default=None)
    parser.add_argument('--worker-resource', type=int, choices=range(len(RESOURCES)), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_resource is not None:
        directory = args.output.resolve(strict=True)
        if not args.output.is_absolute() or not directory.is_dir():
            parser.error('worker output must be an existing absolute capture directory')
        fetch_one(*RESOURCES[args.worker_resource], directory)
        return 0
    terminal = capture(args.output, args.source_head)
    print(json.dumps({key: terminal[key] for key in (
        'capture_complete', 'master_and_at_least_one_info_HTTP200',
        'actual_GSM_vectors_requested', 'source_authority_granted',
        'metadata_semantics_independently_admitted')}))
    return 0 if terminal['master_and_at_least_one_info_HTTP200'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
