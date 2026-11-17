"""Offline full-source evidence verification and host-local materialization.

No network, fitting, actual-input access, permit issuance, or Julia invocation.
The command consumes independently acquired original evidence, never CI_success.
"""
from pathlib import Path, PurePosixPath, PureWindowsPath
from datetime import datetime
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import tomllib
import unicodedata
import uuid
import zipfile
import zlib

sys.dont_write_bytecode = True
META = 4 * 1024**2
RAW = 64 * 1024**2
GIB = 1024**3
CANONICAL_NATIVE_SECONDS = 21600
BRANCH = 'codex/acyclic-b-platform-20261109'
RUNTIMES = ('1.12.6', '1.13.1')
PLATFORMS = ('ubuntu-latest', 'windows-latest', 'macos-latest')
NATIVE_PLATFORMS = ('ubuntu-latest', 'windows-latest', 'macos-26-intel')
PROOF_FIELDS = {'complete','head','tree','source_files','source_sha256','commit_object_ids','all_clean_and_checkout_filters_match'}
COMPONENT_FIELDS = {'schema','complete','evidence_mode','head','source_sha256','harness','native_jobs','docs_and_containers',
                    'root_qualification_sha256','source_proof_sha256','canonical_harness_sha256','software_qualification_only',
                    'actual_panel_fit_or_outcome_qualified','serving_promotion'}
JOBS17 = ('smoke','scanner','independent2272','public-baseline','public-candidate','cache514','original','stage4',
          'stream-baseline','stream-candidate','training362','warm44','tenk-baseline','tenk-candidate','grammar-producer','large79','disk-fit')
PREFIX = ('Julia executable found','Project.toml is present and nonempty','src/ contains Julia source','test/runtests.jl exists',
          'CODE_NOTES.md records implementation correctness context',
          'TEST_REPORT.md records test coverage, tolerances, oracles, or anti-false-test checks')
SUFFIX = ('tests contain @testset and test assertions','no obvious tautological tests found',
          'randomness appears to use explicit seed/RNG','no TODO/FIXME/HACK markers in source/tests',
          'Pkg.test() passed','examples/experiments.jl executed')
WARNINGS = ('loose tolerances found; confirm each is justified','paper_dst_superlearner/data directory not found',
            'paper_dst_superlearner/figs directory not found')
_DEADLINE = None


class SourceQualificationError(ValueError):
    def __init__(self, code, detail):
        self.code, self.detail = code, str(detail)
        super().__init__(code + ': ' + self.detail)


def _need(value, code, detail):
    if not value:
        raise SourceQualificationError(code, detail)


def _clock():
    _need(_DEADLINE is None or time.monotonic_ns() < _DEADLINE,
          'partial_or_late_publication', 'original operation deadline exhausted')


def _scope(**facts):
    return dict(facts, source_qualified=False, software_qualified=False, scientific_qualified=False)


def _hex(value, width=64):
    return type(value) is str and re.fullmatch('[0-9a-f]{' + str(width) + '}', value) is not None


def _exact(value, fields, code='input_error'):
    _need(type(value) is dict and set(value) == set(fields), code, 'closed object fields differ')


def _integer(value, low, high, code='input_error'):
    _need(type(value) is int and low <= value <= high, code, 'integer token/bound differs')
    return value


def _safe_name(name, code='host_local_path_or_chain_rewrite', directory=False):
    _need(type(name) is str and bool(name), code, 'relative member name required')
    clean = name[:-1] if directory and name.endswith('/') else name
    parsed = PurePosixPath(clean)
    _need(str(parsed) == clean and not parsed.is_absolute() and '\\' not in clean and ':' not in clean and
          unicodedata.normalize('NFC', clean) == clean and
          all(part not in ('','.','..','.git') for part in clean.split('/')), code, 'unsafe or non-NFC relative path')
    for part in parsed.parts:
        _need(part == part.rstrip(' .') and all(ord(c) >= 32 and ord(c) != 127 for c in part) and
              not re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])', part.split('.')[0], re.I), code,
              'device/control/ambiguous member name')
    return clean


def _ordinary(path, directory=False, single=True):
    path = Path(path)
    code = 'host_local_path_or_chain_rewrite'
    _need(path.is_absolute() and path == path.resolve(strict=True), code, 'canonical absolute path required')
    for part in (path, *path.parents):
        value = part.lstat()
        _need(not stat.S_ISLNK(value.st_mode) and not getattr(value, 'st_file_attributes', 0) & 0x400,
              code, 'symlink/reparse ancestor rejected')
    value = path.lstat()
    _need(stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode), code, 'ordinary file/directory required')
    if not directory and single:
        _need(value.st_nlink == 1, code, 'hard-linked control rejected')
    return value


def _file_sha(path, limit=None):
    before = _ordinary(path)
    _need(limit is None or before.st_size <= limit, 'input_error', 'file exceeds bound')
    digest, total = hashlib.sha256(), 0
    identity = lambda x: (x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns)
    with Path(path).open('rb') as stream:
        _need(identity(os.fstat(stream.fileno())) == identity(before), 'host_local_path_or_chain_rewrite', 'file changed before hash')
        while True:
            _clock(); raw = stream.read(65536); _clock()
            if not raw:
                break
            total += len(raw)
            _need(total <= before.st_size, 'host_local_path_or_chain_rewrite', 'file grew during hash')
            digest.update(raw)
        _need(identity(os.fstat(stream.fileno())) == identity(before), 'host_local_path_or_chain_rewrite', 'file changed during hash')
    _need(total == before.st_size and identity(_ordinary(path)) == identity(before),
          'host_local_path_or_chain_rewrite', 'file changed after hash')
    return digest.hexdigest()


def _bytes(path, limit=META):
    pin = _file_sha(path, limit)
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    _need(len(raw) <= limit and hashlib.sha256(raw).hexdigest() == pin,
          'host_local_path_or_chain_rewrite', 'bounded file changed while read')
    return raw


def _json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            _need(key not in value, 'input_error', 'duplicate JSON key')
            value[key] = item
        return value
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda _: _need(False,'input_error','nonfinite JSON token'))
    except SourceQualificationError:
        raise
    except (ValueError, TypeError, UnicodeError) as error:
        raise SourceQualificationError('input_error', 'invalid bounded JSON') from error


def _canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False) + '\n').encode('utf8')


def _put(path, value):
    raw = _canonical(value)
    _need(len(raw) <= META, 'partial_or_late_publication', 'metadata output bound')
    with Path(path).open('xb') as stream:
        _need(stream.write(raw) == len(raw), 'partial_or_late_publication', 'short metadata write')
        stream.flush(); os.fsync(stream.fileno())
    _clock()
    return {'path':str(Path(path)), 'sha256':hashlib.sha256(raw).hexdigest()}


def _load(path, name, expected_sha):
    _need(_file_sha(path) == expected_sha, 'dependency_mismatch', 'reviewed source dependency differs')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _need(_file_sha(path) == expected_sha, 'dependency_mismatch', 'dependency changed during file-backed import')
    return module


def parse_component_members(source_bytes):
    code = 'component_closure_mismatch'
    _need(type(source_bytes) is bytes and len(source_bytes) <= META, code, 'bounded Julia source bytes required')
    text = source_bytes.decode('utf8')
    starts = list(re.finditer(r'(?m)^const COMPONENT_SOURCE_FILES=\(\r?$', text))
    _need(len(starts) == 1, code, 'one literal component declaration required')
    tail = text[starts[0].end():]
    end = re.search(r'(?m)^\)\r?$', tail)
    _need(end is not None, code, 'closed component declaration required')
    rows = []
    for line in tail[:end.start()].splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'\s*"([A-Za-z0-9_./-]+)",\s*', line)
        _need(match is not None, code, 'component names must be plain literal strings')
        rows.append(_safe_name(match.group(1), code))
    _need(len(rows) == 136 and rows == sorted(rows) and len(set(rows)) == 136 and
          'tools/actual_b_chain/BMatchedReference342.jl' in rows, code, 'exact sorted136 component closure required')
    return tuple(rows)


def _proof(value):
    code = 'cross_host_source_proof_gap'
    _exact(value, PROOF_FIELDS, code)
    _need(value['complete'] is True and value['all_clean_and_checkout_filters_match'] is True and
          _hex(value['head'],40) and _hex(value['tree'],40), code, 'positive exact7 source proof required')
    _integer(value['source_files'],1,100000,code)
    shas, oids = value['source_sha256'], value['commit_object_ids']
    _need(type(shas) is dict and type(oids) is dict and len(shas) == value['source_files'] and set(shas) == set(oids),
          code, 'complete source/OID maps required')
    folded = set()
    for name, pin in shas.items():
        _safe_name(name,code)
        _need(name.casefold() not in folded and _hex(pin) and _hex(oids[name],40), code, 'source map alias/hash differs')
        folded.add(name.casefold())
    return value


def _git(checkout, args, *, data=None, limit=META):
    _clock()
    env = dict(os.environ)
    allowed = {'GIT_PAGER':'cat','GIT_TERMINAL_PROMPT':'0','GIT_OPTIONAL_LOCKS':'0'}
    _need(all(key in allowed and text == allowed[key] for key,text in env.items() if key.upper().startswith('GIT_')),
          'cross_host_source_proof_gap', 'inherited Git authority override rejected')
    env = {key:text for key,text in env.items() if not key.upper().startswith('GIT_')}
    env.update(GIT_TERMINAL_PROMPT='0',GIT_OPTIONAL_LOCKS='0',GIT_PAGER='cat')
    deadline = min(_DEADLINE or time.monotonic_ns()+120*10**9, time.monotonic_ns()+120*10**9)
    # Output is written to ordinary bounded temporary files, never accumulated by
    # communicate(). Large blobs/filters are still bounded by their source size.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='sourceq-git-') as directory:
        out, err = Path(directory)/'stdout', Path(directory)/'stderr'
        with out.open('xb') as stdout, err.open('xb') as stderr:
            process = subprocess.Popen(['git','--no-optional-locks','-C',str(checkout),*args],
                stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env)
            try:
                process.communicate(input=data, timeout=max(.001,(deadline-time.monotonic_ns())/1e9))
            except BaseException:
                process.kill(); process.wait(timeout=20); raise
        _need(time.monotonic_ns() < deadline and process.returncode == 0 and out.stat().st_size <= limit and err.stat().st_size <= META,
              'cross_host_source_proof_gap', 'native Git failed, exceeded output cap or completed late')
        result = _bytes(out,limit)
    _clock()
    return result


def verify_proof_join(linux7, host7, checkout):
    linux7, host7 = _proof(linux7), _proof(host7)
    code = 'cross_host_source_proof_gap'
    _need(linux7['head'] == host7['head'] and linux7['tree'] == host7['tree'] and
          linux7['commit_object_ids'] == host7['commit_object_ids'], code, 'HEAD/tree/member/OID join differs')
    differences = [name for name in linux7['source_sha256'] if linux7['source_sha256'][name] != host7['source_sha256'][name]]
    _need(set(differences) <= {'test/runtests.jl'}, code, 'unapproved cross-host raw-byte difference')
    facts = _scope(complete_as_structural_join=True, head=host7['head'],tree=host7['tree'],
                   host_checkout_verified=False,raw_map_differences=differences)
    if checkout is None:
        return facts
    checkout = Path(checkout).resolve(strict=True); _ordinary(checkout,True)
    config = _git(checkout,['config','--null','--list'])
    for item in config.split(b'\0'):
        key,_,value = item.partition(b'\n')
        _need(key.lower() != b'core.fsmonitor' or value.lower() in (b'false',b'0',b'no',b'off'),code,'unreviewed fsmonitor rejected')
    _need(_git(checkout,['rev-parse','HEAD']).decode().strip() == host7['head'] and
          _git(checkout,['rev-parse','HEAD^{tree}']).decode().strip() == host7['tree'] and
          not _git(checkout,['status','--porcelain=v1','-z','--untracked-files=all']),code,'actual clean host HEAD/tree differs')
    index = _git(checkout,['ls-files','--stage','-z'])
    tree = _git(checkout,['ls-tree','-r','-z','--full-tree','HEAD'])
    def records(raw, indexed):
        _need(raw.endswith(b'\0'),code,'complete index/tree framing required'); result = {}
        for row in raw[:-1].split(b'\0'):
            _need(row.count(b'\t')==1,code,'regular source record framing differs')
            header,name = row.split(b'\t',1); parts = header.split(b' ');name=name.decode('utf8');_safe_name(name,code)
            _need(len(parts)==3 and parts[0] in (b'100644',b'100755') and (not indexed or parts[2]==b'0') and
                  (indexed or parts[1] == b'blob') and name not in result,code,'stage0 regular source required')
            result[name] = (parts[0].decode(),(parts[1] if indexed else parts[2]).decode())
        return result
    indexed, committed = records(index,True), records(tree,False)
    _need(indexed == committed and {name:row[1] for name,row in indexed.items()} == host7['commit_object_ids'],code,'complete regular index/committed OIDs differ')
    names = sorted(indexed); attrs = _git(checkout,['check-attr','-z','--stdin','filter'],data=b''.join(n.encode()+b'\0' for n in names))
    fields = attrs.split(b'\0')
    _need(fields[-1] == b'' and len(fields) == 3*len(names)+1 and
          all(fields[i*3].decode() == name and fields[i*3+1] == b'filter' and fields[i*3+2] in (b'unspecified',b'unset')
              for i,name in enumerate(names)),code,'external clean/smudge source helpers rejected')
    for name in names:
        path = checkout/name; pin = host7['source_sha256'][name]; _need(_file_sha(path) == pin,code,'host raw source differs: '+name)
        raw = _bytes(path, max(META,path.stat().st_size));oid = indexed[name][1]
        blob = _git(checkout,['cat-file','blob',oid],limit=max(len(raw),1))
        if name == 'test/runtests.jl':
            _need(raw == blob or (raw.replace(b'\r\n',b'\n') == blob and b'\r' not in blob),code,'runtests is not proved LF/CRLF blob')
            _need(linux7['source_sha256'][name] in (hashlib.sha256(blob).hexdigest(),hashlib.sha256(raw).hexdigest()),code,'Linux runtests SHA not the proved blob/host bytes')
        else:
            _need(blob == raw,code,'unexpected committed blob/raw difference')
        _need(_git(checkout,['hash-object','--path='+name,'--stdin'],data=raw).decode().strip() == oid and
              _git(checkout,['cat-file','--filters','--path='+name,oid],limit=max(1,len(raw))) == raw,code,'actual clean/checkout filters differ')
    _need(_git(checkout,['ls-files','--stage','-z']) == index and _git(checkout,['ls-tree','-r','-z','--full-tree','HEAD']) == tree and
          not _git(checkout,['status','--porcelain=v1','-z','--untracked-files=all']) and
          _git(checkout,['config','--null','--list']) == config,code,'host inventory/status/config changed during join')
    for name in names:
        _need(_file_sha(checkout/name) == host7['source_sha256'][name],code,'host source changed after filter check')
    return dict(facts,host_checkout_verified=True,config_sha256=hashlib.sha256(config).hexdigest())


def verify_run_identity(run, *, expected_head, run_id, workflow_id, event, branch):
    _need(type(run) is dict and type(run.get('id')) is int and run['id'] == run_id and run.get('head_sha') == expected_head and
          run.get('workflow_id') == workflow_id and run.get('event') == event and run.get('head_branch') == branch and
          type(run.get('run_attempt')) is int and run['run_attempt'] == 1 and
          run.get('path') == '.github/workflows/'+('ci.yml' if event == 'workflow_dispatch' else 'posix-owner-v3.yml'),
          'wrong_original_identity','original run/workflow/head/ref/attempt differs')
    return run


def verify_job_inventory(jobs, expected_names, run_id):
    _need(type(jobs) is list and len(jobs) == len(expected_names), 'incomplete_inventory','exact complete job inventory required')
    _need(all(type(x) is dict and type(x.get('id')) is int and x['id'] > 0 for x in jobs) and
          len({x['id'] for x in jobs}) == len(jobs) and len({x.get('name') for x in jobs}) == len(jobs) and
          {x.get('name') for x in jobs} == set(expected_names), 'duplicate_or_swapped_job','duplicate/swapped/unknown job')
    _need(all(x.get('run_id') == run_id for x in jobs),'wrong_original_identity','job belongs to another run')
    return {row['name']:row for row in jobs}


def _timestamp(value):
    _need(type(value) is str and bool(value),'unsuccessful_hosted_command','missing original step timestamp')
    try:
        parsed = datetime.fromisoformat(value.replace('Z','+00:00'))
        _need(parsed.tzinfo is not None,'unsuccessful_hosted_command','timezone missing')
        return parsed
    except ValueError as error:
        raise SourceQualificationError('unsuccessful_hosted_command','invalid original step timestamp') from error


def verify_hosted_job(run, job, raw, required_steps):
    _need(type(raw) is bytes and len(raw) <= RAW and bool(raw),'coherent_header_raw_counterfeit','original bounded nonempty job log required')
    _need(job.get('run_id') == run.get('id'),'wrong_original_identity','job/run identity differs')
    _need(not any(key in job for key in ('pid','native','leader_reaped','ownership_boundary_closed','all_descendants_os_reaped')),
          'fabricated_native_job_facts','direct hosted job is not a Julia OS owner receipt')
    _need(job.get('status') == 'completed' and job.get('conclusion') == 'success' and
          type(job.get('id')) is int and job['id'] > 0,'unsuccessful_hosted_command','job did not complete successfully')
    steps = job.get('steps');_need(type(steps) is list,'unsuccessful_hosted_command','original job step inventory absent')
    job_start,job_end=_timestamp(job.get('started_at')),_timestamp(job.get('completed_at'))
    _need(job_start<=job_end,'unsuccessful_hosted_command','job completed before start')
    projections = []
    for name in required_steps:
        selected = [x for x in steps if x.get('name') == name]
        missing = 'missing_second_strict_docs' if name.startswith('Build the manual') else 'filtered_or_incomplete_execution'
        _need(len(selected) == 1,missing,'required command absent or duplicated: '+name)
        row = selected[0]
        _need(row.get('status') == 'completed' and row.get('conclusion') == 'success',
              'unsuccessful_hosted_command','required hosted command failed/skipped: '+name)
        _need(job_start<=_timestamp(row.get('started_at'))<=_timestamp(row.get('completed_at'))<=job_end,
              'unsuccessful_hosted_command','required step is outside original job interval')
        _integer(row.get('number'),1,1000,'unsuccessful_hosted_command')
        projections.append({key:row[key] for key in ('name','number','status','conclusion','started_at','completed_at')})
    ordered=sorted(projections,key=lambda x:x['number'])
    _need([x['name'] for x in ordered]==list(required_steps) and len({x['number'] for x in ordered})==len(ordered) and
          all(left['number']<right['number'] and _timestamp(left['completed_at'])<=_timestamp(right['started_at'])
              for left,right in zip(ordered,ordered[1:])),
          'unsuccessful_hosted_command','required command numbers/timestamps overlap or reverse')
    return _scope(run_id=run['id'],job_id=job['id'],name=job['name'],head=run['head_sha'],raw_sha256=hashlib.sha256(raw).hexdigest(),
                  required_steps=projections,exit_projection=dict(native_exit=0,timed_out=False,basis='github_job_and_required_steps_success'))


def verify_harness_log(raw, proof):
    _proof(proof);_need(type(raw) is bytes and len(raw) <= 16*1024**2,'wrong_harness_names','bounded original H log required')
    text = raw.decode('utf8');names = list(PREFIX)+['parse ok: '+n for n in sorted(proof['source_sha256']) if n.endswith('.jl')]+list(SUFFIX)
    actual = re.findall(r'(?m)^[ \t]*\[PASS\] ([^\r\n]+)\r?$',text)
    _need(actual == names,'wrong_harness_names','ordered exact current harness checks differ')
    warnings = re.findall(r'(?m)^[ \t]*\[WARN\] ([^\r\n]+)\r?$',text)
    _need(warnings == list(WARNINGS) and '[FAIL] ' not in text,'harness_warning_or_counter_counterfeit','original warnings/failures differ')
    counts = {'PASS':len(names),'WARN':3,'FAIL':0}
    for key,count in counts.items():
        _need(re.findall(r'(?m)^'+key+r': ([0-9]+)\r?$',text) == [str(count)],
              'harness_warning_or_counter_counterfeit','exact integer summary differs')
    return _scope(counts=counts,names=names,named_checks_sha256=hashlib.sha256(_canonical(names)).hexdigest(),raw_sha256=hashlib.sha256(raw).hexdigest())


def _helper():
    path = Path(__file__).with_name('sourceq_full_scale_verifier.py')
    return _load(path,'sourceq_portable_full_scale',_file_sha(path))


def verify_workload_inventory(workloads, expected_head=None):
    try:
        return _helper().verify_workload_inventory(workloads,expected_head)
    except ValueError as error:
        raise SourceQualificationError(getattr(error,'code','resource_workload_gap'),str(error)) from error


def verify_agreement_report(report, *, policy_sha256, layouts=('ground-only','mixed'), ram_bytes=8*GIB):
    try:
        return _helper().verify_agreement_report(report,policy_sha256=policy_sha256,layouts=layouts,ram_bytes=ram_bytes)
    except ValueError as error:
        raise SourceQualificationError(getattr(error,'code','agreement48_gap'),str(error)) from error


def verify_component_document(data, *, mode, head, component_sources, root_sha, proof_sha, harness_sha, harness_counts):
    _need(mode in ('synthetic_only','software_qualified'),'mode_or_authority_escalation','unsupported component mode')
    _exact(data,COMPONENT_FIELDS,'mode_or_authority_escalation')
    _need(type(component_sources) is dict and len(component_sources)==136 and all(_hex(x) for x in component_sources.values()),
          'component_closure_mismatch','the declared exact136 component map is required')
    _need(data['schema'] == 'b_reference_component_software_qualification_v1' and data['complete'] is True and
          data['evidence_mode'] == mode and data['head'] == head and data['source_sha256'] == component_sources and
          data['root_qualification_sha256'] == root_sha and data['source_proof_sha256'] == proof_sha and
          data['canonical_harness_sha256'] == harness_sha and data['software_qualification_only'] is True and
          data['actual_panel_fit_or_outcome_qualified'] is False and data['serving_promotion'] is False,
          'mode_or_authority_escalation','component lineage/mode/scope differs')
    _exact(data['harness'],('counts','all17_independent_workers_complete','unfiltered_pkg_test_complete','experiments_entry_complete'),
           'mode_or_authority_escalation')
    _need(data['harness']['counts'] == harness_counts and all(type(data['harness']['counts'].get(k)) is int for k in ('PASS','WARN','FAIL')) and
          harness_counts.get('FAIL') == 0 and all(data['harness'][k] is True for k in
              ('all17_independent_workers_complete','unfiltered_pkg_test_complete','experiments_entry_complete')),
          'mode_or_authority_escalation','component execution gates differ')
    expected_native = {'Julia '+v+' / '+p for v in RUNTIMES for p in PLATFORMS}
    for field,expected in (('native_jobs',expected_native),('docs_and_containers',{'Documentation build','Julia 1.12 deployment images'})):
        rows = data[field]
        _need(type(rows) is list and len(rows) == len(expected) and {x.get('name') for x in rows} == expected,
              'mode_or_authority_escalation','component exact eight cells differ')
        for row in rows:
            _need(not set(row)-{'name','conclusion','receipt_sha256','log_sha256'},'fabricated_native_job_facts','component cell cannot manufacture OS-native facts')
            _exact(row,('name','conclusion','receipt_sha256','log_sha256'),'mode_or_authority_escalation')
            _need(row['conclusion'] == 'success' and _hex(row['receipt_sha256']) and _hex(row['log_sha256']) and
                  row['receipt_sha256'] != '0'*64 and row['log_sha256'] != '0'*64,
                  'mode_or_authority_escalation','component original receipt/log pins absent')
    return _scope(complete_as_structure=True,mode=mode)


def verify_archive(path, *, api_record, run_id, expected_head):
    code = 'archive_identity_or_namespace_failure'
    _need(type(api_record) is dict and api_record.get('expired') is False and
          api_record.get('workflow_run',{}).get('id') == run_id and
          api_record['workflow_run'].get('head_sha') == expected_head and
          type(api_record.get('size_in_bytes')) is int and 0 < api_record['size_in_bytes'] <= 2*GIB and
          type(api_record.get('digest')) is str and re.fullmatch('sha256:[0-9a-f]{64}',api_record['digest']),
          code,'original API archive binding differs')
    _need(_ordinary(path).st_size == api_record['size_in_bytes'] and
          _file_sha(path,2*GIB) == api_record['digest'][7:],code,'raw ZIP bytes/API digest differs')
    before = _ordinary(path);members={};seen=set();prefixes={};expanded=0
    try:
        with zipfile.ZipFile(path) as archive:
            infos=archive.infolist();_need(len(infos)<=100000,code,'archive member count bound')
            for info in infos:
                _clock();name=info.filename;clean=_safe_name(name,code,info.is_dir());fold=clean.casefold()
                _need(fold not in seen,code,'duplicate/case-fold alias');seen.add(fold)
                parts=PurePosixPath(clean).parts
                for n in range(1,len(parts)+1):
                    prefix='/'.join(parts[:n]);key=prefix.casefold()
                    _need(key not in prefixes or prefixes[key]==prefix,code,'parent component case alias');prefixes[key]=prefix
                kind=stat.S_IFMT(info.external_attr>>16)
                _need(kind in (0,stat.S_IFREG,stat.S_IFDIR) and not info.flag_bits&1 and
                      info.compress_type in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED) and
                      (not info.is_dir() or (kind in (0,stat.S_IFDIR) and info.file_size==0)),code,'nonordinary/encrypted ZIP member')
                _need(info.is_dir() or kind in (0,stat.S_IFREG),code,'file with directory/special type')
                expanded+=info.file_size;_need(0<=info.file_size<=2*GIB and expanded<=8*GIB,code,'expanded ZIP cap')
                digest=hashlib.sha256();crc=0;size=0
                with archive.open(info) as stream:
                    while True:
                        _clock();raw=stream.read(65536);_clock()
                        if not raw:break
                        size+=len(raw);_need(size<=info.file_size,code,'member exceeded central-directory size')
                        digest.update(raw);crc=zlib.crc32(raw,crc)
                _need(size==info.file_size and crc&0xffffffff==info.CRC,code,'full member CRC/size differs')
                members[name]={'bytes':size,'sha256':digest.hexdigest(),'CRC32':info.CRC,'directory':info.is_dir()}
    except SourceQualificationError:
        raise
    except (OSError,ValueError,zipfile.BadZipFile,RuntimeError) as error:
        raise SourceQualificationError(code,'invalid original ZIP/member framing') from error
    _need(_file_sha(path,2*GIB)==api_record['digest'][7:] and _ordinary(path).st_ino==before.st_ino,
          code,'archive changed during complete member verification')
    return _scope(complete_as_archive=True,artifact_id=api_record['id'],name=api_record['name'],path=str(path),
                  sha256=api_record['digest'][7:],members=members,expanded_bytes=expanded)


def _archive_data(path):
    """Read formal bounded receipts, not deliberate invalid JSON fixtures."""
    result={}
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            formal=PurePosixPath(item.filename).name in ('TERMINAL.json','RESULT.json','state-owner.json','native-owner.json')
            if not item.is_dir() and formal and item.file_size<=META:
                result[item.filename]=_json(archive.read(item))
    return result


def _zip_bytes(path,name,limit=RAW):
    with zipfile.ZipFile(path) as archive:
        info=archive.getinfo(name)
        _need(info.file_size<=limit,'coherent_header_raw_counterfeit','raw artifact member exceeds bound')
        raw=archive.read(info)
        _need(len(raw)==info.file_size,'coherent_header_raw_counterfeit','raw artifact member truncated')
        return raw


def _positive_native(native,bound,*,expected_pid=None):
    try:
        _helper().verify_native_result(native,bound)
    except ValueError as error:
        raise SourceQualificationError('nonpositive_owned_stage',str(error)) from error
    if expected_pid is not None:
        _need(native['pid']==expected_pid,'nonpositive_owned_stage','actual launch/native PID differs')


def verify_canonical_native(native):
    """Data-only original 21,600-second canonical boundary; no qualification."""
    _positive_native(native,CANONICAL_NATIVE_SECONDS)
    return _scope(complete_as_native_structure=True,native_allowance_seconds=CANONICAL_NATIVE_SECONDS)


def _raw_unittest(raw,count):
    _need(re.findall(rb'(?m)^Ran ([0-9]+) tests in [^\r\n]+\r?$',raw)==[str(count).encode()] and
          re.findall(rb'(?m)^(OK)\r?$',raw)==[b'OK'] and
          re.search(rb'(?m)^(FAILED\b|FAIL:|ERROR:|SKIPPED\b)',raw) is None,
          'coherent_header_raw_counterfeit','original successful unittest transcript differs')


def _match_source_pins(pins,proof):
    _need(type(pins) is dict and bool(pins),'coherent_header_raw_counterfeit','source identity map missing')
    for name,pin in pins.items():
        _need(_hex(pin),'coherent_header_raw_counterfeit','source pin grammar differs')
        normalized=name.replace('\\','/')
        matches=[member for member in proof['source_sha256'] if normalized==member or normalized.endswith('/'+member)]
        _need(len(matches)==1 and proof['source_sha256'][matches[0]]==pin,
              'coherent_header_raw_counterfeit','original source pin does not match qualified proof: '+name)


_OWNER_ROLES=('tools/initial_reference_audit/owned_process.py','research/dbdt_completed_source/native_stage.py',
              'research/dbdt_completed_source/candidate_native_owner.py')
_OWNER_BASE={'pid','native_exit','timed_out','wall_seconds','ownership','descendant_cleanup'}
_OWNER_EXTRA={'leader_reaped','ownership_boundary_closed','all_descendants_os_reaped','observation_method','descendant_evidence'}
_POSITIVE_PHASE='positive_owner_observation'
_NEGATIVE_PHASE='expected_negative_pre_reap_diagnostic'


def _v3_case_checks(row,mode,darwin):
    events=row['events'];nowait=32 if darwin else 0x1000000;stopped=8 if darwin else 2
    signals=[e for e in events if e['operation']=='group_signal_entry']
    reaps=[e for e in events if e.get('first') is True and e['operation'] in
           ('terminal_wait_entry','poll_entry','direct_kill_entry','terminal_consuming_waitid')]
    terminal=[e for e in events if e['operation']=='waitid' and e['code'] in (1,2,3) and e['flags']&nowait and e['flags']&4]
    stops=[e for e in events if e['operation']=='waitid' and e['code'] in (None,5) and e['flags']&stopped and not e['flags']&(nowait|4)]
    checks=dict(one_original_signal=len(signals)==1,
        no_borrowed_signal=not any(e['operation'] in ('borrowed_group_signal','borrowed_direct_signal','repeated_group_signal') for e in events),
        signal_before_reap=len(signals)==1 and not signals[0]['reap_started'],
        no_emergency=not row['emergency'],threads_joined=row['controller_threads_joined'])
    if mode=='other_group_persistent':
        failed=reaps[0].get('snapshot',{}) if len(reaps)==1 else {}
        checks.update(real_unclosed_failure=row['error'] is not None and row['native'] is None,
            live_other_group=row.get('other_group_live_at_owner_return') is True,
            fixture_expiry_observed=row.get('fixture_expiry_observed') is True,
            live_sid_before_failed_reap=failed.get('observation_scope')=='whole_session' and failed.get('complete') is True and
                failed.get('observation_phase')==_NEGATIVE_PHASE and failed.get('positive_acceptance_eligible') is False and
                type(failed.get('session_live')) is int and failed['session_live']>=1)
        return checks
    native=row['native'];timeout=mode in ('other_group_12s','persistent_stop');code=-9 if mode=='persistent_stop' else 7
    snap=reaps[0].get('snapshot',{}) if len(reaps)==1 else {}
    checks.update(no_error=row['error'] is None,actual_exit=row['actual_direct_exit']==code,
        native_exit=isinstance(native,dict) and type(native.get('native_exit')) is int and native['native_exit']==code,
        timeout=isinstance(native,dict) and native.get('timed_out') is timeout,
        cleanup_token=isinstance(native,dict) and native.get('descendant_cleanup')=='owned POSIX session has no live members after stage',
        zero_sid_before_reap=len(reaps)==1 and snap.get('observation_scope')=='whole_session' and snap.get('complete') is True and
            type(snap.get('session_live')) is int and snap['session_live']==0,
        positive_reap_observation_budget=len(reaps)==1 and snap.get('observation_scope')=='whole_session' and
            snap.get('positive_acceptance_eligible') is True and snap.get('observation_phase')==_POSITIVE_PHASE)
    candidate=row['role']==_OWNER_ROLES[2]
    spawns=[e for e in events if e['operation']=='spawn']
    checks.update(closed_public_result=isinstance(native,dict) and set(native)==(_OWNER_BASE|_OWNER_EXTRA if candidate else _OWNER_BASE),
        actual_pid_binding=isinstance(native,dict) and type(native.get('pid')) is int and len(spawns)==1 and native['pid']==spawns[0]['pid'],
        native_wall_finite=isinstance(native,dict) and type(native.get('wall_seconds')) in (int,float) and math.isfinite(native['wall_seconds']) and native['wall_seconds']>=0,
        private_session_method=isinstance(native,dict) and native.get('ownership')=='start_new_session=True')
    checks['full_stage_deadline_observed' if timeout else 'inside_original_stage']=row['elapsed_ns']>=8000000000 if timeout else row['elapsed_ns']<8000000000
    for event in events:
        if event['operation']=='group_signal_error' and event['errno']==1:
            snap=event['snapshot']
            checks['eperm_native_zombie_only']=(snap.get('observation_scope')=='process_group' and snap.get('complete') is True and
                snap.get('group_zombie_only') is True and type(snap.get('group_live')) is int and snap['group_live']==0 and
                snap.get('session_live') is None and snap.get('session_members') is None and snap['held_leader']['zombie'] is True)
            checks['eperm_observation_budget']=snap.get('observation_scope')=='process_group' and snap.get('positive_acceptance_eligible') is True and snap.get('observation_phase')==_POSITIVE_PHASE
    if mode=='same_group_descendant':
        group=signals[0].get('snapshot',{}) if len(signals)==1 else {}
        checks['live_group_before_signal']=len(signals)==1 and group.get('observation_scope') in ('process_group','whole_session') and group.get('complete') is True and type(group.get('group_live')) is int and group['group_live']>=1
        checks['positive_group_observation_budget']=len(signals)==1 and group.get('positive_acceptance_eligible') is True and group.get('observation_phase')==_POSITIVE_PHASE
    if mode in ('persistent_stop','transient_stop') and darwin:checks['stop_only_consumed']=len(stops)>=1
    if mode=='transient_stop':
        checks['controlled_cont']=row['controller_cont_sent']
        checks['terminal_observation_before_cleanup']=bool(terminal) and bool(signals) and terminal[-1]['monotonic_ns']<signals[0]['monotonic_ns']
    if candidate:
        checks.update(candidate_boundary_closed=isinstance(native,dict) and native.get('ownership_boundary_closed') is True,
            candidate_leader_reaped=isinstance(native,dict) and native.get('leader_reaped') is True,
            candidate_descendant_reaping_unobserved=isinstance(native,dict) and native.get('all_descendants_os_reaped') is False,
            candidate_observation_method_retained=isinstance(native,dict) and all(type(native.get(k)) is str and bool(native[k]) for k in ('observation_method','descendant_evidence')))
    return checks


def _anchor_case_checks(row,candidate):
    before=row['independent_before_return_records'];first=row['producer_first_read'];events=row['events']
    reaps=[e for e in events if e['operation']=='actual_terminal_wait_entry']
    checks=dict(actual_owned_wait=row['actual_owner_wait_returned'] and len(reaps)==1,
        no_intervention=not row['test_intervention'] and not row['ownership_unclosed'],
        no_hidden_reap=not any(e['operation'].startswith('forbidden_') for e in events),
        direct_signals_zero=row['actual_owned_direct_signals']==0,live_anchor_retained=row['independent_live_anchor'] is not None,
        raw_retained=row['raw'] is not None)
    if row['mode']!='live_then_zombie_exit7':
        checks['actual_first_zombie_trigger']=before is not None and before['full']['status']==5 and before['short']['status']==5 and bool(before['full']['flags']&0x4000) and bool(before['short']['flags']&0x4000) and first is not None and first['row']['zombie'] is True and first['row']['sid'] is None
    if row['historical']:
        checks.update(old_error_retained=row['error'] is not None and row['native'] is None,
            old_exact_exit7=row['actual_owner_wait_exit']==7,old_no_fake_callback=row['owner_on_launch_calls']==0,
            old_no_numeric_group_signal=row['actual_original_group_signals']==0,
            negative_diagnostic_retained=len(reaps)==1 and reaps[0]['snapshot'].get('complete') is True and reaps[0]['snapshot'].get('positive_acceptance_eligible') is False)
        return checks
    native=row['native'];expected=-9 if row['mode']=='first_seen_zombie_signal9' else 7;proof=row['private_proof'] or {}
    checks.update(no_error=row['error'] is None,closed_result=isinstance(native,dict) and set(native)==(_OWNER_BASE|_OWNER_EXTRA if candidate else _OWNER_BASE),
        native_PID=isinstance(native,dict) and type(native.get('pid')) is int and native['pid']==row['actual_child_pid'],
        exact_exit=isinstance(native,dict) and type(native.get('native_exit')) is int and native['native_exit']==expected and row['actual_owner_wait_exit']==expected,
        untimed=isinstance(native,dict) and native.get('timed_out') is False,inside_stage=row['elapsed_ns']<8000000000,
        one_original_group=row['actual_original_group_signals']==1,one_owner_callback=row['owner_on_launch_calls']==1,
        private_zero_verified=proof.get('session_zero_live_verified') is True,
        independent_zero_before_reap=len(reaps)==1 and reaps[0]['snapshot'].get('complete') is True and reaps[0]['snapshot'].get('session_live')==0 and reaps[0]['snapshot'].get('positive_acceptance_eligible') is True,
        correct_origin=proof.get('leader_identity_origin')==('native_live' if row['mode']=='live_then_zombie_exit7' else 'owned_exec_terminal'))
    if row['mode']!='live_then_zombie_exit7':
        terminal=proof.get('launch_terminal_observation')
        checks['actual_terminal_launch_witness']=isinstance(terminal,dict) and terminal.get('si_pid')==row['actual_child_pid'] and terminal.get('si_code') in (1,2,3)
    if row['mode']=='first_seen_zombie_after_transient_stop':checks['actual_safe_CONT']=row['test_controller_CONT_sent']
    if candidate:checks['candidate_truthful_projection']=native is not None and native.get('leader_reaped') is True and native.get('ownership_boundary_closed') is True and native.get('all_descendants_os_reaped') is False
    return checks


def _fault_case_checks(row):
    events=row['events'];associations=[e for e in events if e['operation']=='test_intervention_nonreaping_child_association']
    kills=[e for e in events if e['operation']=='test_intervention_owned_direct_kill'];waits=[e for e in events if e['operation']=='test_intervention_wait_returned']
    failures=[e for e in events if e['operation']=='test_intervention_identity_unavailable'];stopped=row['independent_stopped_records']
    return dict(explicit_expected_negative_input=row['failure_recovery_test_input'] and row['intentional_identity_fault_injected'],
        genuine_stopped_fixture=stopped is not None and stopped['full']['status']==4 and stopped['short']['status']==4,
        independent_live_anchor_retained=row['independent_live_anchor'] is not None,intentional_read_error=row['error'] is not None and row['error']['errno']==5,
        no_owner_result=row['native'] is None and row['private_proof'] is None,factory_not_returned=not row['Popen_returned_to_producer'] and row['owner_on_launch_calls']==0,
        no_fake_owner_wait=not row['actual_owner_wait_returned'],actual_identity_lookup_failure=len(failures)==1 and failures[0]['error_type']=='OSError',
        nonreaping_matching_child=len(associations)==1 and associations[0]['pid']==row['actual_child_pid'],
        only_owned_direct_intervention=len(kills)==1 and kills[0]['pid']==row['actual_child_pid'] and row['actual_original_group_signals']==0 and row['actual_owned_direct_signals']==0,
        genuine_bounded_test_wait=len(waits)==1 and waits[0]['native_exit']==-9 and waits[0]['owned_child_association_verified'] is True,
        actual_postwait_ECHILD=row['test_intervention_ECHILD_observed'] is True,intervention_explicitly_negative=row['test_intervention'] and not row['ownership_unclosed'],
        no_CONT_or_positive_closure=not row['test_controller_CONT_sent'],raw_retained=row['raw'] is not None)


def verify_owner_case(row,*,family,mode,role,platform,proof,historical=False,failure_cleanup=False):
    code='coherent_header_raw_counterfeit'
    try:
        _need(family in ('v3','anchor') and role in _OWNER_ROLES and type(row) is dict and row['role']==role and row['mode']==mode,code,'case role/mode differs')
        source=role
        if historical:
            source='test/'+('posix_owner_v3' if family=='v3' else 'darwin_anchor_v1')+'/frozen/'+('candidate_native_owner.py' if role==_OWNER_ROLES[2] else 'native_stage.py')
        _need(row['source_sha256']==proof['source_sha256'][source] and row['individual_descendant_OS_reaping_observed'] is False,
              code,'case original source/reaping scope differs')
        _need(type(row['events']) is list and len(row['events'])<=4096 and row['raw'] is not None,code,'original case event/raw record absent')
        first,last=('start_ns','end_ns') if family=='v3' else ('started_ns','finished_ns')
        _need(type(row[first]) is int and type(row[last]) is int and type(row['elapsed_ns']) is int and
              row[last]>=row[first] and row['elapsed_ns']==row[last]-row[first] and 0<=row['elapsed_ns']<480000000000,
              code,'original case monotonic interval differs')
        if family=='v3':
            checks=_v3_case_checks(row,mode,platform=='macos-26-intel')
            if historical:
                _need(row['evidence_kind']=='actual_native_historical_negative' and row['expected_negative_reproduced'] is True and
                      row['new_stop_oracle_checks']==checks and all(type(x) is bool for x in row['new_stop_oracle_checks'].values()) and checks.get('stop_only_consumed') is False and
                      any(e['operation']=='waitid' and e['code']==5 for e in row['events']),code,'original historical STOP negative not reproduced')
            else:
                _need(row['evidence_kind']=='actual_native_current' and row['passed'] is True and row['checks']==checks and all(type(x) is bool for x in row['checks'].values()) and
                      bool(checks) and all(x is True for x in checks.values()),code,'original per-case V3 predicates differ')
        else:
            _need(row['historical'] is historical,code,'anchor historical scope differs')
            checks=_fault_case_checks(row) if failure_cleanup else _anchor_case_checks(row,role==_OWNER_ROLES[2])
            _need(row['passed'] is True and row['checks']==checks and all(type(x) is bool for x in row['checks'].values()) and bool(checks) and all(x is True for x in checks.values()),code,'original per-case anchor predicates differ')
            if historical:_need(row['expected_negative_reproduced'] is True,code,'original anchor historical negative absent')
            if failure_cleanup:_need(row['expected_failure_cleanup_negative_reproduced'] is True and row['evidence_kind']=='actual_Darwin_fault_recovery_expected_negative',code,'explicit fault recovery negative absent')
        return _scope(complete_as_owner_case=True,family=family,role=role,mode=mode,historical=historical,failure_cleanup=failure_cleanup)
    except SourceQualificationError:raise
    except (KeyError,TypeError,ValueError,AttributeError) as error:
        raise SourceQualificationError(code,'malformed original per-case record') from error


def _verify_owner_case_inventory(path,prefix,platform,proof,is_v3):
    family='v3' if is_v3 else 'anchor';expected=[]
    modes=('no_descendant','same_group_descendant','other_group_12s','persistent_stop','transient_stop','other_group_short','other_group_persistent') if is_v3 else (
        'live_then_zombie_exit7','first_seen_zombie_exit7','first_seen_zombie_signal9','first_seen_zombie_after_transient_stop')
    for index,role in enumerate(_OWNER_ROLES):
        for mode in modes:expected.append((str(index)+'-'+mode if is_v3 else 'current-'+str(index)+'-'+mode,mode,role,False,False))
    if platform=='macos-26-intel':
        for index,role in enumerate(_OWNER_ROLES):
            mode='transient_stop' if is_v3 else 'first_seen_zombie_exit7'
            expected.append(('old-'+str(index)+'-'+mode,mode,role,True,False))
        if not is_v3:expected.append(('failure-cleanup-identity-fault','failure_cleanup_identity_fault',_OWNER_ROLES[1],False,True))
    with zipfile.ZipFile(path) as archive:
        names=set(archive.namelist());actual={name for name in names if name.startswith(prefix) and '/' not in name[len(prefix):] and name.endswith('.json') and name!=prefix+'RESULT.json' and name!=prefix+'PREEXECUTION.json'}
        _need(actual=={prefix+name+'.json' for name,_,_,_,_ in expected},'missing_current_owner_evidence','exact original per-case inventory absent/extra')
        for name,mode,role,old,fault in expected:
            row=_json(_zip_bytes(path,prefix+name+'.json',META))
            verify_owner_case(row,family=family,mode=mode,role=role,platform=platform,proof=proof,historical=old,failure_cleanup=fault)
            reference=row['raw'];_exact(reference,('path','sha256','bytes'),'coherent_header_raw_counterfeit')
            raw_name=prefix+name+'/native.raw';_need(raw_name in names,'missing_current_owner_evidence','original per-case raw missing')
            raw=_zip_bytes(path,raw_name,4*1024**2)
            normalized=reference['path'].replace('\\','/')
            _need(normalized.endswith('/'+name+'/native.raw') and reference['sha256']==hashlib.sha256(raw).hexdigest() and
                  type(reference['bytes']) is int and reference['bytes']==len(raw),'coherent_header_raw_counterfeit','original case raw SHA/bytes/path differs')


def _matched_raw(path,member,reference):
    code='coherent_header_raw_counterfeit';_exact(reference,('path','sha256','bytes'),code)
    raw=_zip_bytes(path,member,4*1024**2)
    _need(type(reference['path']) is str and reference['path'].replace('\\','/').endswith('/'+'/'.join(member.split('/')[-2:])) and
          reference['sha256']==hashlib.sha256(raw).hexdigest() and type(reference['bytes']) is int and reference['bytes']==len(raw),
          code,'original matched owner case raw differs')
    return raw


def _matched_native(native,pid,exit_code,timeout):
    code='coherent_header_raw_counterfeit'
    _need(type(native) is dict and type(pid) is int and pid>0 and native.get('pid')==pid and
          type(native.get('native_exit')) is int and (exit_code is None or native['native_exit']==exit_code) and native.get('timed_out') is timeout and
          native.get('leader_reaped') is True and native.get('ownership_boundary_closed') is True and native.get('all_descendants_os_reaped') is False,
          code,'original matched case native/launch/exit differs')
    pairs={('start_new_session=True','owned POSIX session has no live members after stage'),
           ('CREATE_SUSPENDED -> AssignProcessToJobObject -> ResumeThread','private Windows Job Object closed with KILL_ON_JOB_CLOSE')}
    wall=native.get('wall_seconds')
    _need(type(native.get('ownership')) is str and type(native.get('descendant_cleanup')) is str and
          (native['ownership'],native['descendant_cleanup']) in pairs and type(wall) in (int,float) and
          math.isfinite(wall) and 0<=wall<180,code,'original matched case ownership/native wall differs')


def verify_lifecycle_zero_session_events(row):
    """Recheck the original event-derived summary; observes no OS process."""
    code='coherent_header_raw_counterfeit'
    _need(type(row) is dict and type(row.get('events')) is list,code,'retained lifecycle event list required')
    try:
        derived=any(event['operation']=='before_actual_terminal_wait' and event['snapshot'].get('complete') is True and
            event['snapshot'].get('session_live')==0 and event['snapshot'].get('positive_acceptance_eligible') is True for event in row['events'])
    except (KeyError,TypeError,AttributeError) as error:
        raise SourceQualificationError(code,'malformed retained lifecycle event predicate') from error
    _need(row.get('independent_zero_live_SID_before_reap') is derived and derived is True,
          code,'reported lifecycle zero-session summary lacks original event support')
    return _scope(complete_as_lifecycle_event_metadata=True,independent_zero_live_SID_before_reap=derived)

def _verify_matched_case_rows(path,result_name,result,platform,proof):
    code='coherent_header_raw_counterfeit';base=result_name[:-len('RESULT.json')];rows=result.get('native_rows')
    _need(type(rows) is list,code,'original matched native case rows absent')
    if result['schema']=='matched342_owner_small_engineering_tests_v1':
        expected=(('fail','test_attempt_marker_survives_real_failed_child_and_owner_change'),
                  ('ok','test_native_predicates_reject_modified_observations'),
                  ('callback-error','test_real_callback_exception_never_returns_native_success'),
                  ('ok','test_real_native_pid_wait_and_raw'),('sleep','test_real_timeout_is_not_positive_wait'))
        _need(len(rows)==5 and [row.get('mode') for row in rows]==[mode for mode,_ in expected] and
              result.get('target_sha256')==proof['source_sha256']['tools/actual_b_chain/run_matched342_native.py'],code,'original owner11 native inventory/source differs')
        for row,(mode,directory) in zip(rows,expected):
            raw=_matched_raw(path,base+directory+'/'+mode+'.raw',row['raw'])
            if mode=='callback-error':
                _need(type(row.get('actual_on_launch_pid')) is int and row['actual_on_launch_pid']>0 and row.get('native_result_returned') is False and
                      all(row.get(key) is False for key in ('native_exit_observed','leader_reaping_observed','ownership_boundary_closed_observed')) and
                      row.get('expected_exception')=='RuntimeError: independent callback rejection',code,'callback exception invented native closure')
            else:
                expected_exit=7 if mode=='fail' else (None if platform=='windows-latest' else -9) if mode=='sleep' else 0
                _matched_native(row['native'],row['actual_on_launch_pid'],expected_exit,mode=='sleep')
                if mode in ('fail','ok'):_need(raw==(b'PROFILE_OWNER_CHILD_FAIL\n' if mode=='fail' else b'PROFILE_OWNER_CHILD_OK\n'),code,'original harmless child output differs')
    else:
        windows=platform=='windows-latest';cases=('windows_normal','windows_exception') if windows else ('posix_normal','posix_first_term','posix_repeat_term')
        _need([row.get('case') for row in rows]==list(cases),code,'original lifecycle native inventory differs')
        for row in rows:
            _need(row.get('passed') is True and row.get('individual_descendant_OS_reaping_observed') is False,code,'original lifecycle case failed or invented descendant waits')
            directory=row['case'].replace('_','-',1);raw=_matched_raw(path,base+directory+'/native.raw',row['raw'])
            observed=_json(_zip_bytes(path,base+directory+'/OBSERVATIONS.json',META))
            _need(observed==row,code,'original lifecycle OBSERVATIONS differs from aggregate row')
            normal=row['case'].endswith('_normal')
            if windows:
                _need(row.get('evidence_kind')=='actual_Windows_Job' and row.get('exception_Job_boundary_independently_observed') is False,
                      code,'Windows exception Job closure invented')
                if normal:
                    _need(row.get('native_result_returned') is True and row.get('exception_type') is None and row.get('direct_exit_via_retained_handle')==0,
                          code,'Windows normal result/retained handle differs')
                    _matched_native(row['native'],row['actual_on_launch_pid'],0,False)
                else:
                    _need(row.get('native_result_returned') is False and row.get('native') is None and row.get('exception_type')=='RuntimeError' and
                          type(row.get('direct_exit_via_retained_handle')) is int and row['direct_exit_via_retained_handle']!=259,
                          code,'Windows exception lost original retained-handle evidence')
            else:
                verify_lifecycle_zero_session_events(row)
                _need(row.get('evidence_kind')=='actual_POSIX_lifecycle' and row.get('failure_type') is None and
                      row.get('actual_owner_terminal_wait_returned') is True and row.get('waitpid_ECHILD_observed') is True and
                      row.get('test_intervention_reap') is False and row.get('previous_handler_restored') is True and
                      row.get('independent_zero_live_SID_before_reap') is True,code,'original POSIX lifecycle reap/zero-session/restoration differs')
                if normal:
                    _need(row.get('native_result_returned') is True and row.get('exception_type') is None and row.get('actual_direct_exit_observed')==0 and row.get('actual_TERM_sends_to_self')==0,
                          code,'normal POSIX lifecycle result differs')
                    _matched_native(row['returned_native'],row['actual_child_pid'],0,False)
                else:
                    _need(row.get('native_result_returned') is False and row.get('returned_native') is None and row.get('exception_type')=='InterruptedError' and
                          row.get('actual_direct_exit_observed')==-9 and row.get('actual_TERM_sends_to_self')==(2 if row['case']=='posix_repeat_term' else 1),
                          code,'POSIX cancellation invented result or lost repeated-TERM proof')

def verify_owner_artifact(path, *, platform, proof):
    _proof(proof);data=_archive_data(path);code='missing_current_owner_evidence'
    _need(platform in NATIVE_PLATFORMS or platform in PLATFORMS or platform=='canonical',code,'unknown engineering platform')
    v3=[(name,row) for name,row in data.items() if row.get('schema')=='owned_posix_v3_registered_controls_v1' and name.endswith('TERMINAL.json')]
    anchor=[(name,row) for name,row in data.items() if row.get('schema')=='independent_darwin_anchor_registered_controls_v1' and name.endswith('TERMINAL.json')]
    matched=[(name,row) for name,row in data.items() if row.get('schema') in
             ('matched342_owner_small_engineering_tests_v1','matched342_independent_owner_lifecycle_v1')]
    _need(bool(v3 or anchor or matched),code,'registered owner/state result absent')
    state_count=None
    for terminal_name,terminal in v3+anchor:
        _need(terminal.get('complete') is True and terminal.get('source_unchanged') is True,
              'coherent_header_raw_counterfeit','registered owner terminal did not complete with stable sources')
        if 'source_pins' in terminal:_match_source_pins(terminal['source_pins'],proof)
        for name,pin in terminal.get('test_sources',{}).items():
            prefix='test/posix_owner_v3/' if (terminal_name,terminal) in v3 else 'test/darwin_anchor_v1/'
            _need(proof['source_sha256'].get(prefix+name)==pin,'coherent_header_raw_counterfeit','registered test source differs from qualified proof')
        base=terminal_name[:-len('TERMINAL.json')]
        is_v3=bool(v3) and (terminal_name,terminal) in v3
        state_name=base+'state/RESULT.json'
        _need(state_name in data,code,'independent state RESULT absent')
        state=data[state_name];_need(state.get('complete') is True,'coherent_header_raw_counterfeit','state result false')
        count=8 if is_v3 else 3;rows_count=113 if is_v3 else 72
        rows=state.get('rows')
        _need(type(rows) is list and len(rows)==rows_count and all(row.get('passed') is True for row in rows) and
              state.get('unittest_methods')==count,'coherent_header_raw_counterfeit','actual state row/method results differ')
        _raw_unittest(_zip_bytes(path,base+'state.log'),count);state_count=count
        for stage in ('state','native'):
            receipt_name=base+stage+'-owner.json'
            applicable=(platform!='windows-latest') if is_v3 else platform=='macos-26-intel'
            if stage=='native':
                field='native_POSIX_applicable' if is_v3 else 'native_Darwin_applicable'
                _need(terminal.get(field) is applicable,'coherent_header_raw_counterfeit','original native platform applicability differs')
                if not applicable:
                    _need(terminal.get('native') is None,'coherent_header_raw_counterfeit','unsupported platform invented a native result')
                    continue
            _need(receipt_name in data,code,'original positive owner receipt absent')
            receipt=data[receipt_name];native=receipt.get('native',{})
            allowance=90 if stage=='state' else 500
            _need(receipt.get('allowance_seconds')==allowance,'nonpositive_owned_stage','original fixed stage allowance differs')
            _positive_native(native,allowance,expected_pid=receipt.get('launch_pid'))
            raw=_zip_bytes(path,base+stage+'.log')
            _need(receipt.get('raw_sha256')==hashlib.sha256(raw).hexdigest() and receipt.get('raw_bytes')==len(raw),
                  'coherent_header_raw_counterfeit','owner/raw header differs')
            result_name=base+stage+'/RESULT.json';_need(result_name in data,code,'original stage result absent')
            stage_result=data[result_name]
            declared=terminal.get(stage,{})
            _need(all(declared.get(key)==value for key,value in receipt.items()) and
                  declared.get('result_sha256')==hashlib.sha256(_zip_bytes(path,result_name,META)).hexdigest() and
                  stage_result.get('complete') is True,'coherent_header_raw_counterfeit','owner/result raw join differs')
            if stage=='native':
                _verify_owner_case_inventory(path,base+'native/',platform,proof,is_v3)
                if is_v3:
                    _need(stage_result.get('actual_current_cases')==21 and
                          stage_result.get('case_passes')==[True]*21 and
                          stage_result.get('historical_stop_negative_cases')==(3 if platform=='macos-26-intel' else 0) and
                          stage_result.get('historical_negative_passes')==[True]*(3 if platform=='macos-26-intel' else 0),
                          'coherent_header_raw_counterfeit','native V3 cases are incomplete or false')
                else:
                    _need(stage_result.get('actual_native_cases')==16 and stage_result.get('current_case_passes')==[True]*12 and
                          stage_result.get('historical_negative_passes')==[True]*3 and
                          stage_result.get('failure_cleanup_negative_passes')==[True],
                          'coherent_header_raw_counterfeit','anchor actual native cases differ')
        if not is_v3:
            _need(terminal.get('state_methods')==3 and terminal.get('literal_rows')==72,
                  'coherent_header_raw_counterfeit','anchor registered counts differ')
    if matched:
        schemas={row['schema'] for _,row in matched}
        _need(schemas=={'matched342_owner_small_engineering_tests_v1','matched342_independent_owner_lifecycle_v1'},code,'both owner/lifecycle results required')
        for name,row in matched:
            _need(row.get('complete') is True and row.get('Julia_launched') is False and row.get('Q_issued') is False,
                  'coherent_header_raw_counterfeit','matched engineering result/scope differs')
            _verify_matched_case_rows(path,name,row,platform,proof)
            if row['schema']=='matched342_owner_small_engineering_tests_v1':
                _need(row.get('tests_run')==11 and row.get('failures')==row.get('errors')==row.get('skipped')==0 and
                      row.get('protected_source_pins_unchanged') is True,'coherent_header_raw_counterfeit','owner11 method outcome differs')
                raw_name=name[:-len('RESULT.json')]+'unittest.log'
                with zipfile.ZipFile(path) as archive:
                    if raw_name in archive.namelist():_raw_unittest(archive.read(raw_name),11)
            else:
                _need(row.get('unittest_methods')==3 and row.get('simulated_controls')==11 and
                      len(row.get('simulated_rows',[]))==11 and all(x.get('passed') is True for x in row['simulated_rows']) and
                      row.get('actual_POSIX_cases')==(0 if platform=='windows-latest' else 3) and
                      row.get('actual_Windows_cases')==(2 if platform=='windows-latest' else 0) and
                      row.get('source_pins_unchanged') is True,'coherent_header_raw_counterfeit','lifecycle3/11/native results differ')
                _match_source_pins(row['source_pins'],proof)
        with zipfile.ZipFile(path) as archive:
            notes=[x for x in archive.namelist() if x.endswith('ARTIFACT_NOTES.txt')]
            _need(bool(notes),code,'deliberate sparse omission/hardlink byte-scope notes absent')
            text='\n'.join(archive.read(x).decode('utf8') for x in notes)
            _need('67108865' in text and 'oversize.raw' in text and 'hardlink' in text.lower(),code,'owner artifact omission/scope note differs')
    return _scope(complete_as_owner_evidence=True,platform=platform,state_methods=state_count,families={'v3':len(v3),'anchor':len(anchor),'matched':len(matched)})


def _native_identity(path):
    value=Path(path).lstat()
    _need(not stat.S_ISLNK(value.st_mode) and not getattr(value,'st_file_attributes',0)&0x400 and
          stat.S_IFMT(value.st_mode) in (stat.S_IFREG,stat.S_IFDIR) and value.st_ino!=0,
          'partial_or_late_publication','ordinary native object identity unavailable')
    return value.st_dev,value.st_ino,stat.S_IFMT(value.st_mode)


def _quarantine_owned(path, identity, members=None):
    """Move only the observed object created by this call; never delete a path."""
    path=Path(path)
    try:
        if not os.path.lexists(path):return {'state':'absent','owned_object_moved':False}
        if identity is None or _native_identity(path)!=identity:
            return {'state':'unknown_replaced','owned_object_moved':False,'path':str(path)}
        if members is not None:
            current={p.name:_native_identity(p) for p in path.iterdir()}
            if current!={name:row['identity'] for name,row in members.items()}:
                return {'state':'unknown_members','owned_object_moved':False,'path':str(path)}
        _ordinary(path.parent,True)
        destination=path.with_name(path.name+'.unaccepted-'+uuid.uuid4().hex)
        _need(not os.path.lexists(destination),'partial_or_late_publication','fresh quarantine path required')
        try:
            path.rename(destination)
        except BaseException as error:
            if os.path.lexists(destination) and _native_identity(destination)==identity and not os.path.lexists(path):
                return {'state':'quarantined','owned_object_moved':True,'path':str(destination),'rename_error':repr(error)}
            raise
        _need(_native_identity(destination)==identity and not os.path.lexists(path),
              'partial_or_late_publication','quarantine outcome not verified')
        return {'state':'quarantined','owned_object_moved':True,'path':str(destination)}
    except BaseException as error:
        return {'state':'unknown_error','owned_object_moved':False,'path':str(path),'error':repr(error)}


def _pending_snapshot(path):
    _ordinary(path,True);identity=_native_identity(path)
    children=list(Path(path).iterdir())
    _need(0<len(children)<=8,'partial_or_late_publication','bounded flat staged metadata required')
    members={}
    for child in children:
        _safe_name(child.name);_ordinary(child)
        members[child.name]={'identity':_native_identity(child),'sha256':_file_sha(child,META)}
    return identity,members


def finalize_materialization(pending, final, *, emit, recheck):
    """File protocol only. Operational materialize always verifies evidence first."""
    pending,final=Path(pending),Path(final)
    _need(pending.parent==final.parent and not os.path.lexists(final),
          'partial_or_late_publication','fresh sibling publication target required')
    identity,members=_pending_snapshot(pending);attempted=False
    try:
        _clock()
        emit()  # A failure/blocked output cannot leave an accepted directory.
        _clock();recheck();_clock()
        _need(_pending_snapshot(pending)==(identity,members) and not os.path.lexists(final),
              'partial_or_late_publication','owned staged metadata changed before commit')
        attempted=True;pending.rename(final)
        _need(_native_identity(final)==identity,'partial_or_late_publication','publication object identity differs')
        _clock()
        return final
    except BaseException as error:
        status=_quarantine_owned(final,identity,members) if attempted else {'state':'not_attempted','owned_object_moved':False}
        failure=SourceQualificationError('partial_or_late_publication',
            'metadata publication failed; rollback status='+status['state'])
        failure.rollback_status=status
        raise failure from error


def copy_verified_file(source, target, *, expected_sha256, expected_bytes):
    source,target=Path(source),Path(target)
    _ordinary(source);_ordinary(target.parent,True)
    _need(not os.path.lexists(target) and source.resolve()!=target.parent/target.name,
          'host_local_path_or_chain_rewrite','fresh disjoint copy target required')
    _need(_hex(expected_sha256) and type(expected_bytes) is int and expected_bytes>=0,
          'input_error','copy SHA/byte grammar differs')
    partial=target.with_name(target.name+'.copying-'+uuid.uuid4().hex);digest=hashlib.sha256();total=0
    identity=None;attempted=False
    try:
        _need(_ordinary(source).st_size==expected_bytes and _file_sha(source)==expected_sha256,
              'host_local_path_or_chain_rewrite','original copied member differs')
        with source.open('rb') as stream,partial.open('xb') as output:
            value=os.fstat(output.fileno());identity=(value.st_dev,value.st_ino,stat.S_IFMT(value.st_mode))
            _need(value.st_ino!=0,'partial_or_late_publication','created copy native identity unavailable')
            while True:
                _clock();raw=stream.read(65536);_clock()
                if not raw:break
                total+=len(raw);_need(total<=expected_bytes,'partial_or_late_publication','copy source grew')
                _need(output.write(raw)==len(raw),'partial_or_late_publication','short evidence copy write');digest.update(raw)
            output.flush();os.fsync(output.fileno())
        _need(total==expected_bytes and digest.hexdigest()==expected_sha256 and _file_sha(source)==expected_sha256 and
              _file_sha(partial)==expected_sha256,'partial_or_late_publication','copy identity/byte/SHA changed')
        _clock();_need(not os.path.lexists(target),'host_local_path_or_chain_rewrite','target appeared during copy')
        attempted=True;partial.rename(target)
        _need(_native_identity(target)==identity,'partial_or_late_publication','copied target identity differs')
        _clock()
        return {'path':str(target),'sha256':expected_sha256,'bytes':expected_bytes}
    except BaseException as error:
        status=_quarantine_owned(target,identity) if attempted else {'state':'not_attempted','owned_object_moved':False}
        if isinstance(error,SourceQualificationError) and not attempted:
            error.rollback_status=status
            raise
        failure=SourceQualificationError('partial_or_late_publication',
            'evidence copy failed; no accepted copy claimed; rollback status='+status['state'])
        failure.rollback_status=status;failure.partial_copy_path=str(partial)
        raise failure from error


def _relative_ref(value, root, files=None, code='host_local_path_or_chain_rewrite'):
    _exact(value,('path','sha256'),code);name=_safe_name(value['path'],code)
    _need(_hex(value['sha256']),code,'relative reference SHA required')
    path=Path(root)/name
    _need(_file_sha(path)==value['sha256'],code,'relative reference bytes differ: '+name)
    if files is not None:
        _need(name in files and files[name]['sha256']==value['sha256'],code,'reference is outside complete evidence inventory')
    return path


def verify_worker_threads(receipt):
    """Only the original two integer thread fields; grants no qualification."""
    code='workers17_unready'
    _need(type(receipt) is dict,code,'worker receipt metadata object required')
    _integer(receipt.get('blas_threads'),1,1,code)
    _integer(receipt.get('julia_threads'),1,1,code)
    return _scope(complete_as_worker_thread_metadata=True,blas_threads=1,julia_threads=1)


def _source_literal_map(raw,constant):
    text=raw.decode('utf8');match=re.search(r'(?m)^const '+re.escape(constant)+r'\s*=\s*Dict\(\r?\n(.*?)^\)\r?$',text,re.S|re.M)
    _need(match is not None,'workers17_unready','qualified worker source map declaration absent')
    result={}
    for line in match.group(1).splitlines():
        if not line.strip():continue
        row=re.fullmatch(r'\s*"([A-Za-z0-9_.-]+)"\s*=>\s*"([0-9a-f]{64})",\s*',line)
        _need(row is not None and row.group(1) not in result,'workers17_unready','worker source map is not closed literal data')
        result[row.group(1)]=row.group(2)
    _need(bool(result),'workers17_unready','empty worker source identity');return result


def _origin_path(text,directory=False):
    """Compare historical absolute paths without reopening or rewriting them."""
    _need(type(text) is str and bool(text),'workers17_unready','original absolute path required')
    windows=bool(PureWindowsPath(text).drive)
    normalized=text.replace('\\','/') if windows else text
    _need(all(ord(c)>=32 and ord(c)!=127 for c in text),'workers17_unready','original path contains controls')
    if normalized.endswith('/') and normalized not in ('/',PureWindowsPath(text).drive+'/'):
        _need(directory and not normalized.endswith('//'),'workers17_unready','unexpected trailing original path separator')
        normalized=normalized[:-1]
    suffix=normalized[2:] if normalized.startswith('//') and windows else normalized.lstrip('/')
    _need(all(x not in ('','.','..') for x in suffix.split('/')),'workers17_unready','original path has empty/dot components')
    value=PureWindowsPath(normalized) if windows else PurePosixPath(normalized)
    _need(value.is_absolute() and '..' not in value.parts,'workers17_unready','original path grammar differs')
    return value


def verify_worker_argv(argv, *, runtime, science_checkout, evidence_root, job):
    """Fixed driver tail plus the closed, data-observed Base.julia_cmd prefix."""
    code='workers17_unready'
    _need(type(argv) is list and len(argv)==14 and all(type(x) is str for x in argv),code,'original closed fourteen-item worker argv required')
    science,root=_origin_path(science_checkout,True),_origin_path(evidence_root,True)
    _need(_origin_path(argv[0])==_origin_path(runtime['julia_path']) and
          argv[-7:-4]==['--startup-file=no','--pkgimages=no','--threads=1'] and
          argv[-4].startswith('--project=') and _origin_path(argv[-4][10:],True)==science and
          argv[-4][10:].rstrip('/\\')==science_checkout.rstrip('/\\') and
          _origin_path(argv[-3])==science/'test/interval_mean_forecaster/worker.jl' and
          argv[-2]==job and _origin_path(argv[-1])==root/job,code,'fixed worker file/project/output tail differs')
    _need(argv[1]=='-C' and 0<len(argv[2])<=2048 and re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.:+,;=\-]*',argv[2]) is not None and
          argv[3].startswith('-J') and len(argv[3])>2 and
          argv[4]=='--pkgimages=no' and re.fullmatch('-g[012]',argv[5]) is not None and argv[6]=='--startup-file=no',
          code,'unreviewed Base.julia_cmd prefix option')
    image=argv[3][2:]
    images=runtime.get('runtime_files',{})
    _need(type(images) is dict and 'sysimage' in images and image is not None and
          _origin_path(image)==_origin_path(images['sysimage']['path']),code,'original sysimage/runtime byte link differs')
    return _scope(complete_as_worker_argv=True)


def verify_workers17(value, evidence_root, *, expected_head, proof, checkout, files=None):
    code='workers17_unready'
    _need(value is not None,code,'both original seventeen-worker runtime trees are required')
    _exact(value,('schema','head','runtimes'),code)
    _need(value['schema']=='matched342_workers17_evidence_v1' and value['head']==expected_head and
          type(value['runtimes']) is list and len(value['runtimes'])==2 and
          {x.get('julia_version') for x in value['runtimes']}==set(RUNTIMES),code,'both exact runtime evidence sets required')
    expected_sources=_source_literal_map(_bytes(Path(checkout)/'test/interval_mean_forecaster/bindings.jl'),'SOURCE_PINS')
    for name,pin in expected_sources.items():
        _need(proof['source_sha256'].get('src/interval_mean_forecaster/'+name)==pin,code,'worker namespace source differs from whole proof')
    worker_sha=proof['source_sha256']['test/interval_mean_forecaster/worker.jl']
    bindings_sha=proof['source_sha256']['test/interval_mean_forecaster/bindings.jl']
    facts=[]
    terminal_fields={'schema','complete','head','tree','julia_version','science_checkout','started_ns','finished_ns','deadline_ns',
      'native_deadline_ns','started_utc','finished_utc','preflight','source_proof','source_after','runtime','argv','owner','launch',
      'raw','driver_evidence_root','driver_terminal','workers','admission','controls','error','source_qualified','software_qualified','scientific_qualified'}
    for entry in value['runtimes']:
        _exact(entry,('julia_version','terminal','owner','raw','members'),code);version=entry['julia_version']
        _need(type(entry['members']) is dict and 0<len(entry['members'])<=128,code,'bounded original worker member inventory required')
        prefix='workers17/'+version+'/'
        _need(type(entry['terminal']) is dict and entry['terminal'].get('path')==prefix+'TERMINAL.json',code,'only the fixed accepted terminal name is admissible')
        for name,row in entry['members'].items():
            _safe_name(name,code);_exact(row,('bytes','sha256'),code)
            _need(name.startswith(prefix) and type(row['bytes']) is int and 0<=row['bytes']<=2*GIB and _hex(row['sha256']),code,'runtime member scope differs')
            path=Path(evidence_root)/name
            _need(_ordinary(path).st_size==row['bytes'] and _file_sha(path)==row['sha256'],code,'original worker member bytes differ')
            if files is not None:_need(files.get(name)==row,code,'worker member outside acquisition inventory')
        terminal_path=_relative_ref(entry['terminal'],evidence_root,entry['members'],code)
        terminal=_json(_bytes(terminal_path));_exact(terminal,terminal_fields,code)
        _need(terminal['schema']=='interval_workers_qualification_terminal_v1' and terminal['complete'] is True and
              terminal['error'] is None and terminal['head']==expected_head and terminal['tree']==proof['tree'] and
              terminal['julia_version']==version and all(terminal[x] is False for x in ('source_qualified','software_qualified','scientific_qualified')),
              code,'original complete runtime terminal differs')
        for key in ('started_ns','finished_ns','deadline_ns','native_deadline_ns'):_integer(terminal[key],1,2**63-1,code)
        _need(terminal['started_ns']<terminal['finished_ns']<terminal['deadline_ns'] and
              terminal['started_ns']<terminal['native_deadline_ns']<=terminal['deadline_ns'],code,'original runtime clocks differ')
        for key in ('source_proof','source_after'):
            original=_json(_bytes(_relative_ref(terminal[key],evidence_root,entry['members'],code)))
            _proof(original)
            _need(original['head']==expected_head and original['tree']==proof['tree'] and
                  original['commit_object_ids']==proof['commit_object_ids'],code,'worker original before/after source proof differs')
            _need(all(original['source_sha256'][n]==proof['source_sha256'][n] for n in proof['source_sha256'] if n!='test/runtests.jl'),
                  code,'worker raw source changed outside known host checkout difference')
        runtime=terminal['runtime']
        _exact(runtime,('julia_version','julia_path','julia_binary_sha256','blas_threads','julia_threads','omp_threads','kernel','architecture','runtime_files'),code)
        _need(runtime['julia_version']==version and all(type(runtime[x]) is int and runtime[x]==1 for x in
              ('blas_threads','julia_threads','omp_threads')) and _hex(runtime['julia_binary_sha256']),code,'worker runtime/thread identity differs')
        _need(type(runtime['runtime_files']) is dict and set(runtime['runtime_files'])=={'julia','libjulia','sysimage'},code,'original executable/libjulia/sysimage identities absent')
        runtime_records=list(runtime['runtime_files'].values())
        for row in runtime_records:
            _exact(row,('path','bytes','sha256'),code)
            _need(type(row['path']) is str and type(row['bytes']) is int and row['bytes']>0 and _hex(row['sha256']),code,'historical runtime byte identity differs')
        _need(any(row['path']==runtime['julia_path'] and row['sha256']==runtime['julia_binary_sha256'] for row in runtime_records),code,'Julia executable identity link absent')
        admission=terminal['admission']
        _need(type(admission.get('available_physical_bytes')) is int and admission['available_physical_bytes']>=8*GIB and
              type(admission.get('free_disk_bytes')) is int and admission['free_disk_bytes']>=16*GIB,code,'real Julia8GiB/16GiB admission absent')
        _match_source_pins(terminal['controls'],proof)
        pre=_json(_bytes(_relative_ref(terminal['preflight'],evidence_root,entry['members'],code)))
        _need(pre.get('schema')=='interval_workers_preflight_v1' and pre.get('complete') is True and
              pre.get('mode')=='preflight_only' and pre.get('head')==expected_head and pre.get('tree')==proof['tree'] and
              pre.get('science_checkout')==terminal['science_checkout'] and pre.get('runtime')==version and
              pre.get('project_sha256')==proof['source_sha256']['Project.toml'] and
              pre.get('manifest_sha256')==proof['source_sha256']['Manifest.toml'] and pre.get('controls')==terminal['controls'] and
              type(pre.get('finished_ns')) is int and pre['finished_ns']<terminal['started_ns'],code,'original preflight/source/setup chain differs')
        _need(terminal['argv']==[runtime['julia_path'],'--startup-file=no','--pkgimages=no','--threads=1',
              '--project='+terminal['science_checkout'],str(_origin_path(terminal['science_checkout'])/'test/test_interval_mean_forecaster.jl')],
              code,'original direct driver argv differs')
        owner_path=_relative_ref(terminal['owner'],evidence_root,entry['members'],code)
        _need(entry['owner']==terminal['owner'] and entry['raw']==terminal['raw'],code,'aggregate runtime owner/raw differs')
        native=_json(_bytes(owner_path));launch=_json(_bytes(_relative_ref(terminal['launch'],evidence_root,entry['members'],code)))
        _positive_native(native,3600,expected_pid=launch.get('pid'))
        _need(native.get('leader_reaped') is True and native.get('ownership_boundary_closed') is True and
              native.get('all_descendants_os_reaped') is False,code,'original direct owner must prove boundary/reap without invented descendant waits')
        _need(type(launch.get('monotonic_ns')) is int and terminal['started_ns']<=launch['monotonic_ns']<terminal['native_deadline_ns'] and
              terminal['native_deadline_ns']-launch['monotonic_ns']<=3600*10**9 and
              type(native.get('wrapper_observed_wall_seconds')) in (int,float) and
              math.isfinite(native['wrapper_observed_wall_seconds']) and 0<=native['wrapper_observed_wall_seconds']<3600,
              code,'original on-launch/native clock window differs')
        raw_driver=_bytes(_relative_ref(terminal['raw'],evidence_root,entry['members'],code),RAW)
        warning='Warning: detected a stack overflow; program state may be corrupted, so further execution might be unreliable.'
        _need(not any(warning.encode(x) in raw_driver for x in ('utf8','utf-16-le','utf-16-be')),code,'unsafe Julia runtime warning')
        markers=re.findall(rb'(?m)^INTERVAL_FORECASTER_RETAINED_EVIDENCE=([^\r\n]+)\r?$',raw_driver)
        _need(len(markers)==1 and _origin_path(markers[0].decode('utf8'))==_origin_path(terminal['driver_evidence_root']),
              code,'original raw driver evidence-root marker differs')
        driver_path=_relative_ref(terminal['driver_terminal'],evidence_root,entry['members'],code)
        driver=tomllib.loads(_bytes(driver_path).decode('utf8'))
        _need(driver.get('complete') is True and driver.get('required_jobs')==list(JOBS17) and driver.get('worker_sha256')==worker_sha and
              type(driver.get('runs')) is list and len(driver['runs'])==17,code,'all ordered17 original driver results required')
        workers=terminal['workers']
        _need(type(workers) is list and len(workers)==17 and [x.get('job') for x in workers]==list(JOBS17),code,'ordered worker refs differ')
        for job,run,record in zip(JOBS17,driver['runs'],workers):
            _exact(record,('job','worker','raw'),code)
            receipt=tomllib.loads(_bytes(_relative_ref(record['worker'],evidence_root,entry['members'],code),2*1024**2).decode('utf8'))
            log_path=_relative_ref(record['raw'],evidence_root,entry['members'],code)
            _need(run.get('job')==job and type(run.get('pid')) is int and run['pid']>0 and
                  type(run.get('native_exit')) is int and run['native_exit']==0 and run.get('timed_out') is False and
                  run.get('receipt_valid') is True and run.get('log_sha256')==_file_sha(log_path),code,'worker native/receipt/raw join differs')
            try:
                _need(datetime.fromisoformat(run['started_utc'].replace('Z','+00:00'))<=datetime.fromisoformat(run['completed_utc'].replace('Z','+00:00')),
                      code,'worker UTC interval differs')
            except (ValueError,KeyError,TypeError) as error:raise SourceQualificationError(code,'original worker UTC missing') from error
            verify_worker_argv(run.get('argv'),runtime=runtime,science_checkout=terminal['science_checkout'],
                               evidence_root=terminal['driver_evidence_root'],job=job)
            environment=run.get('environment_controls');_exact(environment,('TMPDIR','TMP','TEMP'),code)
            _need(all(_origin_path(x)==_origin_path(terminal['driver_evidence_root'])/job/'tmp' for x in environment.values()),
                  code,'original per-worker temporary-root controls differ')
            worker_raw=_bytes(log_path,RAW)
            _need(re.findall(rb'(?m)^PACKAGE_BOUND_WORKER_COMPLETE=([^\r\n]+)\r?$',worker_raw)==[job.encode('ascii')] and
                  not any(warning.encode(x) in worker_raw for x in ('utf8','utf-16-le','utf-16-be')),
                  code,'original worker raw completion/warning differs')
            verify_worker_threads(receipt)
            _need(receipt.get('complete') is True and receipt.get('job')==job and receipt.get('julia_version')==version and
                  receipt.get('kernel')==runtime['kernel'] and receipt.get('architecture')==runtime['architecture'] and
                  receipt.get('worker_sha256')==worker_sha and receipt.get('bindings_sha256')==bindings_sha and
                  receipt.get('source_sha256')==expected_sources and
                  _origin_path(receipt.get('package_source'))==_origin_path(terminal['science_checkout'])/'src/SolarSINDy.jl',
                  code,'original loaded worker source/runtime/thread receipt differs')
        facts.append({'julia_version':version,'workers':17,'terminal_sha256':_file_sha(terminal_path),'native_pid':native['pid']})
    return _scope(complete_as_workers17=True,runtimes=facts)


def _expected_jobs():
    return {'Julia '+v+' / '+p for v in RUNTIMES for p in PLATFORMS} | {
      'Documentation build','Julia 1.12 deployment images','Canonical development harness','Full-scale cross-runtime agreement'} | {
      'Full-scale '+v+' / '+r+' / '+l for v in RUNTIMES for r in ('primary','independent') for l in ('ground-only','mixed')}


def _artifact_names(ci_run,native_run):
    tail=str(ci_run)+'-1'
    main={'canonical-harness-'+tail,'full-scale-agreement','matched342-owner-canonical-'+tail} | {
      'full-scale-'+v+'-'+r+'-'+l for v in RUNTIMES for r in ('primary','independent') for l in ('ground-only','mixed')} | {
      'matched342-owner-'+p+'-'+v+'-'+tail for v in RUNTIMES for p in PLATFORMS}
    native={prefix+'-'+p+'-'+str(native_run)+'-1' for prefix in ('posix-v3','darwin-anchor-v1') for p in NATIVE_PLATFORMS} | {
      'posix-v3-identity-'+str(native_run)+'-1'}
    return main,native


def verify_acquisition_context(owners,reviews,*,head,ci_run,native_run,proof):
    """Pure original-record joins only; structural fixtures remain unqualified."""
    code='coherent_header_raw_counterfeit';_proof(proof)
    _need(proof['head']==head and type(ci_run) is int and type(native_run) is int and ci_run>0 and native_run>0 and ci_run!=native_run,
          'wrong_original_identity','external source/run tuple differs')
    _need(type(owners) is list and len(owners)==2 and type(reviews) is list and len(reviews)==2,code,'two distinct original cohorts/reviews required')
    expected={'ci18':ci_run,'native3':native_run};by_cohort={};identities=[]
    for item in owners:
        _exact(item,('cohort','terminal','contract','launch','actor_sha256','raw_sha256','raw_bytes','terminal_sha256','contract_sha256','launch_sha256','proof','proof_sha256'),code)
        kind=item['cohort'];_need(kind in expected and kind not in by_cohort,code,'duplicate/missing original acquisition cohort')
        terminal,contract,launch,original=item['terminal'],item['contract'],item['launch'],_proof(item['proof'])
        _need(terminal.get('head')==head and terminal.get('run_id')==expected[kind] and original['head']==head and
              original['tree']==proof['tree'] and original['commit_object_ids']==proof['commit_object_ids'],
              'wrong_original_identity','original acquisition head/run/source differs')
        _need(set(n for n in original['source_sha256'] if original['source_sha256'][n]!=proof['source_sha256'][n])<={'test/runtests.jl'},
              'wrong_original_identity','acquisition source raw map differs outside proved checkout exception')
        before=terminal.get('source_before');after=terminal.get('source_after')
        _need(type(before) is dict and before==after==contract.get('source_before') and terminal.get('source_unchanged') is True,
              code,'original acquisition source/control guards disagree')
        guard='all'+str(original['source_files'])+'_raw_source_hashes_match'
        _exact(before,('actor_sha256',guard,'bindings_sha256','control_sha256','head','tree','source_files','source_proof_sha256'),code)
        _need(before['head']==head and before['tree']==original['tree'] and before['source_files']==original['source_files'] and
              before['source_proof_sha256']==item['proof_sha256'], 'wrong_original_identity','original source guard/proof lineage differs')
        _need(before[guard] is True and _hex(before['actor_sha256']) and _hex(before['bindings_sha256']) and
              type(before['control_sha256']) is dict and bool(before['control_sha256']) and all(_hex(x) for x in before['control_sha256'].values()),
              code,'original complete source/control byte guard absent')
        for key in ('actor_sha256','raw_sha256','terminal_sha256','contract_sha256','launch_sha256','proof_sha256'):_need(_hex(item[key]),code,'original acquisition pin grammar differs')
        _need(contract.get('actor_sha256')==item['actor_sha256'] and terminal.get('actor_sha256',item['actor_sha256'])==item['actor_sha256'] and
              terminal.get('complete') is True and terminal.get('raw_sha256')==item['raw_sha256'] and
              type(item['raw_bytes']) is int and terminal.get('raw_bytes')==item['raw_bytes'],code,'original actor/completion/raw join differs')
        keys=('started_ns','deadline_ns','whole_deadline_ns','original_deadline_ns')
        for key in keys:_integer(contract.get(key),1,2**63-1,code)
        for key in ('started_ns','finished_ns','worker_deadline_ns','deadline_ns','original_deadline_ns'):_integer(terminal.get(key),1,2**63-1,code)
        _integer(launch.get('monotonic_ns'),1,2**63-1,code);_integer(launch.get('pid'),1,2**63-1,code)
        _need(terminal['started_ns']==contract['started_ns'] and terminal['worker_deadline_ns']==contract['deadline_ns'] and
              terminal['deadline_ns']==contract['whole_deadline_ns'] and terminal['original_deadline_ns']==contract['original_deadline_ns'] and
              contract['started_ns']<=launch['monotonic_ns']<contract['deadline_ns']<=contract['whole_deadline_ns']<=contract['original_deadline_ns'] and
              launch['monotonic_ns']<=terminal['finished_ns']<contract['whole_deadline_ns'],code,'original acquisition clocks reset/reordered/late')
        native=terminal.get('native',{})
        try:
            _positive_native(native,(contract['deadline_ns']-contract['started_ns'])/1e9,expected_pid=launch['pid'])
        except SourceQualificationError as error:
            raise SourceQualificationError(code,'original acquisition native/launch/clock evidence differs') from error
        _need(native.get('leader_reaped') is True and native.get('ownership_boundary_closed') is True and native.get('all_descendants_os_reaped') is False,
              code,'original acquisition boundary/reap facts absent')
        wall=native.get('wrapper_observed_wall_seconds',native['wall_seconds'])
        _need(launch['monotonic_ns']+int(wall*1e9)<contract['deadline_ns'] and
              all(terminal.get(key) is False for key in ('source_qualified','software_qualified','scientific_qualified')) and
              all(type(terminal.get(key)) is int and terminal[key]==0 for key in ('API_POSTs','reruns','cancellations')),
              code,'original native clock/scope/mutation facts differ')
        identities.append((item['terminal_sha256'],item['contract_sha256'],item['launch_sha256'],launch['pid']))
        by_cohort[kind]=item
    _need(all(len({row[index] for row in identities})==2 for index in range(4)),code,'same original owner/launch/PID reused for both cohorts')
    seen=set();joined=set()
    for item in reviews:
        _exact(item,('sha256','data'),code);value=item['data']
        _need(_hex(item['sha256']) and item['sha256'] not in seen and type(value) is dict,code,'duplicate independent review bytes')
        seen.add(item['sha256']);matches=[kind for kind,owner in by_cohort.items() if value.get('terminal_sha256')==owner['terminal_sha256']]
        _need(len(matches)==1 and matches[0] not in joined,code,'review must bind one distinct original collector terminal')
        kind=matches[0];owner=by_cohort[kind];joined.add(kind)
        _need(value.get('head')==head and value.get('run_id')==expected[kind] and value.get('source_proof_sha256')==owner['proof_sha256'] and
              value.get('actor_sha256')==owner['actor_sha256'],'wrong_original_identity','independent original review tuple differs')
        _need(value.get('complete_as_data_only_artifact_reconciliation') is True,code,'original data-only acquisition review is incomplete')
    _need(joined==set(expected),code,'both original collector reviews required')
    return _scope(complete_as_acquisition_context=True,cohorts=sorted(joined),owners=2,reviews=2)


def _acquisition(ref,root,files,head,ci_run,native_run,proof):
    path=_relative_ref(ref,root,files,'coherent_header_raw_counterfeit');data=_json(_bytes(path));code='coherent_header_raw_counterfeit'
    _exact(data,('schema','head','ci_run','native_run','files','owners','reviews'),code)
    _need(data['schema']=='matched342_sourceq_acquisition_evidence_v1' and data['head']==head and data['ci_run']==ci_run and data['native_run']==native_run,
          'wrong_original_identity','original acquisition tuple differs')
    _need(data['files']=={name:row for name,row in files.items() if name!=ref['path']},code,'acquisition must bind all original evidence files')
    _need(type(data['owners']) is list and len(data['owners'])==2 and type(data['reviews']) is list and len(data['reviews'])==2,
          code,'exact distinct CI18/native3 owners/reviews required')
    _need(len({_canonical(x) for x in data['owners']})==2 and len({_canonical(x) for x in data['reviews']})==2,
          code,'duplicate original owner/review references')
    facts=[]
    for record in data['owners']:
        _exact(record,('terminal','contract','launch','raw','actor'),code)
        terminal_path=_relative_ref(record['terminal'],root,files);terminal=_json(_bytes(terminal_path))
        contract=_json(_bytes(_relative_ref(record['contract'],root,files)))
        launch=_json(_bytes(_relative_ref(record['launch'],root,files)))
        actor=_relative_ref(record['actor'],root,files);raw=_relative_ref(record['raw'],root,files)
        run=terminal.get('run_id');kind='ci18' if run==ci_run else 'native3' if run==native_run else None
        _need(kind is not None,'wrong_original_identity','unsupported original acquisition run')
        original_pin=terminal.get('source_before',{}).get('source_proof_sha256')
        candidates=[root/name for name,row in files.items() if row['sha256']==original_pin]
        _need(bool(candidates),code,'original acquisition seven-field proof bytes not retained')
        original=_json(_bytes(candidates[0]));_proof(original)
        api=_json(_bytes(root/kind/'ARTIFACTS.json'));expected={row['id']:(row['size_in_bytes'],row['digest'][7:]) for row in api['artifacts']}
        rows=terminal.get('archives');_need(type(rows) is list and len(rows)==(17 if kind=='ci18' else 7),code,'collector did not retain complete cohort archive coverage')
        _need(len({row.get('id') for row in rows})==len(rows) and {row['id']:(row['bytes'],row['sha256']) for row in rows}==expected,
              code,'original collector/API archive bytes differ')
        if kind=='ci18':
            jobs=_json(_bytes(root/'ci18/JOBS.json'))['jobs'];job_ids={x['id'] for x in jobs};logs=terminal.get('job_logs')
            _need(len(jobs)==18 and type(logs) is list and len(logs)==18 and len({x.get('job_id') for x in logs})==18,
                  code,'unsupported single-job/partial CI collector; full18 logs required')
            for row in logs:
                _exact(row,('job_id','raw_sha256','raw_bytes'),code);name='ci18/logs/'+str(row['job_id'])+'.log'
                _need(row['job_id'] in job_ids and name in files and row['raw_sha256']==files[name]['sha256'] and
                      type(row['raw_bytes']) is int and row['raw_bytes']==files[name]['bytes'],code,'original full18 job log coverage/raw differs')
        facts.append(dict(cohort=kind,terminal=terminal,contract=contract,launch=launch,actor_sha256=_file_sha(actor),raw_sha256=_file_sha(raw),
            raw_bytes=_ordinary(raw).st_size,terminal_sha256=record['terminal']['sha256'],contract_sha256=record['contract']['sha256'],
            launch_sha256=record['launch']['sha256'],proof=original,proof_sha256=original_pin))
    reviews=[dict(sha256=value['sha256'],data=_json(_bytes(_relative_ref(value,root,files)))) for value in data['reviews']]
    checked=verify_acquisition_context(facts,reviews,head=head,ci_run=ci_run,native_run=native_run,proof=proof)
    return dict(checked,sha256=_file_sha(path))

def verify_evidence(manifest, evidence_root, expected_head, ci_run, native_run, checkout):
    global _DEADLINE
    if _DEADLINE is None:_DEADLINE=time.monotonic_ns()+1800*10**9
    _clock();_need(not sys.flags.optimize,'input_error','assert-based original verifiers require unoptimized Python')
    _exact(manifest,('schema','head','ci18','native3','linux_proof','host_proof','acquisition','workers17','files'))
    _need(manifest['schema']=='matched342_sourceq_input_v1' and manifest['head']==expected_head and _hex(expected_head,40),
          'wrong_original_identity','external expected head differs from input')
    for key,ident in (('ci18',ci_run),('native3',native_run)):
        _exact(manifest[key],('run_id','workflow_id'));_integer(manifest[key]['workflow_id'],1,2**63-1)
        _need(type(ident) is int and ident>0 and manifest[key]['run_id']==ident,'wrong_original_identity','expected original run differs')
    root=Path(evidence_root).resolve(strict=True);checkout=Path(checkout).resolve(strict=True)
    _ordinary(root,True);_ordinary(checkout,True)
    _need(not root.is_relative_to(checkout) and not checkout.is_relative_to(root),'host_local_path_or_chain_rewrite','evidence overlaps science checkout')
    def document(name):return _json(_bytes(root/name))
    # Original run/job semantics are checked before any publication or absent
    # supplemental worker evidence can hide a genuinely unready CI cohort.
    run=document('ci18/RUN.json');nr=document('native3/RUN.json')
    verify_run_identity(run,expected_head=expected_head,run_id=ci_run,workflow_id=manifest['ci18']['workflow_id'],event='workflow_dispatch',branch=BRANCH)
    verify_run_identity(nr,expected_head=expected_head,run_id=native_run,workflow_id=manifest['native3']['workflow_id'],event='push',branch=BRANCH)
    _need(run.get('status')=='completed' and nr.get('status')=='completed','incomplete_inventory','original cohort is still live')
    _need(run.get('conclusion')==nr.get('conclusion')=='success','unsuccessful_hosted_command','original cohort did not succeed')
    jobs_doc=document('ci18/JOBS.json');native_jobs_doc=document('native3/JOBS.json')
    _need(type(jobs_doc.get('total_count')) is int and jobs_doc['total_count']==len(jobs_doc.get('jobs',[])),
          'incomplete_inventory','full18 API pagination/count differs')
    jobs=verify_job_inventory(jobs_doc['jobs'],_expected_jobs(),ci_run)
    native_jobs=verify_job_inventory(native_jobs_doc.get('jobs'),{'V3 engineering / '+p for p in NATIVE_PLATFORMS},native_run)
    files=manifest['files'];_need(type(files) is dict and 0<len(files)<=100000,'input_error','bounded complete file inventory required')
    folded=set();actual=set()
    for path in root.rglob('*'):
        _ordinary(path,path.is_dir())
        if path.is_file():actual.add(path.relative_to(root).as_posix())
    _need(actual==set(files),'host_local_path_or_chain_rewrite','unexpected/missing input evidence file')
    for name,row in files.items():
        _safe_name(name);_exact(row,('bytes','sha256'));_integer(row['bytes'],0,2*GIB)
        _need(name.casefold() not in folded and _hex(row['sha256']),'host_local_path_or_chain_rewrite','evidence alias/hash grammar differs');folded.add(name.casefold())
        path=root/name
        _need(_ordinary(path).st_size==row['bytes'] and _file_sha(path)==row['sha256'],
              'coherent_header_raw_counterfeit','original evidence member differs: '+name)
    for name in ('ci18/RUN.json','ci18/JOBS.json','ci18/ARTIFACTS.json','native3/RUN.json','native3/JOBS.json','native3/ARTIFACTS.json'):
        _need(name in files,'incomplete_inventory','original API input absent')
    linux_path=_relative_ref(manifest['linux_proof'],root,files);host_path=_relative_ref(manifest['host_proof'],root,files)
    linux=_json(_bytes(linux_path));host=_json(_bytes(host_path));join=verify_proof_join(linux,host,checkout)
    _need(join['host_checkout_verified'] is True and host['head']==expected_head,'cross_host_source_proof_gap','actual host proof required')
    acquisition=_acquisition(manifest['acquisition'],root,files,expected_head,ci_run,native_run,host)
    component_file=checkout/'tools/actual_b_chain/BMatchedReference342.jl'
    component_names=parse_component_members(_bytes(component_file));component_sources={name:host['source_sha256'][name] for name in component_names}
    for name,pin in component_sources.items():_need(_file_sha(checkout/name)==pin,'component_closure_mismatch','component physical bytes differ')
    docs_source=_bytes(checkout/'docs/make.jl').decode('utf8')
    _need(all(text in docs_source for text in ('checkdocs = :exports','doctest = true','warnonly = false')),
          'missing_second_strict_docs','qualified strict documentation configuration changed')
    main_names,native_names=_artifact_names(ci_run,native_run);artifacts={};archive_checks={};original_api={}
    for namespace,ident,expected in (('ci18',ci_run,main_names),('native3',native_run,native_names)):
        api=document(namespace+'/ARTIFACTS.json');rows=api.get('artifacts')
        _need(type(rows) is list and type(api.get('total_count')) is int and api['total_count']==len(rows),
              'incomplete_inventory','complete archive API page required')
        names={row.get('name') for row in rows};missing=expected-names
        owner_missing={name for name in missing if name.startswith(('matched342-owner-','posix-v3-','darwin-anchor-v1-'))}
        _need(not owner_missing,'missing_current_owner_evidence','mandatory current engineering archive absent')
        _need(names==expected and len(rows)==len(expected) and len({row.get('id') for row in rows})==len(rows),
              'incomplete_inventory','exact17/main plus7/native archive inventory differs')
        for row in rows:
            _integer(row.get('id'),1,2**63-1,'archive_identity_or_namespace_failure')
            name=namespace+'/archives/'+str(row['id'])+'.zip';_need(name in files,'archive_identity_or_namespace_failure','original archive bytes missing')
            path=root/name;checked=verify_archive(path,api_record=row,run_id=ident,expected_head=expected_head)
            artifacts[row['name']]=path;archive_checks[row['name']]=checked;original_api[row['name']]=row
    canonical=_load(checkout/'tools/verify_canonical_harness_artifact.py','sourceq_science_canonical',host['source_sha256']['tools/verify_canonical_harness_artifact.py'])
    contract=_load(checkout/'tools/run_full_scale_generated.py','sourceq_science_full_scale',host['source_sha256']['tools/run_full_scale_generated.py'])
    resource=_helper();hname='canonical-harness-'+str(ci_run)+'-1'
    try:h=canonical.verify(artifacts[hname],linux)
    except (AssertionError,ValueError,KeyError,OSError) as error:raise SourceQualificationError('nonpositive_owned_stage','original canonical source/native evidence failed') from error
    _need(h['source_proof_sha256']==manifest['linux_proof']['sha256'],'cross_host_source_proof_gap','original physical canonical Linux proof bytes differ')
    hraw=_zip_bytes(artifacts[hname],'harness.log',16*1024**2);harness=verify_harness_log(hraw,linux)
    canonical_runner=_bytes(checkout/'tools/run_canonical_harness.py').decode('utf8')
    _need(len(re.findall(r'(?m)^\s*logpath=out/\x27harness.log\x27,timeout=21600\)',canonical_runner))==1,
          'nonpositive_owned_stage','source-bound original canonical native allowance differs')
    hd=_archive_data(artifacts[hname]);verify_canonical_native(hd['TERMINAL.json']['native'])
    workloads={}
    for version in RUNTIMES:
        for role in ('primary','independent'):
            for layout in ('ground-only','mixed'):
                name='full-scale-'+version+'-'+role+'-'+layout
                try:workloads[name]=resource.verify(artifacts[name],linux,mode='workload',runtime=version,role=role,layout=layout,
                                                   contract=contract,canonical=canonical,deadline_ns=_DEADLINE)
                except Exception as error:
                    if isinstance(error,SourceQualificationError):raise
                    raise SourceQualificationError(getattr(error,'code','resource_workload_gap'),'original full workload verification failed: '+name) from error
    verify_workload_inventory(workloads,expected_head)
    try:agreement=resource.verify(artifacts['full-scale-agreement'],linux,mode='agreement',runtime='1.13.1',workloads=workloads,
                                  contract=contract,canonical=canonical,deadline_ns=_DEADLINE)
    except Exception as error:raise SourceQualificationError(getattr(error,'code','agreement48_gap'),'original48-comparison evidence failed') from error
    job_facts={};raw_paths={}
    for name,job in jobs.items():
        raw_path=root/'ci18/logs'/str(job['id']);raw_path=raw_path.with_suffix('.log')
        _need(raw_path.relative_to(root).as_posix() in files,'incomplete_inventory','original18 job log missing')
        raw=_bytes(raw_path,RAW)
        if name.startswith('Julia ') and name.endswith('-latest'):
            required=('Verify matched342 owner contract engineering controls','Verify matched342 owner lifecycle engineering controls',
              'Run julia-actions/julia-runtest@v1','Run the research experiments')
            plain=re.sub(rb'\x1b\[[0-9;]*m',b'',raw)
            _need(b'SolarSINDy tests passed' in plain and
                  b'Generated B1 native-entry handoff: 2 laws, 36 gain fits, all36 independently read back; no actual source or selection authority.' in plain,
                  'filtered_or_incomplete_execution','raw package/final experiments marker absent')
            if name.endswith('macos-latest'):
                _need(re.search(rb'Successfully set up CPython \(3\.14\.[0-9]+\)',plain),
                      'filtered_or_incomplete_execution','actual macOS Python3.14 setup missing')
        elif name=='Documentation build':required=('Build the manual (Julia 1.12.6)','Build the manual (Julia 1.13.1)')
        elif name=='Julia 1.12 deployment images':required=('Build both images','Load the monitor and frozen artifacts as the runtime user','Start the dashboard as the runtime user')
        elif name=='Canonical development harness':required=('Verify matched342 owner contract engineering controls','Verify matched342 owner lifecycle engineering controls',
          'Run unchanged canonical audit with full tests and experiments')
        elif name=='Full-scale cross-runtime agreement':required=('Verify all eight source proofs and all 48 native table comparisons',)
        else:required=('Verify full-size contracts and corruption rejection','Execute whole-schedule oracles and full input resource checks')
        job_facts[name]=verify_hosted_job(run,job,raw,required);raw_paths[name]=raw_path
    owner_checks={}
    for name,path in artifacts.items():
        if name.startswith('matched342-owner-'):
            platform='canonical' if '-canonical-' in name else next(p for p in PLATFORMS if '-'+p+'-' in name)
            owner_checks[name]=verify_owner_artifact(path,platform=platform,proof=host)
        elif name.startswith(('posix-v3-','darwin-anchor-v1-')) and not name.startswith('posix-v3-identity-'):
            platform=next(p for p in NATIVE_PLATFORMS if '-'+p+'-' in name)
            owner_checks[name]=verify_owner_artifact(path,platform=platform,proof=host)
    for platform in NATIVE_PLATFORMS:
        job=native_jobs['V3 engineering / '+platform]
        _need(job.get('status')=='completed' and job.get('conclusion')=='success','unsuccessful_hosted_command','native3 hosted job failed')
        for step in ('Run bounded independent current-source controls','Run independent launch-anchor and cancellation controls'):
            rows=[x for x in job.get('steps',[]) if x.get('name')==step]
            _need(len(rows)==1 and rows[0].get('status')=='completed' and rows[0].get('conclusion')=='success',
                  'missing_current_owner_evidence','native3 current source step missing/skipped')
    worker_path=_relative_ref(manifest['workers17'],root,files,'workers17_unready') if manifest['workers17'] is not None else None
    workers=verify_workers17(_json(_bytes(worker_path)) if worker_path else None,root,expected_head=expected_head,proof=host,checkout=checkout,files=files)
    _clock()
    return _scope(verified=True,head=expected_head,ci_run=ci_run,native_run=native_run,evidence_root=root,checkout=checkout,
                  manifest=manifest,files=files,run=run,jobs=jobs,job_facts=job_facts,raw_paths=raw_paths,
                  linux_proof=linux,host_proof=host,proof_join=join,component_sources=component_sources,
                  harness=harness,harness_raw=hraw,canonical=h,workloads=workloads,agreement=agreement,
                  archives=archive_checks,owner_checks=owner_checks,workers17=workers,acquisition=acquisition)


def materialize(manifest_path, *, manifest_sha256, evidence_root, expected_head, ci_run, native_run, checkout, output_directory, emit=None):
    global _DEADLINE
    _DEADLINE=time.monotonic_ns()+1800*10**9
    _need(not sys.flags.optimize and _hex(manifest_sha256),'input_error','unoptimized pinned file entry required')
    manifest_path=Path(manifest_path).resolve(strict=True)
    _need(_file_sha(manifest_path,META)==manifest_sha256,'input_error','Root-pinned manifest bytes differ')
    manifest=_json(_bytes(manifest_path));output=Path(output_directory)
    _need(output.is_absolute() and output.parent==output.parent.resolve(strict=True) and not os.path.lexists(output),
          'host_local_path_or_chain_rewrite','fresh canonical materialization root required')
    _ordinary(output.parent,True)
    source=Path(evidence_root).resolve(strict=True);science=Path(checkout).resolve(strict=True)
    _need(all(not output.is_relative_to(path) and not path.is_relative_to(output) for path in (source,science)),
          'host_local_path_or_chain_rewrite','fresh output overlaps source/evidence')
    controls={str(Path(__file__).resolve()):_file_sha(__file__),
              str(Path(__file__).with_name('sourceq_full_scale_verifier.py').resolve()):_file_sha(Path(__file__).with_name('sourceq_full_scale_verifier.py'))}
    # Resource admission is for bounded offline engineering work, never permission
    # to run Julia or the resource experiment on this host.
    memory_path=science/'tools/run_canonical_harness.py'
    host_ref=manifest['host_proof'];host_input=_json(_bytes(_relative_ref(host_ref,source)))
    _proof(host_input)
    memory=_load(memory_path,'sourceq_available_memory',host_input['source_sha256']['tools/run_canonical_harness.py'])
    available,method=memory.available_memory();free=shutil.disk_usage(output.parent).free
    _need(available>=512*1024**2 and free>=8*GIB,'input_error','512MiB available physical RAM/8GiB disk required')
    checked=verify_evidence(manifest,source,expected_head,ci_run,native_run,science)
    _need(checked['verified'] is True and checked['proof_join']['host_checkout_verified'] is True,
          'partial_or_late_publication','complete verified original evidence required')
    _need(sum(row['bytes'] for row in checked['files'].values())+8*GIB<=shutil.disk_usage(output.parent).free,
          'partial_or_late_publication','copied evidence bytes plus original 8GiB reserve do not fit')
    output.mkdir();evidence=output/'evidence';evidence.mkdir();pending=output/'.qualifications.pending';pending.mkdir()
    try:
        copied={}
        for name,row in checked['files'].items():
            target=evidence/name;target.parent.mkdir(parents=True,exist_ok=True)
            copied[name]=copy_verified_file(source/name,target,expected_sha256=row['sha256'],expected_bytes=row['bytes'])
        # The one needed internal canonical member is copied as ordinary bytes;
        # no archived links, sockets or FIFOs are reconstructed.
        hpath=evidence/'canonical-harness.log'
        with hpath.open('xb') as stream:
            _need(stream.write(checked['harness_raw'])==len(checked['harness_raw']),'partial_or_late_publication','short harness raw write')
            stream.flush();os.fsync(stream.fileno())
        _need(_file_sha(hpath)==checked['harness']['raw_sha256'],'partial_or_late_publication','ordinary H copy changed')
        receipt_root=evidence/'hosted-jobs';receipt_root.mkdir()
        receipts={};raw_refs={};original_run={'path':str(evidence/'ci18/RUN.json'),'sha256':checked['files']['ci18/RUN.json']['sha256']}
        original_jobs={'path':str(evidence/'ci18/JOBS.json'),'sha256':checked['files']['ci18/JOBS.json']['sha256']}
        for name,job in checked['jobs'].items():
            facts=checked['job_facts'][name];raw_name='ci18/logs/'+str(job['id'])+'.log'
            raw={'path':str(evidence/raw_name),'sha256':checked['files'][raw_name]['sha256']};raw_refs[name]=raw
            receipt={'schema':'matched342_hosted_job_evidence_v1','head':expected_head,'run_id':ci_run,'run_attempt':1,
              'job_id':job['id'],'name':name,'original_run':original_run,'original_jobs':original_jobs,'raw':raw,
              'required_steps':facts['required_steps'],'exit_projection':facts['exit_projection']}
            receipts[name]=_put(receipt_root/(str(job['id'])+'.json'),receipt)
        counts=checked['harness']['counts'];legacy_names={'Julia '+v+' / '+p for v in RUNTIMES for p in PLATFORMS} | {
          'Documentation build','Julia 1.12 deployment images'}
        legacy=[];extension_jobs=[]
        for name in sorted(checked['jobs']):
            job=checked['jobs'][name];raw=raw_refs[name]
            extension_jobs.append({'name':name,'job_id':job['id'],'head':expected_head,'conclusion':'success',
              'raw_path':raw['path'],'raw_sha256':raw['sha256'],'native_exit':0,'timed_out':False})
            if name in legacy_names:
                legacy.append({'name':name,'job_id':job['id'],'log':raw,'receipt':receipts[name],
                               'original_log':raw,'original_receipt':original_jobs})
        host_proof={'path':str(evidence/manifest['host_proof']['path']),'sha256':manifest['host_proof']['sha256']}
        extension={'complete':True,'head':expected_head,'jobs':extension_jobs,'harness_counts':counts,
          'harness_raw_path':str(hpath),'harness_raw_sha256':checked['harness']['raw_sha256'],
          'harness_named_checks_sha256':checked['harness']['named_checks_sha256'],'both_actual_strict_docs_passed':True,
          'unfiltered_Pkg_and_complete_examples_passed':True,'eight_full_resources_verified':True,'agreement48_verified':True}
        rootq={'complete':True,'head':expected_head,'source_proof':host_proof,'software_qualification_only':True,
          'unfiltered_package_and_full_experiments_executed':True,'strict_docs_both_complete':True,
          'all_eight_latest_ci_jobs_passed':True,'all_eight_raw_logs_retained':True,'harness_counts':counts,
          'ci_jobs':legacy,'candidate_pipeline_full18':extension}
        root_ref=_put(pending/'ROOT_QUALIFICATION.json',rootq)
        components={}
        for mode,filename in (('synthetic_only','COMPONENT_SYNTHETIC.json'),('software_qualified','COMPONENT_SOFTWARE.json')):
            native=[];other=[]
            for name in sorted(legacy_names):
                row={'name':name,'conclusion':'success','receipt_sha256':receipts[name]['sha256'],'log_sha256':raw_refs[name]['sha256']}
                (native if name.startswith('Julia ') and name.endswith('-latest') else other).append(row)
            document={'schema':'b_reference_component_software_qualification_v1','complete':True,'evidence_mode':mode,'head':expected_head,
              'source_sha256':checked['component_sources'],'harness':{'counts':counts,'all17_independent_workers_complete':True,
              'unfiltered_pkg_test_complete':True,'experiments_entry_complete':True},'native_jobs':native,'docs_and_containers':other,
              'root_qualification_sha256':root_ref['sha256'],'source_proof_sha256':host_proof['sha256'],
              'canonical_harness_sha256':checked['harness']['raw_sha256'],'software_qualification_only':True,
              'actual_panel_fit_or_outcome_qualified':False,'serving_promotion':False}
            verify_component_document(document,mode=mode,head=expected_head,component_sources=checked['component_sources'],
                root_sha=root_ref['sha256'],proof_sha=host_proof['sha256'],harness_sha=checked['harness']['raw_sha256'],harness_counts=counts)
            components[mode]=_put(pending/filename,document)
        _need(components['synthetic_only']['sha256']!=components['software_qualified']['sha256'],
              'mode_or_authority_escalation','two distinct mode-specific component identities required')
        def recheck():
            for name,row in checked['files'].items():
                _need(_file_sha(source/name)==row['sha256'] and _file_sha(evidence/name)==row['sha256'],
                      'partial_or_late_publication','original/copied evidence changed before qualification publication')
            _need(_file_sha(manifest_path)==manifest_sha256 and all(_file_sha(Path(path))==pin for path,pin in controls.items()),
                  'partial_or_late_publication','manifest/materializer controls changed')
            verify_proof_join(checked['linux_proof'],checked['host_proof'],science)
        final=output/'qualifications'
        result=_scope(complete_as_materialization=True,head=expected_head,ci_run=ci_run,native_run=native_run,
          original_manifest_sha256=manifest_sha256,linux_proof_sha256=manifest['linux_proof']['sha256'],host_proof_sha256=host_proof['sha256'],
          proof_join=checked['proof_join'],workers17=checked['workers17'],acquisition=checked['acquisition'],
          ROOT_QUALIFICATION={'path':str(final/'ROOT_QUALIFICATION.json'),'sha256':root_ref['sha256']},
          components={mode:{'path':str(final/Path(ref['path']).name),'sha256':ref['sha256']} for mode,ref in components.items()},
          copied_original_files=len(copied),API_POSTs=0,Julia_launched=False,actual_permit_issued=False,admission={'available_physical_bytes':available,'memory_method':method,'free_disk_bytes':free})
        _put(pending/'MATERIALIZATION.json',result)
        finalize_materialization(pending,final,emit=(emit if emit is not None else lambda:None),recheck=recheck)
        return result
    except BaseException as error:
        status=getattr(error,'rollback_status',{'state':'not_attempted','owned_object_moved':False})
        try:
            _put(output/'FAILURE.json',_scope(complete=False,error_type=type(error).__name__,error=str(error),
                qualification_directory_installed=None if os.path.lexists(output/'qualifications') else False,
                rollback_status=status,original_inputs_preserved=True))
        except BaseException as diagnostic_error:
            try:
                error.add_note('failure diagnostic was not accepted: '+repr(diagnostic_error))
            except BaseException:
                pass
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--expected-head',required=True)
    parser.add_argument('--ci-run',type=int,required=True)
    parser.add_argument('--native-run',type=int,required=True)
    parser.add_argument('--science-checkout',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        def emit_pending():
            print(json.dumps(_scope(complete=False,state='verified_pending_publication',head=args.expected_head,
                output_directory=str(args.output_directory),qualification_issued=False),sort_keys=True),flush=True)
        materialize(args.manifest,manifest_sha256=args.manifest_sha256,evidence_root=args.evidence_root,
          expected_head=args.expected_head,ci_run=args.ci_run,native_run=args.native_run,checkout=args.science_checkout,
          output_directory=args.output_directory,emit=emit_pending)
        return 0
    except SourceQualificationError as error:
        print(json.dumps({'complete':False,'rejection_code':error.code,'detail':error.detail,'qualification_issued':False},sort_keys=True),file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
