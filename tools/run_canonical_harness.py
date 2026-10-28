"""Bounded exact-source canonical research audit with retained raw evidence.

The mirrored shell audit is byte-pinned to the canonical workflow source.
Evidence lives outside the checkout; this command never grants actual-data use.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ctypes,hashlib,importlib.util,json,os,re,shlex,shutil,stat,subprocess,sys,time,traceback

AUDIT_SHA='c7cb2ad1d05a77cf0462c9af6f82f17394d2fc37476fe8c68e57bd4a625c2b28'
OWNER_SHA='0b0f2327841230e5ac87d03bd67d1cbaf10717e0a52ce0134d71169a36146b89'
GIB=1024**3
def pin(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write_new(path,value):
    with path.open('x',encoding='utf8',newline='\n') as stream:json.dump(value,stream,indent=2);stream.write('\n')
def available_memory():
    if os.name=='nt':
        class Memory(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ('total','available','pt','pa','vt','va','ex')]
        value=Memory();value.length=ctypes.sizeof(value)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):raise ctypes.WinError()
        return value.available,'Windows GlobalMemoryStatusEx available physical bytes'
    if sys.platform.startswith('linux'):
        text=Path('/proc/meminfo').read_text();matches=re.findall(r'^MemAvailable:\s+(\d+) kB$',text,re.M)
        if len(matches)!=1:raise RuntimeError('Linux MemAvailable evidence missing')
        return int(matches[0])*1024,'Linux /proc/meminfo MemAvailable'
    if sys.platform=='darwin':
        text=subprocess.check_output(['/usr/bin/vm_stat'],timeout=30).decode()
        match=re.search(r'page size of (\d+) bytes',text)
        rows=dict(re.findall(r'^(Pages free|Pages speculative):\s+(\d+)\.$',text,re.M))
        if match is None or len(rows)!=2:raise RuntimeError('macOS free/speculative page evidence missing')
        return sum(map(int,rows.values()))*int(match[1]),'Conservative macOS free plus speculative pages;inactive pages excluded'
    raise RuntimeError('Physical admission evidence unavailable on this platform')
def bash_path():
    specified=os.environ.get('SOLARSINDY_TEST_BASH')
    if specified:return Path(specified).resolve(strict=True)
    if os.name=='nt':
        for name in ('ProgramFiles','ProgramFiles(x86)'):
            root=os.environ.get(name)
            if root and (Path(root)/'Git/bin/bash.exe').is_file():return (Path(root)/'Git/bin/bash.exe').resolve()
        raise RuntimeError('Native Git Bash required;set SOLARSINDY_TEST_BASH explicitly')
    path=shutil.which('bash')
    if path is None:raise RuntimeError('Bash required for the canonical audit')
    return Path(path).resolve(strict=True)
def shell_path(path):
    text=Path(path).resolve().as_posix()
    if os.name=='nt':
        if len(text)<3 or text[1:3]!=':/':raise ValueError('Git Bash needs an absolute drive path')
        text='/'+text[0].lower()+text[2:]
    if '\n' in text or '\r' in text:raise ValueError('Unsupported shell path characters')
    return text
def link_path(path):
    # Python3.11 has no Path.is_junction();the native reparse attribute
    # covers Windows junctions as well as symlinks without following them.
    info=path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info,'st_file_attributes',0)&0x400)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--evidence-dir',type=Path,required=True)
    parser.add_argument('--julia',type=Path)
    args=parser.parse_args();root=args.project_root.resolve(strict=True);out=args.evidence_dir.resolve()
    if out==root or out.is_relative_to(root) or out.exists():raise ValueError('Evidence directory must be fresh and outside the checkout')
    audit=root/'tools/canonical_dev_harness.sh';owner=root/'research/dbdt_completed_source/native_stage.py'
    if pin(audit)!=AUDIT_SHA or pin(owner)!=OWNER_SHA:raise RuntimeError('Canonical audit/owned runner pin differs')
    julia=(args.julia or Path(shutil.which('julia') or '')).resolve(strict=True)
    if not julia.is_file():raise RuntimeError('Native Julia executable required')
    bash=bash_path();out.mkdir(parents=True);raw=out/'raw';raw.mkdir();operational=out/'operational';operational.mkdir()
    def git(*argv,input=None):return subprocess.check_output(['git','-C',str(root),*argv],input=input,stderr=subprocess.PIPE,timeout=120)
    if git('status','--porcelain'):raise RuntimeError('Canonical audit requires a clean committed checkout')
    head=git('rev-parse','HEAD').decode().strip();names=git('ls-files','-z').decode().split('\0')[:-1]
    source={name:pin(root/name) for name in names};objects={}
    for name in names:
        body=(root/name).read_bytes();oid=git('rev-parse','HEAD:'+name).decode().strip();objects[name]=oid
        if git('hash-object','--path='+name,'--stdin',input=body).decode().strip()!=oid:raise RuntimeError('Source clean filter differs:'+name)
        if hashlib.sha256(git('cat-file','--filters','HEAD:'+name)).hexdigest()!=source[name]:raise RuntimeError('Source checkout filter differs:'+name)
        if not stat.S_ISREG((root/name).lstat().st_mode) or (root/name).is_symlink():raise RuntimeError('Source member is not ordinary:'+name)
    proof=dict(complete=True,head=head,tree=git('rev-parse','HEAD^{tree}').decode().strip(),source_files=len(source),source_sha256=source,
        commit_object_ids=objects,all_clean_and_checkout_filters_match=True)
    write_new(out/'SOURCE_PROOF.json',proof)
    wrapper=out/'run-harness.sh'
    wrapper.write_text('#!/bin/bash\nset -euo pipefail\n'+
        "julia() { "+shlex.quote(shell_path(julia))+" --pkgimages=no \"$@\"; }\nexport -f julia\n"+
        'exec '+shlex.quote(shell_path(bash))+' '+shlex.quote(shell_path(audit))+' '+shlex.quote(shell_path(root))+'\n',encoding='utf8',newline='\n')
    evidence_helper=root/'tools/harness_evidence.py'
    controls={str(path):pin(path) for path in (Path(__file__).resolve(),audit,owner,evidence_helper,julia,bash,wrapper,out/'SOURCE_PROOF.json')}
    environment=dict(os.environ,PATH=str(julia.parent)+os.pathsep+str(bash.parent)+os.pathsep+os.environ.get('PATH',''),
        TMP=str(raw),TEMP=str(raw),TMPDIR=str(raw),RUNNER_TEMP=str(raw),JULIA_NUM_THREADS='1',JULIA_PKG_PRECOMPILE_AUTO='0',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        SOLARSINDY_TEST_BASH=str(bash),SOLARSINDY_NO_OPEN='1',SOLARSINDY_OPERATIONAL_OUTPUT_DIR=str(operational),PYTHONUTF8='1')
    # Every generated native/support proof belongs to this invocation's retained
    # raw tree;never inherit an earlier diagnostic or synthetic-chain directory.
    for key in ('SOLARSINDY_SYNTHETIC_REFIT_TEST_ROOT','SOLARSINDY_CLI_TEST_LOG_DIR','SOLARSINDY_CALIBRATION_TEST_OUTPUT'):
        environment.pop(key,None)
    if os.name=='nt':environment['PATH']=str(bash.parent.parent/'usr/bin')+os.pathsep+environment['PATH']
    record=dict(complete=False,started_utc=now(),head=head,source_sha256=source,controls_sha256=controls,
        source_proof_sha256=pin(out/'SOURCE_PROOF.json'),canonical_audit_sha256=AUDIT_SHA,owner_sha256=OWNER_SHA,
        runtime_binary_sha256=pin(julia),actual_source_access=False,actual_fits=0,scientific_performance_claim=False)
    write_new(out/'PREEXECUTION.json',record)
    def differences():
        changed=[];modified=[];errors=[]
        for mapping,base,selected in ((source,root,changed),(controls,Path(),modified)):
            for name,sha in mapping.items():
                path=base/name
                try:
                    if not stat.S_ISREG(path.lstat().st_mode) or path.is_symlink() or pin(path)!=sha:selected.append(name)
                except OSError as error:selected.append(name);errors.append(dict(path=str(path),error=repr(error)))
        try:status=git('status','--porcelain').decode();current=git('rev-parse','HEAD').decode().strip()
        except (OSError,subprocess.CalledProcessError) as error:status='unreadable';current='unreadable';errors.append(dict(kind='git',error=repr(error)))
        return dict(changed_sources=changed,changed_controls=modified,guard_read_errors=errors,git_status=status,head_changed=current!=head)
    try:
        deadline=time.monotonic()+43200
        while True:
            memory,method=available_memory();assert not any(differences().values())
            if memory>=8*GIB:break
            assert time.monotonic()<deadline,'8GiB physical admission unavailable after12h'
            (out/'WAITING_ADMISSION.json').write_text(json.dumps(dict(utc=now(),available_physical_bytes=memory,method=method))+'\n',encoding='utf8');time.sleep(15)
        disk=shutil.disk_usage(out).free;assert disk>=8*GIB,'8GiB disk admission unavailable'
        record.update(available_physical_bytes_at_admission=memory,memory_admission_method=method,available_disk_bytes_at_admission=disk)
        spec=importlib.util.spec_from_file_location('canonical_audit_native_owner',owner);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
        result=native.run_owned([str(bash),'--noprofile','--norc',shell_path(wrapper)],cwd=str(root),environment=environment,
            logpath=out/'harness.log',timeout=21600)
        record['native']=result;assert result['native_exit']==0 and not result['timed_out']
        text=(out/'harness.log').read_text(encoding='utf8',errors='replace')
        counts={key:int(re.search(r'^'+key+r': (\d+)$',text,re.M).group(1)) for key in ('PASS','WARN','FAIL')}
        assert counts['FAIL']==0 and '[PASS] Pkg.test() passed' in text and '[PASS] examples/experiments.jl executed' in text
        assert len(list(raw.glob('dev-harness-pkg-test.*')))==1 and len(list(raw.glob('dev-harness-experiments.*')))==1
        record.update(complete=True,audit_counts=counts,unfiltered_Pkg_test_and_complete_experiments_passed=True)
    except BaseException as error:record['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        delta=differences();record.update(delta);record['complete']=record['complete'] and not any(delta.values())
        inventory={};inventory_errors=[];snapshot=out/'artifact'
        try:
            spec=importlib.util.spec_from_file_location('canonical_audit_evidence_snapshot',evidence_helper)
            capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture)
            captured=capture.snapshot_members(raw,operational,snapshot)
            inventory=captured['members']
            for name in ('SOURCE_PROOF.json','PREEXECUTION.json','harness.log','run-harness.sh','WAITING_ADMISSION.json'):
                path=out/name
                if path.exists():capture.checked_copy(path,snapshot/name)
            record.update(fixture_metadata_sha256=captured['fixture_metadata_sha256'],
                special_fixture_objects=captured['special_fixture_objects'],
                artifact_snapshot_ordinary_files_only=True,fixture_targets_followed=False,
                snapshot_source_identities_and_bytes_reverified=captured['source_identities_unchanged'])
            record.update(differences())
        except BaseException as error:inventory_errors.append(dict(path=str(snapshot),error=repr(error)))
        record['complete']=record['complete'] and not inventory_errors and not any(record[key] for key in
            ('changed_sources','changed_controls','guard_read_errors','git_status','head_changed'))
        record.update(raw_and_operational_members=inventory,inventory_errors=inventory_errors,finished_utc=now())
        log=out/'harness.log';record['harness_log_sha256']=pin(log) if log.exists() else None
        write_new(out/'TERMINAL.json',record)
        if snapshot.is_dir():write_new(snapshot/'TERMINAL.json',record)
        print(json.dumps({k:v for k,v in record.items() if k not in ('source_sha256','controls_sha256','raw_and_operational_members')}),flush=True)
    return 0 if record['complete'] else 1
if __name__=='__main__':raise SystemExit(main())
