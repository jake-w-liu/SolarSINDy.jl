"""Portable data-only full-scale evidence verifier; no execution or authority.

The original transport verifier semantics remain in verify(). Contract and JSON
reader dependencies are explicit reviewed modules, never a machine-specific path.
"""
from pathlib import Path,PurePosixPath
import hashlib,json,math,re,stat,sys,time,zipfile

class FullScaleVerificationError(ValueError):
    def __init__(self,code,detail):
        self.code,self.detail=code,detail
        super().__init__(code+': '+detail)

def _need(value,code,detail):
    if not value:raise FullScaleVerificationError(code,detail)

def _clock(deadline):
    _need(deadline is None or time.monotonic_ns()<deadline,'partial_or_late_publication','original verification deadline exhausted')

def _stream_sha(stream,deadline):
    digest=hashlib.sha256()
    while True:
        _clock(deadline);body=stream.read(65536);_clock(deadline)
        if not body:break
        digest.update(body)
    return digest.hexdigest()

def verify_native_result(native,bound):
    code='nonpositive_owned_stage'
    _need(isinstance(native,dict) and type(native.get('pid')) is int and native['pid']>0 and
          type(native.get('native_exit')) is int and native['native_exit']==0 and native.get('timed_out') is False,code,'native exit/PID/timeout differs')
    wall=native.get('wall_seconds')
    _need(type(wall) in (int,float) and math.isfinite(wall) and 0<=wall<bound,code,'observed native wall exceeds original bound')
    for key,value in (('leader_reaped',True),('ownership_boundary_closed',True),('all_descendants_os_reaped',False)):
        _need(key not in native or native[key] is value,code,'native closure/reaping differs')
    pairs={('start_new_session=True','owned POSIX session has no live members after stage'),
           ('CREATE_SUSPENDED -> AssignProcessToJobObject -> ResumeThread',
            'private Windows Job Object closed with KILL_ON_JOB_CLOSE')}
    _need((native.get('ownership'),native.get('descendant_cleanup')) in pairs,code,'original source-bound ownership/cleanup pair differs')
    if 'wrapper_observed_wall_seconds' in native:
        wall=native['wrapper_observed_wall_seconds']
        _need(type(wall) in (int,float) and math.isfinite(wall) and 0<=wall<bound,code,'observed wrapper wall exceeds original bound')
    return native

def verify_stage_evidence(stage,name):
    """Original collector98f3 whole-stage guard, separate from native wall."""
    code='nonpositive_owned_stage'
    _need(type(stage) is dict and type(name) is str and bool(name),code,'original stage object/name required')
    bound=60 if name=='runtime' else 7200
    verify_native_result(stage.get('native'),bound)
    elapsed=stage.get('elapsed_seconds')
    _need(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<bound,
          code,'original whole-stage elapsed exceeds source-bound allowance')
    _need(type(stage.get('timeout_seconds')) is int and stage['timeout_seconds']==bound,
          code,'original stage timeout differs')
    return stage

def verify_workload_inventory(workloads,expected_head=None):
    keys={'full-scale-'+v+'-'+r+'-'+l for v in ('1.12.6','1.13.1') for r in ('primary','independent') for l in ('ground-only','mixed')}
    _need(type(workloads) is dict and set(workloads)==keys,'resource_workload_gap','all eight exact workloads required')
    for key,row in workloads.items():
        _need(row.get('complete') is True and row.get('mode')=='workload' and
              (expected_head is None or row.get('head')==expected_head),'resource_workload_gap','workload verification is incomplete or wrong head')
        _need(key=='full-scale-'+str(row.get('runtime'))+'-'+str(row.get('role'))+'-'+str(row.get('layout')) and
              re.fullmatch('[0-9a-f]{40}',str(row.get('head',''))) is not None and
              all(re.fullmatch('[0-9a-f]{64}',str(row.get(field,''))) is not None for field in
                  ('artifact_sha256','terminal_sha256','source_proof_sha256')),'resource_workload_gap','workload original identity/pins differ')
    return {'complete_as_structure':True,'workloads':8,'source_qualified':False,'software_qualified':False,'scientific_qualified':False}

def verify_agreement_report(report,*,policy_sha256,layouts=('ground-only','mixed'),ram_bytes=8*1024**3):
    code='agreement48_gap'
    _need(type(report) is dict and report.get('complete') is True and report.get('policy_sha256')==policy_sha256,code,'agreement header/policy differs')
    for key,count in (('table_comparisons',48),('rows_compared',7914240)):
        _need(type(report.get(key)) is int and report[key]==count,code,'agreement count differs')
    _need(type(report.get('peak_native_process_RSS_bytes')) is int and 0<report['peak_native_process_RSS_bytes']<=ram_bytes,code,'agreement RSS differs')
    _need(report.get('actual_source_access') is False and type(report.get('actual_fits')) is int and report['actual_fits']==0,code,'agreement scope differs')
    rows=report.get('reports');_need(type(rows) is list and len(rows)==48,code,'all48 agreement rows required');seen=set()
    for row in rows:
        key=(row.get('layout'),row.get('other'),row.get('index'))
        _need(key[0] in layouts and type(key[1]) is int and key[1] in (2,3,4) and type(key[2]) is int and 1<=key[2]<=8 and key not in seen,code,'agreement row identity differs');seen.add(key)
        value=row.get('result',{});n=525600 if key[2] in (1,5) else 44640
        _need(value.get('complete') is True and value.get('policy_sha256')==policy_sha256,code,'agreement row incomplete')
        for field,count in (('scheduled',n),('features_accepted',n-31),('labels_accepted',n-30),('supervised',n-61)):
            _need(type(value.get(field)) is int and value[field]==count,code,'agreement row cohort differs')
        _need(all(value.get(field) is False for field in ('source_provenance_qualified','actual_source_authority_granted','actual_fit_performed')),code,'agreement authority scope differs')
        fraction=value.get('maximum_fraction_of_bound')
        _need(type(fraction) in (int,float) and math.isfinite(fraction) and 0<=fraction<=1,code,'agreement bound differs')
    _need(seen=={(l,o,i) for l in layouts for o in (2,3,4) for i in range(1,9)},code,'agreement identity set differs')
    return {'complete_as_structure':True,'comparisons':48,'source_qualified':False,'software_qualified':False,'scientific_qualified':False}

def verify(path,expected,*,mode,runtime,role=None,layout=None,workloads=None,contract,canonical,deadline_ns=None):
    C,J=contract,canonical
    if sys.flags.optimize:raise FullScaleVerificationError('resource_workload_gap','unoptimized verification required')
    _clock(deadline_ns)
    path=Path(path);assert path.is_file() and not path.is_symlink() and path.stat().st_size<=2*1024**3
    archive_before=path.stat()
    with zipfile.ZipFile(path) as archive:
        entries=archive.infolist();assert len(entries)<=100000
        files={};folded=set();expanded=0
        for entry in entries:
            _clock(deadline_ns)
            name=entry.filename;parts=PurePosixPath(name).parts
            assert name and not name.startswith('/') and chr(92) not in name and ':' not in name
            assert all(part not in ('.','..') for part in parts) and PurePosixPath(name).as_posix()==name.rstrip('/')
            assert name.casefold() not in folded;folded.add(name.casefold())
            kind=stat.S_IFMT(entry.external_attr>>16)
            assert kind in (0,stat.S_IFREG,stat.S_IFDIR) and not entry.flag_bits&1
            if entry.is_dir():continue
            assert kind in (0,stat.S_IFREG);expanded+=entry.file_size;assert expanded<=8*1024**3
            files[name]=entry
        def read(name,limit=4*1024**2):
            assert files[name].file_size<=limit;return archive.read(name)
        def data(name,limit=4*1024**2):return J.strict_json(read(name,limit))
        source_bytes=read('SOURCE_PROOF.json');source=J.strict_json(source_bytes)
        assert source==expected and source['complete'] is True and source['all_clean_and_checkout_filters_match'] is True
        before=data('PREEXECUTION.json');terminal=data('TERMINAL.json')
        for value in (before,terminal):
            assert value['mode']==mode and value['runtime']==runtime and value['role']==role and value['layout']==layout
            assert value['head']==expected['head'] and value['source_proof_sha256']==J.digest(source_bytes)
            assert value['owner_sha256']==C.OWNER_SHA and value['actual_source_access'] is False
            assert type(value['actual_fits']) is int and value['actual_fits']==0
            assert value['scientific_performance_claim'] is False and value['software_qualification'] is False
        assert terminal['complete'] is True and 'failure' not in terminal and 'guard_failure' not in terminal
        assert not terminal['changed_sources'] and terminal['head_changed'] is False and terminal['git_status']==''
        members=terminal['members'];assert isinstance(members,dict) and 0<len(members)<=10000
        assert set(files)==set(members)|{'TERMINAL.json'}
        for name,row in members.items():
            _clock(deadline_ns)
            assert set(row)=={'bytes','sha256'} and type(row['bytes']) is int and 0<=row['bytes']<=2*1024**3
            assert files[name].file_size==row['bytes']
            with archive.open(name) as stream:assert _stream_sha(stream,deadline_ns)==row['sha256']
        expected_stages=({'runtime','case-workload','input-workload','independent-input'} if role=='primary' else {'runtime','case-workload'}) if mode=='workload' else {'runtime','native-agreement'}
        assert set(terminal['stages'])==expected_stages
        for name,stage in terminal['stages'].items():
            verify_stage_evidence(stage,name)
            C.admit(stage['available_physical_bytes'],stage['free_disk_bytes'])
            assert stage['timeout_seconds']==(60 if name=='runtime' else 7200)
            assert type(stage['native']['native_exit']) is int and stage['native']['native_exit']==0 and stage['native']['timed_out'] is False
            assert stage['raw_sha256']==members[name+'/native.log']['sha256']
        if mode=='workload':
            assert runtime in C.RUNTIMES and role in C.ROLES and layout in C.LAYOUTS
            C.validate_case_report(data('output/SCALE_RESULT.json'),runtime,role,layout)
            assert terminal['case_report_sha256']==members['output/SCALE_RESULT.json']['sha256']
            if role=='primary':
                report=data('INDEPENDENT_RESOURCE.json')
                assert report['complete'] is True and report['runtime']==runtime and report['layout']==layout
                assert type(report['verified_primitive_rows']) is int and report['verified_primitive_rows']==600000
                assert type(report['verified_full2017_input_rows']) is int and report['verified_full2017_input_rows']==1051078
                assert report['actual_source_access'] is False and report['actual_fits']==0 and report['software_qualification'] is False
                assert terminal['independent_input_sha256']==members['INDEPENDENT_RESOURCE.json']['sha256']
        else:
            assert mode=='agreement' and runtime=='1.13.1' and role is None and layout is None
            for field,count in (('full_scale_case_jobs',8),('full_scale_input_jobs',4),('table_comparisons',48),('rows_compared',7914240)):
                assert type(terminal[field]) is int and terminal[field]==count
            keys={'full-scale-'+v+'-'+r+'-'+l for v in C.RUNTIMES for r in C.ROLES for l in C.LAYOUTS}
            verify_workload_inventory(workloads,expected['head'])
            assert isinstance(workloads,dict) and set(workloads)==keys
            links=terminal['input_job_evidence'];assert isinstance(links,list) and len(links)==8
            seen=set()
            for item in links:
                key=PurePosixPath(item['directory']).name;assert key in keys and key not in seen;seen.add(key)
                assert item['terminal_sha256']==workloads[key]['terminal_sha256'] and item['source_proof_sha256']==workloads[key]['source_proof_sha256']
            report=data('NATIVE_AGREEMENT.json')
            verify_agreement_report(report,policy_sha256=C.POLICY,layouts=C.LAYOUTS,ram_bytes=C.RAM_BYTES)
            assert report['complete'] is True and report['policy_sha256']==C.POLICY
            assert type(report['table_comparisons']) is int and report['table_comparisons']==48
            assert type(report['rows_compared']) is int and report['rows_compared']==7914240
            C.integer(report['peak_native_process_RSS_bytes'],1,C.RAM_BYTES)
            assert report['actual_source_access'] is False and report['actual_fits']==0
            assert terminal['agreement_sha256']==members['NATIVE_AGREEMENT.json']['sha256']
            rows=report['reports'];assert isinstance(rows,list) and len(rows)==48;seen=set()
            for row in rows:
                key=(row['layout'],row['other'],row['index']);assert key not in seen;seen.add(key)
                assert key[0] in C.LAYOUTS and type(key[1]) is int and key[1] in (2,3,4) and type(key[2]) is int and 1<=key[2]<=8
                result=row['result'];n=525600 if key[2] in (1,5) else 44640
                assert result['complete'] is True and result['policy_sha256']==C.POLICY
                for field,count in (('scheduled',n),('features_accepted',n-31),('labels_accepted',n-30),('supervised',n-61)):
                    assert type(result[field]) is int and result[field]==count
                assert result['source_provenance_qualified'] is False and result['actual_source_authority_granted'] is False and result['actual_fit_performed'] is False
                assert type(result['maximum_fraction_of_bound']) in (int,float) and 0<=result['maximum_fraction_of_bound']<=1
            assert seen=={(l,o,i) for l in C.LAYOUTS for o in (2,3,4) for i in range(1,9)}
        with path.open('rb') as stream:artifact_sha256=_stream_sha(stream,deadline_ns)
        after=path.stat()
        assert (archive_before.st_dev,archive_before.st_ino,archive_before.st_size,archive_before.st_mtime_ns,archive_before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
        return dict(complete=True,mode=mode,runtime=runtime,role=role,layout=layout,head=expected['head'],
            artifact_sha256=artifact_sha256,terminal_sha256=J.digest(read('TERMINAL.json')),
            source_proof_sha256=J.digest(source_bytes),all_regular_members_verified=len(members),expanded_bytes=expanded,
            native_numerical_oracles_reexecuted_by_this_reader=False,actual_source_access=False,actual_fits=0,
            scientific_qualification=False)
