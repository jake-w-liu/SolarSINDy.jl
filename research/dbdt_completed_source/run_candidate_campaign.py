"""Fixed142-stage candidate campaign and explicit, separately counted previews.

Supplied grants only. No caller code, data discovery, automatic retry, source
read/fit permission from metadata, or automatic preview on missing authority.
"""
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,math,os,re,shutil,sys,time,traceback
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
DRIVER=Path(__file__).with_name('run_candidate_stage.py')
PREVIEW_ENTRY=Path(__file__).with_name('run_candidate_descriptor_preview.jl')
spec=importlib.util.spec_from_file_location('fixed_campaign_stage_driver',DRIVER)
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
LOADED_DRIVER_SHA=D.pin(DRIVER)
STATIONS=('CMO','FRD');FAMILIES=('ridge','boost','forest')
CANDIDATES=('ridge-0.01','ridge-0.1','ridge-1','ridge-10','ridge-100',
 'boost-2-50','boost-2-100','boost-2-200','boost-3-50','boost-3-100','boost-3-200',
 'boost-4-50','boost-4-100','boost-4-200','forest-6-64','forest-9-64')
MAX_INVOCATIONS=1024;MAX_SECONDS=2048400;MAX_PREVIEWS=142
class PreviewPrelaunchRejected(ValueError):
    """Source-observed failure before the fixed owner could be invoked."""
need=D.need;pin=D.pin;load=D.load;write=D.write;canonical=D.canonical
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def exact(v,names):need(isinstance(v,dict) and set(v)==set(names),'campaign exact metadata fields')
def duration(value,maximum=None):
    need(type(value) in (int,float) and math.isfinite(value) and value>=0 and
        (maximum is None or value<=maximum),'finite nonnegative non-Bool elapsed duration required')
    return value
def utc_clock(value):
    need(isinstance(value,str) and re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z',value),
        'strict UTC millisecond clock required')
    return datetime.strptime(value,'%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)
def validate_table_clock(table):
    start=utc_clock(table['start_inclusive']);stop=utc_clock(table['stop_exclusive']);asof=utc_clock(table['label_asof'])
    need(start<=utc_clock('2017-01-01T00:00:00.000Z') and stop>=utc_clock('2017-10-01T00:00:00.000Z') and
        asof>=utc_clock('2018-01-01T00:00:00.000Z') and start<stop<=asof,
        'actual campaign requires sealed JanSep2017 coverage and Jan2018 label asof before any stage call')
def queue():
    rows=[]
    for station in STATIONS:
        for candidate in CANDIDATES:
            for operation in ('candidate-input','candidate-fit','candidate-validation','independent-verify'):
                rows.append(dict(station=station,candidate_id=candidate,family=None,operation=operation,
                    key=station+'/'+candidate+'/'+operation))
        rows.append(dict(station=station,candidate_id=None,family=None,operation='select-R2',key=station+'/select-R2'))
        for family in FAMILIES:
            for operation in ('selected-refit-input','selected-refit'):
                rows.append(dict(station=station,candidate_id=None,family=family,operation=operation,key=station+'/'+family+'/'+operation))
    need(len(rows)==142 and sum(r['operation'] in ('candidate-fit','selected-refit') for r in rows)==38,'fixed campaign queue')
    return tuple(dict(r,number=i+1,directory_name=f'{i+1:03d}-'+r['key'].replace('/','-')) for i,r in enumerate(rows))
QUEUE=queue();BY_KEY={r['key']:r for r in QUEUE}
def study_directory(config,stage):return Path(config['root'])/'stages'/stage['directory_name']
def preview_directory(config,stage):return Path(config['root'])/'previews'/stage['directory_name']
def native_ref(directory,receipt):
    return dict(receipt_path=str(directory/'CANDIDATE_NATIVE_RECEIPT.json'),receipt_sha256=pin(directory/'CANDIDATE_NATIVE_RECEIPT.json'),
        raw_path=str(directory/'julia.log'),raw_sha256=receipt['raw_sha256'])
def artifact_metadata(ref,kind,*,metadata_only=False):
    exact(ref,('directory','expected','pins') if kind=='fit' else ('directory','pins'))
    directory=D.ordinary(ref['directory'],True);p=ref['pins']
    names={'input':{'descriptor_sha256':'request.json','summary_sha256':'summary.json','receipt_sha256':'receipt.json','manifest_sha256':'manifest.json','complete_sha256':'COMPLETE.json'},
        'fit':{'context_sha256':'context.json','recipe_sha256':'recipe.json','model_sha256':'model.json','receipt_sha256':'receipt.json','complete_sha256':'COMPLETE.json'},
        'validation':{'request_sha256':'request.json','context_sha256':'context.json','metric_sha256':'metric.json','receipt_sha256':'receipt.json','manifest_sha256':'manifest.json','complete_sha256':'COMPLETE.json'},
        'witness':{'manifest_sha256':'manifest.json','complete_sha256':'COMPLETE.json'}}[kind]
    for key,name in names.items():
        if metadata_only and name=='model.json':continue
        path=directory/name;need(path.stat().st_size<=64*1024**2 and pin(path)==p[key],'completed output pin differs: '+name)
    if kind=='fit':need(p['actual_fit_performed'] is True,'campaign requires actual completed fit')
def check_closed(config,stage,*,metadata_only=False):
    """Read only closed pinned execution evidence; never mint a checked token."""
    directory=study_directory(config,stage);path=directory/'CANDIDATE_NATIVE_RECEIPT.json'
    if not path.exists():
        if directory.exists():
            for name in ('CAMPAIGN_REQUEST.json','NATIVE_STAGE_ATTEMPT.json','NATIVE_OWNER_LAUNCH.json','output','JULIA_NATIVE_WAIT.json'):
                need(not (directory/name).exists() and not (directory/name).is_symlink(),'live/failed/unknown or spent stage cannot relaunch')
        return None
    r=load(path,262144)
    need(r['schema']=='candidate_native_stage_evidence_v1' and r['complete'] is True and 'failure' not in r and
        r['actual_authority'] is True and r['operation']==stage['operation'] and r['head']==config['_proof']['head'] and
        r['source_proof_sha256']==config['source_proof']['sha256'] and r['qualification_sha256']==config['qualification']['sha256'],
        'matching complete actual stage evidence required')
    reqpath=directory/'CAMPAIGN_REQUEST.json';need(Path(r['request_path'])==reqpath and pin(reqpath)==r['request_sha256'],'closed request differs')
    request=load(reqpath);D.validate_headers(request)
    need(request['execution_directory']==str(directory) and request['source_proof']==config['source_proof'] and
        request['qualification']==config['qualification'] and request['julia']==config['julia'] and request['runtime']==config['runtime'] and
        request['operation']==stage['operation'] and request['candidate_id']==stage['candidate_id'] and
        request['case']==config['stations'][stage['station']]['case'] and request['table_index']==config['stations'][stage['station']]['table_index'],
        'closed station/source/input lineage differs')
    need(r['driver_sha256']==LOADED_DRIVER_SHA==pin(DRIVER) and r['entry_sha256']==pin(D.ENTRY) and
        r['runner_sha256']==pin(D.OWNER) and r['legacy_runner_sha256']==pin(D.LEGACY_OWNER),'closed worker closure differs')
    inner=r['julia_native'];outer=r['supervisor_native']
    need(all(type(x['native_exit']) is int and x['native_exit']==0 and x['timed_out'] is False for x in (inner,outer)) and
        inner['complete'] is True and inner['reader_joined'] is True and inner['reader_pipe_close_pending'] is False and
        inner['raw_byte_limit_exceeded'] is False and inner['exact_child_reaped'] is True and outer['leader_reaped'] is True and
        r['ownership_boundary_closed'] is True and r['all_descendants_os_reaped'] is False and
        duration(r['native_owner_wall_seconds'],7200)>=0 and duration(inner['wall_seconds'],7200)>=0 and
        duration(outer['wall_seconds'],7200)>=0,'closed genuine inner/outer completion required')
    raw=directory/'julia.log';need(raw.stat().st_size==inner['raw_bytes']<=D.RAW_LIMIT and pin(raw)==r['raw_sha256']==inner['raw_sha256'],'closed raw differs')
    need(pin(directory/'supervisor.log')==r['supervisor_raw_sha256'],'supervisor raw differs')
    before=D.pinned(r['controls_before'],8*1024**2);after=D.pinned(r['controls_after'],8*1024**2)
    need(before==after and r['changed_control_count']==r['control_read_error_count']==0,'closed guarded controls differ')
    need(Path(r['controls_before']['path'])==directory/'CONTROLS_BEFORE.json' and Path(r['controls_after']['path'])==directory/'CONTROLS_AFTER.json','foreign closed controls')
    result=load(directory/'JULIA_STAGE_RESULT.json')
    need(pin(directory/'JULIA_STAGE_RESULT.json')==r['internal_result_sha256'] and result==r['outputs'] and
        result['complete'] is True and result['operation']==stage['operation'] and result['request_sha256']==r['request_sha256'] and
        result['actual_source_access'] is True and result['actual_fits']==int(stage['operation'] in ('candidate-fit','selected-refit')),
        'closed internal outputs differ')
    need(result['runtime']==r['runtime'] and r['argv']==D.julia_argv(request,reqpath,r['request_sha256']) and r['runtime']['argv']==r['argv'],
        'closed runtime/argv differs')
    for key in ('input','fit','validation','witness'):
        if key in result:artifact_metadata(result[key],key,metadata_only=metadata_only)
    if stage['operation']=='selected-refit':need(pin(directory/'selected-binding.json')==result['selected_binding_sha256'],'selected binding differs')
    return dict(stage=stage,request=request,receipt=r,outputs=result,native=native_ref(directory,r))
def prefix(config,*,metadata_only=False):
    completed={};pending=None
    for stage in QUEUE:
        value=check_closed(config,stage,metadata_only=metadata_only)
        if value is None:
            pending=stage;break
        completed[stage['key']]=value
    if pending:
        for later in QUEUE[pending['number']:]:
            d=study_directory(config,later)
            need(not (d/'CANDIDATE_NATIVE_RECEIPT.json').exists() and not (d/'NATIVE_STAGE_ATTEMPT.json').exists() and
                not (d/'CAMPAIGN_REQUEST.json').exists(),'non-prefix completed/attempted stage')
    return completed,pending
def completed_candidates(station,done):
    rows=[]
    for candidate in CANDIDATES:
        base=station+'/'+candidate+'/'
        f=done[base+'candidate-fit'];v=done[base+'candidate-validation'];w=done[base+'independent-verify']
        rows.append(dict(candidate_id=candidate,fit=f['outputs']['fit'],fit_authority=f['request']['inputs']['fit_authority'],fit_native=f['native'],
            validation=v['outputs']['validation'],validation_authority=v['request']['inputs']['authority'],validation_native=v['native'],
            verification_authority=w['request']['inputs']['authority'],verification_native=w['native'],witness=w['outputs']['witness']))
    return rows
def authority_slots(stage):
    op=stage['operation']
    return () if op=='select-R2' else ('fit_authority',) if op=='candidate-fit' else ('authority','fit_authority') if op=='selected-refit' else ('authority',)
def build_request(config,stage,done,grants):
    """Pure request construction: no source reads, callbacks, fitting or launch."""
    station=stage['station'];candidate=stage['candidate_id'];op=stage['operation'];x={}
    missing=[stage['key']+'#'+slot for slot in authority_slots(stage) if stage['key']+'#'+slot not in grants]
    if missing:return None,missing
    for slot in authority_slots(stage):x[slot]=grants[stage['key']+'#'+slot]
    if op=='candidate-fit':x['input']=done[station+'/'+candidate+'/candidate-input']['outputs']['input']
    elif op in ('candidate-validation','independent-verify'):
        f=done[station+'/'+candidate+'/candidate-fit'];x.update(fit=f['outputs']['fit'],fit_authority=f['request']['inputs']['fit_authority'],fit_native=f['native'])
        if op=='independent-verify':
            v=done[station+'/'+candidate+'/candidate-validation'];x.update(validation=v['outputs']['validation'],
                validation_authority=v['request']['inputs']['authority'],validation_native=v['native'])
    elif op in ('select-R2','selected-refit-input','selected-refit'):
        x['candidates']=completed_candidates(station,done)
        if stage['family'] is not None:x['family']=stage['family']
        if op=='selected-refit':x['input']=done[station+'/'+stage['family']+'/selected-refit-input']['outputs']['input']
    anchor=x['fit_authority'] if op=='candidate-fit' else x['candidates'][0]['verification_authority'] if op=='select-R2' else x['authority']
    return dict(schema='candidate_native_stage_request_v1',evidence_mode='actual_archival_cache',operation=op,
        execution_directory=str(study_directory(config,stage)),source_proof=config['source_proof'],qualification=config['qualification'],
        qualification_authority=anchor,julia=config['julia'],runtime=config['runtime'],case=config['stations'][station]['case'],
        table_index=config['stations'][station]['table_index'],candidate_id=candidate,inputs=x),[]
@contextmanager
def campaign_lock(root):
    path=root/'CAMPAIGN_LOCK';root=D.ordinary(root,True)
    if path.exists():D.ordinary(path)
    handle=path.open('a+b',buffering=0)
    try:
        handle.seek(0)
        if os.name=='nt':
            import msvcrt;msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl;fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if path.stat().st_size==0:handle.write(b'0');handle.flush()
        need(path.stat().st_size==1,'campaign lock file shape')
        yield
    finally:handle.close()
def validate_config(ref):
    config=D.pinned(ref);exact(config,('schema','root','source_proof','qualification','julia','runtime','preview_authority','stations'))
    need(config['schema']=='fixed_candidate_campaign_v1' and set(config['stations'])==set(STATIONS),'fixed campaign configuration')
    root=D.ordinary(config['root'],True);need(not root.is_relative_to(ROOT) and not ROOT.is_relative_to(root),'campaign outside checkout')
    proof=D.pinned(config['source_proof'],4*1024**2);D.source_guard(proof)
    need(LOADED_DRIVER_SHA==pin(DRIVER)==proof['source_sha256'][DRIVER.relative_to(ROOT).as_posix()],'fixed loaded driver outside source')
    for path in (Path(__file__),PREVIEW_ENTRY,ROOT/'research/dbdt_completed_source/candidate_descriptor_metadata.jl'):
        need(proof['source_sha256'].get(path.relative_to(ROOT).as_posix())==pin(path),'campaign/preview absent from source closure')
    D.qualification(config['qualification'],proof,config['source_proof']['sha256'])
    runtime=D.pinned(config['runtime'],131072);need(pin(config['julia']['path'])==config['julia']['sha256']==runtime['executable_sha256'],'runtime executable differs')
    for station,value in config['stations'].items():
        exact(value,('case','table_index','admission_metadata'))
        c=value['case'];exact(c,('request_path','directory','request_sha256','manifest_sha256','complete_sha256','producer','reader','admission_sha256'))
        header=D.pinned(dict(path=c['request_path'],sha256=c['request_sha256']),131072)
        manifest=D.pinned(dict(path=str(Path(c['directory'])/'manifest.json'),sha256=c['manifest_sha256']))
        complete=D.pinned(dict(path=str(Path(c['directory'])/'COMPLETE.json'),sha256=c['complete_sha256']),4096)
        admission=D.pinned(value['admission_metadata'],131072)
        need(type(value['table_index']) is int and 1<=value['table_index']<=len(header['tables'])<=16,'campaign table index')
        table=header['tables'][value['table_index']-1]
        validate_table_clock(table)
        need(header['schema']=='archive_catalog_case_request_v1' and header['evidence_mode']=='actual_archival_cache' and header['layout']=='ground-only' and
            header['runtime']==runtime['julia_version'] and table['station']==station and manifest['request_sha256']==c['request_sha256'] and
            complete['manifest_sha256']==c['manifest_sha256'] and value['admission_metadata']['sha256']==c['admission_sha256'] and
            admission['request_sha256']==c['request_sha256'] and admission['artifact_manifest_sha256']==c['manifest_sha256'] and
            admission['artifact_complete_sha256']==c['complete_sha256'],'campaign pinned header/admission lineage')
    need(config['stations']['CMO']['case']['request_sha256']==config['stations']['FRD']['case']['request_sha256'] and
        config['stations']['CMO']['table_index']!=config['stations']['FRD']['table_index'],'one fixed two-station case lineage')
    config['_proof']=proof;return config
def inbox(ref,config_sha):
    value=D.pinned(ref);exact(value,('schema','campaign_sha256','grants','previous_invocation'))
    need(value['schema']=='fixed_candidate_campaign_grant_inbox_v1' and value['campaign_sha256']==config_sha and
        isinstance(value['grants'],dict) and len(value['grants'])<=146,'bounded separately supplied inbox')
    allowed={s['key']+'#'+slot for s in QUEUE for slot in authority_slots(s)}
    need(set(value['grants'])<=allowed,'unknown grant inbox slot')
    return value['grants'],value['previous_invocation']
def invocation(root,config_ref,inbox_ref,grants,mode,previous_ref):
    directory=root/'invocations'
    if not directory.exists():directory.mkdir()
    D.ordinary(directory,True);previous={};number=None;last=None;seconds=0.;auxiliary=0
    for i in range(MAX_INVOCATIONS):
        start=directory/f'{i:04d}.START.json'
        if not start.exists():number=i;break
        old=load(start)
        need(old['campaign']==config_ref and old['previous_terminal']==last,'campaign configuration/history changed')
        previous=D.pinned(old['grant_snapshot'])
        closed=directory/f'{i:04d}.TERMINAL.json'
        need(closed.exists(),'previous invocation lost/unknown; no budget reset or automatic recovery')
        terminal=load(closed)
        need(terminal['start_sha256']==pin(start),'invocation terminal start pin differs')
        elapsed=terminal['invocation_wall_seconds'];count=terminal['auxiliary_previews_this_invocation']
        need(terminal['auxiliary_launch_count_known'] is True and terminal['study_launch_count_known'] is True,
            'unknown actual launch count blocks aggregate reuse')
        duration(elapsed,MAX_SECONDS)
        need(type(count) is int and 0<=count<=1,'closed invocation duration/count')
        seconds+=elapsed;auxiliary+=count
        duration(seconds,MAX_SECONDS)
        need(terminal['cumulative_active_seconds']==seconds and terminal['cumulative_auxiliary_previews']==auxiliary,'cumulative invocation budget changed')
        last=dict(path=str(closed),sha256=pin(closed))
    need(number is not None,'campaign invocation bound exhausted')
    need(previous_ref==last,'outside previous-invocation pin differs')
    need(all(grants.get(k)==v for k,v in previous.items()),'previously supplied grant pins changed or removed')
    snapshot=directory/f'{number:04d}.GRANTS.json';write(snapshot,grants)
    write(directory/f'{number:04d}.START.json',dict(schema='fixed_candidate_campaign_invocation_v1',campaign=config_ref,inbox=inbox_ref,mode=mode,
        grant_snapshot=dict(path=str(snapshot),sha256=pin(snapshot)),previous_terminal=last,started_utc=datetime.now(timezone.utc).isoformat()))
    need(auxiliary<=MAX_PREVIEWS,'cumulative auxiliary count exceeded')
    return directory/f'{number:04d}.TERMINAL.json',seconds,auxiliary,pin(directory/f'{number:04d}.START.json')
def closed_previews(config):
    """Across both modes, a lost preview owner blocks all further native work."""
    launched=0
    for stage in QUEUE:
        directory=preview_directory(config,stage)
        if not directory.exists():continue
        D.ordinary(directory,True);path=directory/'PREVIEW_NATIVE_RECEIPT.json'
        need(path.exists(),'live/unknown auxiliary preview blocks campaign work')
        r=load(path,262144);outer=r.get('supervisor_native')
        need(r['schema']=='candidate_descriptor_preview_native_evidence_v1' and r['stage_key']==stage['key'] and
            r['head']==config['_proof']['head'] and r['source_proof_sha256']==config['source_proof']['sha256'] and
            isinstance(outer,dict) and type(outer['native_exit']) is int and outer['leader_reaped'] is True and
            type(outer['timed_out']) is bool and outer['ownership_boundary_closed'] is True and
            r['authority_granted'] is False and r['actual_fits']==0 and r['actual_source_bodies_read'] is False and
            r['scientific_qualification'] is False,'auxiliary preview has no genuine closed ownership boundary')
        need(r['launch_count_known'] is True and r['observed_owner_launches']==1 and r['owned_attempts_charged']==1 and
            pin(directory/'PREVIEW_NATIVE_ATTEMPT.json')==r['attempt_sha256'],'closed preview actual launch/attempt evidence')
        duration(r['wall_seconds'],MAX_SECONDS);duration(outer['wall_seconds'],MAX_SECONDS)
        if r['complete'] is True:duration(r['wall_seconds'],7200)
        request=directory/'PREVIEW_REQUEST.json';need(pin(request)==r['request_sha256'],'closed preview request pin differs')
        launch=load(directory/'PREVIEW_OWNER_LAUNCH.json',16384)
        need(launch['pid']==outer['pid'] and launch['request_sha256']==r['request_sha256'],'closed preview launch identity')
        need(pin(directory/'supervisor.log')==r['supervisor_raw_sha256'],'closed preview supervisor raw differs')
        before=D.pinned(r['controls_before'],8*1024**2);after=D.pinned(r['controls_after'],8*1024**2)
        need(before==after and not r['changed_controls'] and not r['control_read_errors'],'closed preview guarded controls differ')
        need(r['driver_sha256']==LOADED_DRIVER_SHA==pin(DRIVER) and r['campaign_driver_sha256']==pin(__file__) and
            r['entry_sha256']==pin(PREVIEW_ENTRY) and r['runner_sha256']==pin(D.OWNER),'closed preview source closure differs')
        if 'julia_native' in r:
            inner=r['julia_native'];raw=directory/'julia.log'
            duration(inner['wall_seconds'],MAX_SECONDS)
            need(type(inner['native_exit']) is int and inner['exact_child_reaped'] is True and raw.stat().st_size==inner['raw_bytes']<=D.RAW_LIMIT and
                pin(raw)==inner['raw_sha256'],'closed preview observed inner wait/raw differs')
        else:need(r['complete'] is False and 'failure' in r,'missing inner wait cannot be a successful preview')
        launched+=1
    need(launched<=MAX_PREVIEWS,'fixed auxiliary count exceeded');return launched
def metadata_for_preview(config,stage,done):
    need(stage['operation']!='select-R2','pure selection needs no grant descriptor preview')
    value=config['stations'][stage['station']];case=value['case'];op=stage['operation'];candidate=stage['candidate_id'];point=None;selection=None;selection_ref=None
    if op.startswith('selected-refit'):
        choice=done[stage['station']+'/select-R2']['outputs']['selection']
        candidate=next(w['candidate_id'] for w in choice['family_winners'] if w['family']==stage['family']);selection=choice['selection_evidence_sha256']
        closed=done[stage['station']+'/select-R2'];directory=study_directory(config,closed['stage'])
        selection_ref=dict(family=stage['family'],result=dict(path=str(directory/'JULIA_STAGE_RESULT.json'),sha256=closed['receipt']['internal_result_sha256']),
            receipt=dict(path=str(directory/'CANDIDATE_NATIVE_RECEIPT.json'),sha256=closed['native']['receipt_sha256']))
    if op in ('candidate-validation','independent-verify'):
        fit=done[stage['station']+'/'+candidate+'/candidate-fit']['outputs']['fit'];directory=Path(fit['directory'])
        point=dict(descriptor=done[stage['station']+'/'+candidate+'/candidate-fit']['outputs']['point'],
            context=dict(path=str(directory/'context.json'),sha256=fit['pins']['context_sha256']),
            receipt=dict(path=str(directory/'receipt.json'),sha256=fit['pins']['receipt_sha256']))
    return dict(stage_key=stage['key'],stage_operation=op,candidate_id=candidate,table_index=value['table_index'],station=stage['station'],
        case_request=dict(path=case['request_path'],sha256=case['request_sha256']),
        manifest=dict(path=str(Path(case['directory'])/'manifest.json'),sha256=case['manifest_sha256']),
        complete=dict(path=str(Path(case['directory'])/'COMPLETE.json'),sha256=case['complete_sha256']),
        admission=value['admission_metadata'],point=point,selection_sha256=selection,selection=selection_ref)
def preview_argv(config,path,sha,directory):
    return [config['julia']['path'],'--startup-file=no','--compiled-modules=no','--pkgimages=no','--threads=1','--project='+str(ROOT),str(PREVIEW_ENTRY),
        '--request',str(path),'--request-sha256',sha,'--argv',str(directory/'preview-argv.json')]
def preview_admission(path,sha):
    need(pin(path)==sha,'preview request changed');r=load(path)
    exact(r,('schema','campaign','stage_key','execution_directory','metadata'))
    need(r['schema']=='candidate_descriptor_preview_request_v1' and r['stage_key'] in BY_KEY,'fixed preview request')
    config=validate_config(r['campaign']);stage=BY_KEY[r['stage_key']];root=preview_directory(config,stage)
    need(r['execution_directory']==str(root),'preview output outside fixed directory');D.ordinary(root,True)
    done,pending=prefix(config,metadata_only=True)
    need(pending is not None and pending['key']==stage['key'] and r['metadata']==metadata_for_preview(config,stage,done),
        'preview must bind exactly the current pending known inputs')
    authority=config['preview_authority'];grant=D.grant_ref(authority)
    need(grant['head']==config['_proof']['head'] and authority['source_proof_sha256']==config['source_proof']['sha256'] and
        authority['qualification_sha256']==config['qualification']['sha256'],'preview qualification anchor differs')
    H=D.module(ROOT/'tools/run_canonical_harness.py','candidate_preview_memory');memory,method=H.available_memory();disk=shutil.disk_usage(root).free
    need(type(memory) is int and memory>=D.RAM and disk>=D.DISK,'preview8GiB/16GiB admission before Julia')
    return r,config,stage,memory,method,disk
def preview_supervisor(path,sha):
    r,config,stage,_,_,_=preview_admission(path,sha);root=Path(r['execution_directory'])
    deadline=time.monotonic()+5
    while not (root/'PREVIEW_OWNER_LAUNCH.json').exists():
        need(time.monotonic()<deadline,'preview owned-launch binding missing');time.sleep(.01)
    launch=load(root/'PREVIEW_OWNER_LAUNCH.json',16384)
    need(launch['pid']==os.getpid() and launch['owner_pid']==os.getppid() and launch['request_sha256']==sha,'preview private supervisor binding')
    argv=load(root/'preview-argv.json',65536);need(argv==preview_argv(config,path,sha,root),'fixed preview argv differs')
    environment=dict(os.environ,JULIA_LOAD_PATH=os.pathsep.join(('@','@stdlib')),JULIA_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        JULIA_PKG_PRECOMPILE_AUTO='0',SOLARSINDY_NO_OPEN='1',PYTHONDONTWRITEBYTECODE='1')
    outcome=D.pump_process(argv,str(ROOT),environment,root/'julia.log')
    try:write(root/'JULIA_NATIVE_WAIT.json',outcome)
    finally:
        if outcome.get('reader_joined') is False:os._exit(1)
    return 0 if outcome['native_exit']==0 and outcome['complete'] and not outcome['timed_out'] else 1
def prepare_preview(config,stage,done,config_ref):
    known=metadata_for_preview(config,stage,done)
    H=D.module(ROOT/'tools/run_canonical_harness.py','candidate_preview_initial_memory');memory,_=H.available_memory()
    need(type(memory) is int and memory>=D.RAM and shutil.disk_usage(config['root']).free>=D.DISK,
        'preview resource gate before reserving its single attempt')
    root=preview_directory(config,stage);need(not root.exists() and not root.is_symlink(),'auxiliary preview already attempted; no retry')
    parent=root.parent
    if not parent.exists():parent.mkdir()
    D.ordinary(parent,True);root.mkdir()
    request=dict(schema='candidate_descriptor_preview_request_v1',campaign=config_ref,stage_key=stage['key'],execution_directory=str(root),
        metadata=known)
    path=root/'PREVIEW_REQUEST.json';write(path,request);sha=pin(path)
    r,_,_,memory,method,disk=preview_admission(path,sha)
    argv=preview_argv(config,path,sha,root);write(root/'preview-argv.json',argv)
    controls={str(p):pin(p) for p in (Path(__file__),DRIVER,D.OWNER,D.LEGACY_OWNER,PREVIEW_ENTRY,path,root/'preview-argv.json',
        Path(sys.executable),Path(config['julia']['path']),Path(config_ref['path']),Path(config['source_proof']['path']),
        Path(config['qualification']['path']),Path(config['runtime']['path']))}
    for name in ('grant','source_proof','qualification'):
        ref=config['preview_authority'];controls[ref[name+'_path']]=ref[name+'_sha256']
    qualified=D.pinned(config['qualification']);proofs=qualified['candidate_pipeline_full18']
    for job in proofs['jobs']:controls[job['raw_path']]=job['raw_sha256']
    controls[proofs['harness_raw_path']]=proofs['harness_raw_sha256']
    for known in config['stations'].values():
        case=known['case']
        for name,ref in (('request',dict(path=case['request_path'],sha256=case['request_sha256'])),
            ('manifest',dict(path=str(Path(case['directory'])/'manifest.json'),sha256=case['manifest_sha256'])),
            ('complete',dict(path=str(Path(case['directory'])/'COMPLETE.json'),sha256=case['complete_sha256'])),
            ('admission',known['admission_metadata'])):
            controls[ref['path']]=ref['sha256']
    # Explicit metadata whitelist: never generic input_controls/case-directory traversal.
    for name in ('case_request','manifest','complete','admission'):
        ref=request['metadata'][name];D.pinned(ref,1048576 if name=='manifest' else 131072);controls[ref['path']]=ref['sha256']
    point=request['metadata']['point']
    if point:
        for name in ('context','receipt'):
            ref=point[name];D.pinned(ref,131072);controls[ref['path']]=ref['sha256']
    selection=request['metadata']['selection']
    if selection:
        for name in ('result','receipt'):
            ref=selection[name];D.pinned(ref,1048576 if name=='result' else 262144);controls[ref['path']]=ref['sha256']
    need(all(pin(p)==s for p,s in controls.items()),'preview initial pinned controls differ')
    write(root/'PREVIEW_CONTROLS_BEFORE.json',controls)
    record=dict(schema='candidate_descriptor_preview_native_evidence_v1',complete=False,stage_key=stage['key'],head=config['_proof']['head'],
        request_sha256=sha,source_proof_sha256=config['source_proof']['sha256'],available_physical_bytes=memory,memory_method=method,free_disk_bytes=disk,
        stage_seconds_limit=7200,raw_bytes_limit=D.RAW_LIMIT,descriptor_only=True,authority_granted=False,actual_source_bodies_read=False,
        actual_fits=0,scientific_qualification=False,all_descendants_os_reaped=False,campaign_driver_sha256=pin(__file__),driver_sha256=pin(DRIVER),
        entry_sha256=pin(PREVIEW_ENTRY),runner_sha256=pin(D.OWNER),
        owned_attempts_charged=0,observed_owner_launches=0,launch_count_known=True,
        controls_before=dict(path=str(root/'PREVIEW_CONTROLS_BEFORE.json'),sha256=pin(root/'PREVIEW_CONTROLS_BEFORE.json')))
    return root,path,sha,controls,record
def run_preview(config,stage,done,config_ref):
    try:root,path,sha,controls,record=prepare_preview(config,stage,done,config_ref)
    except BaseException as error:raise PreviewPrelaunchRejected(str(error)) from error
    started=time.monotonic()
    try:
        owner=D.module(D.OWNER,'candidate_campaign_preview_owner')
        attempt=root/'PREVIEW_NATIVE_ATTEMPT.json'
        write(attempt,dict(schema='candidate_descriptor_preview_attempt_v1',request_sha256=sha,head=config['_proof']['head'],
            owner_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),owned_attempt_charge=1))
        record['owned_attempts_charged']=1;record['attempt_sha256']=pin(attempt)
        record['launch_count_known']=False
        def capture_launch(pid):
            need(type(pid) is int and pid>0,'preview observed owner PID')
            record['observed_owner_launches']=1;record['observed_owner_pid']=pid;record['launch_count_known']=True
            write(root/'PREVIEW_OWNER_LAUNCH.json',dict(pid=pid,owner_pid=os.getpid(),request_sha256=sha))
        outer=owner.run_owned([sys.executable,'-B','-X','utf8',str(Path(__file__)),'--preview-supervisor','--request',str(path),'--request-sha256',sha],
            cwd=str(ROOT),environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),logpath=root/'supervisor.log',timeout=7200,
            on_launch=capture_launch)
        record['supervisor_native']=outer;record['supervisor_raw_sha256']=pin(root/'supervisor.log')
        inner=load(root/'JULIA_NATIVE_WAIT.json',16384);record['julia_native']=inner
        need(type(outer['native_exit']) is int and outer['native_exit']==0 and not outer['timed_out'] and outer['leader_reaped'] is True and
            outer['ownership_boundary_closed'] is True and type(inner['native_exit']) is int and inner['native_exit']==0 and inner['complete'] is True and
            inner['timed_out'] is False and inner['reader_joined'] is True and inner['exact_child_reaped'] is True and
            inner['raw_bytes']==(root/'julia.log').stat().st_size<=D.RAW_LIMIT and pin(root/'julia.log')==inner['raw_sha256'] and
            time.monotonic()-started<=7200,'preview genuine inner/outer wait required')
        result=load(root/'PREVIEW_RESULT.json');need(result['request_sha256']==sha and result['descriptor_only'] is True and
            result['actual_source_bodies_read'] is False and result['actual_fits']==0 and result['authority_granted'] is False,'preview descriptor-only result differs')
        expected=D.pinned(config['runtime'],131072);runtime=result['runtime']
        need(runtime['argv']==argv and {k:v for k,v in runtime.items() if k!='argv'}=={k:v for k,v in expected.items() if k!='argv'},'preview native runtime differs')
        record.update(complete=True,result_sha256=pin(root/'PREVIEW_RESULT.json'),descriptor_sha256=result['descriptor_sha256'],runtime=runtime)
    except BaseException as error:record['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        record['wall_seconds']=time.monotonic()-started;after={};errors={}
        for p,s in controls.items():
            try:after[p]=pin(p)
            except BaseException as error:errors[p]=dict(type=type(error).__name__,message=str(error)[:128])
        write(root/'PREVIEW_CONTROLS_AFTER.json',after);write(root/'PREVIEW_CONTROL_READ_ERRORS.json',errors)
        record['controls_after']=dict(path=str(root/'PREVIEW_CONTROLS_AFTER.json'),sha256=pin(root/'PREVIEW_CONTROLS_AFTER.json'))
        record['changed_controls']=[p for p,s in controls.items() if after.get(p)!=s];record['control_read_errors']=errors
        try:D.source_guard(config['_proof'])
        except BaseException as error:record['source_failure']=str(error)
        record['complete']=record['complete'] and not record['changed_controls'] and not errors and 'failure' not in record and 'source_failure' not in record
        write(root/'PREVIEW_NATIVE_RECEIPT.json',record)
    return record
def main(config_ref,inbox_ref,*,preview=False):
    started=time.monotonic();need(not sys.flags.optimize,'unoptimized campaign required')
    header=D.pinned(config_ref);root=D.ordinary(header['root'],True)
    need(not root.is_relative_to(ROOT) and not ROOT.is_relative_to(root),'campaign outside checkout')
    grants,previous=inbox(inbox_ref,config_ref['sha256'])
    with campaign_lock(root):
        terminal,prior_seconds,prior_auxiliary,start_sha=invocation(root,config_ref,inbox_ref,grants,'preview' if preview else 'run',previous)
        result=dict(schema='fixed_candidate_campaign_terminal_v1',complete=False,status='started',campaign=config_ref,inbox=inbox_ref,
            study_operations_expected=142,study_fits_expected=38,study_launches_this_invocation=0,auxiliary_previews_this_invocation=0,
            launch_counts_are_observed_lower_bounds=True,auxiliary_launch_count_known=True,study_launch_count_known=True,owned_preview_attempts_charged=0,
            start_sha256=start_sha,actual_scientific_qualification=False,source_or_fit_permission_from_metadata=False)
        try:
            need(prior_seconds+time.monotonic()-started<MAX_SECONDS,'cumulative active campaign budget exhausted')
            config=validate_config(config_ref)
            for ref in grants.values():D.grant_ref(ref)
            observed_auxiliary=closed_previews(config)
            need(observed_auxiliary==prior_auxiliary,'auxiliary launch history differs or was lost')
            done,stage=prefix(config,metadata_only=preview)
            if preview:
                need(prior_auxiliary<MAX_PREVIEWS and prior_seconds+time.monotonic()-started+7200<=MAX_SECONDS,'cumulative auxiliary/time budget exhausted')
                need(stage is not None,'campaign already complete')
                result['auxiliary_preview_attempts_this_invocation']=1
                result['auxiliary_launch_count_known']=False;prelaunch_rejected=False
                try:value=run_preview(config,stage,done,config_ref)
                except PreviewPrelaunchRejected:
                    prelaunch_rejected=True;result['auxiliary_launch_count_known']=True
                    raise
                finally:
                    receipt=preview_directory(config,stage)/'PREVIEW_NATIVE_RECEIPT.json'
                    if not prelaunch_rejected and receipt.exists():
                        observed=load(receipt,262144)
                        result['owned_preview_attempts_charged']=observed['owned_attempts_charged']
                        result['auxiliary_previews_this_invocation']=observed['observed_owner_launches']
                        result['auxiliary_launch_count_known']=observed['launch_count_known']
                    elif not prelaunch_rejected and (preview_directory(config,stage)/'PREVIEW_NATIVE_ATTEMPT.json').exists():
                        result['owned_preview_attempts_charged']=1;result['auxiliary_launch_count_known']=False
                result['auxiliary_preview_wall_seconds']=value['wall_seconds'];result['status']='preview_complete' if value['complete'] else 'preview_failed'
            else:
                while stage is not None:
                    need(prior_seconds+time.monotonic()-started<MAX_SECONDS,'cumulative active campaign envelope exceeded')
                    request,missing=build_request(config,stage,done,grants)
                    if missing:
                        result.update(status='awaiting_authority',pending_stage=stage['key'],missing_grant_slots=missing,
                            known_metadata=metadata_for_preview(config,stage,done) if stage['operation']!='select-R2' else None);break
                    need(prior_seconds+time.monotonic()-started+7200<=MAX_SECONDS,'remaining cumulative budget cannot admit a full fixed stage')
                    directory=study_directory(config,stage)
                    if not directory.parent.exists():directory.parent.mkdir()
                    D.ordinary(directory.parent,True)
                    if not directory.exists():directory.mkdir()
                    pending_path=terminal.parent/(terminal.stem+f'.{stage["number"]:03d}.REQUEST.json')
                    write(pending_path,request);sha=pin(pending_path)
                    D.admit_request(pending_path,sha)
                    path=directory/'CAMPAIGN_REQUEST.json';write(path,request);need(pin(path)==sha,'reserved request changed')
                    # Fixed sealed implementation owns qualification/resources and actual native work.
                    result['study_calls_this_invocation']=result.get('study_calls_this_invocation',0)+1
                    result['study_launch_count_known']=False
                    try:status=D.main(path,sha)
                    finally:
                        launch=directory/'NATIVE_OWNER_LAUNCH.json'
                        if launch.exists():
                            need(load(launch,16384)['request_sha256']==sha,'campaign observed launch request differs')
                            result['study_launches_this_invocation']+=1
                            result['study_launch_count_known']=True
                    need(type(status) is int and status==0,'fixed stage failed; never retry')
                    completed=check_closed(config,stage);need(completed is not None,'fixed stage closed evidence missing')
                    done[stage['key']]=completed
                    stage=QUEUE[len(done)] if len(done)<142 else None
                if stage is None:result.update(complete=True,status='complete')
            result['completed_study_operations']=len(done)
            result['completed_study_fits']=sum(v['stage']['operation'] in ('candidate-fit','selected-refit') for v in done.values())
            result['completed_prefix']=[dict(stage=k,native=v['native']) for k,v in done.items()]
        except BaseException as error:result.update(status='blocked',failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        finally:
            result['invocation_wall_seconds']=time.monotonic()-started
            result['cumulative_active_seconds']=prior_seconds+result['invocation_wall_seconds']
            result['cumulative_auxiliary_previews']=prior_auxiliary+result['auxiliary_previews_this_invocation']
            if result['cumulative_active_seconds']>MAX_SECONDS:result.update(complete=False,status='cumulative_budget_exceeded')
            need(len(canonical(result))<=1048576,'campaign terminal bound');write(terminal,result)
        return 0 if result['complete'] or result['status'] in ('awaiting_authority','preview_complete') else 1
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path);parser.add_argument('--campaign-sha256')
    parser.add_argument('--inbox',type=Path);parser.add_argument('--inbox-sha256');parser.add_argument('--preview',action='store_true')
    parser.add_argument('--preview-supervisor',action='store_true');parser.add_argument('--request',type=Path);parser.add_argument('--request-sha256')
    args=parser.parse_args()
    if args.preview_supervisor:
        need(args.request is not None and args.request_sha256 is not None and not args.preview and args.campaign is None and args.inbox is None,'fixed private preview supervisor')
        raise SystemExit(preview_supervisor(args.request,args.request_sha256))
    need(args.request is None and args.request_sha256 is None and args.campaign is not None and args.inbox is not None,'fixed campaign arguments')
    raise SystemExit(main(dict(path=str(args.campaign),sha256=args.campaign_sha256),dict(path=str(args.inbox),sha256=args.inbox_sha256),preview=args.preview))
