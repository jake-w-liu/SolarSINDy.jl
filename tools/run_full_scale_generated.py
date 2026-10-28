"""Owned full-size generated workloads and cross-runtime evidence agreement.

All output is retained outside the checkout. The workload provides numerical
and resource evidence; it grants no archival access or scientific acceptance.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib.util, json, math, os, re, shutil, stat, subprocess, sys, time, traceback

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
RAM_BYTES = 8*GIB
DISK_BYTES = 16*GIB
STAGE_SECONDS = 7200
RUNTIMES = ('1.12.6', '1.13.1')
ROLES = ('primary', 'independent')
LAYOUTS = ('ground-only', 'mixed')
POLICY = '666f9e8112c4ee7747e2121e4cb3b5ed38cf081484468e4cc2399bea271b8c94'
OWNER_SHA = '0b0f2327841230e5ac87d03bd67d1cbaf10717e0a52ce0134d71169a36146b89'

def pin(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def need(condition, message):
    if not condition:
        raise ValueError(message)

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

def ordinary(path, directory=False):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts, 'absolute ordinary path without parent traversal required')
    for item in (path, *path.parents):
        info = item.lstat()
        need(not stat.S_ISLNK(info.st_mode) and not (getattr(info, 'st_file_attributes', 0)&0x400), 'linked path forbidden')
    need(path.is_dir() if directory else stat.S_ISREG(path.lstat().st_mode), 'ordinary path kind differs')
    return path

def fresh_output(path):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts and not path.exists() and not path.is_symlink(), 'fresh absolute output without parent traversal required')
    ordinary(path.parent, True)
    need(not path.is_relative_to(ROOT) and not ROOT.is_relative_to(path), 'output must be outside and not contain checkout')
    return path

def integer(value, minimum, maximum):
    need(type(value) is int and minimum <= value <= maximum, 'bounded integer required')
    return value

def _pairs(pairs):
    value = {}
    for key, item in pairs:
        need(key not in value, 'duplicate JSON key')
        value[key] = item
    return value

def load(path, limit=1048576):
    path = ordinary(path)
    before = path.stat()
    need(0 <= before.st_size <= limit, 'JSON byte bound')
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    after = path.stat()
    need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
         (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'metadata changed')
    need(len(raw)==before.st_size, 'metadata length differs')
    def constant(value):
        raise ValueError('nonfinite JSON constant')
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=constant)
    def shape(item, depth=0):
        need(depth<=20, 'metadata nesting bound')
        if isinstance(item,dict):
            need(len(item)<=8192 and all(isinstance(key,str) for key in item), 'metadata object bound')
            for key,child in item.items():
                shape(key,depth+1);shape(child,depth+1)
        elif isinstance(item,list):
            need(len(item)<=8192, 'metadata array bound')
            for child in item:shape(child,depth+1)
        elif isinstance(item,str):need(len(item.encode('utf-8'))<=4096, 'metadata string bound')
        elif isinstance(item,float):need(math.isfinite(item), 'metadata nonfinite float')
        else:need(item is None or type(item) in (bool,int), 'metadata atom type')
    shape(value)
    return value

def write_new(path, value, canonical=False):
    text = json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,
                      **({'separators':(',',':')} if canonical else {'indent':2}))+'\n'
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(text)

def admit(memory, disk):
    integer(memory,0,2**64-1);integer(disk,0,2**64-1)
    need(memory>=RAM_BYTES, '8GiB available physical memory required')
    need(disk>=DISK_BYTES, '16GiB free disk required')

def validate_case_report(report, runtime, role, layout):
    need(runtime in RUNTIMES and role in ROLES and layout in LAYOUTS, 'full-size job identity differs')
    need(report['schema']=='generated_case_artifact_scale_report_v1' and report['complete'] is True, 'case report incomplete')
    need((report['runtime'],report['role'],report['layout'])==(runtime,role,layout), 'case report identity differs')
    need(report['actual_source_access'] is False and report['actual_features'] is False and
         report['forecast_skill_qualified'] is False and report['novelty_qualified'] is False, 'generated evidence permissions differ')
    integer(report['actual_fits'],0,0);integer(report['artifact_members'],334,334)
    integer(report['peak_native_process_RSS_bytes'],1,RAM_BYTES)
    need(isinstance(report['tables'],list) and len(report['tables'])==8, 'eight full-size tables required')
    expected=[(station+'-generated-'+month,525600 if month=='2017-01' else 44640)
              for station in ('CMO','FRD') for month in ('2017-01','2023-03','2024-05','2024-10')]
    for table,(name,rows) in zip(report['tables'],expected):
        need(table['id']==name, 'full-size table order/identity differs')
        for key,value in (('scheduled',rows),('features_accepted',rows-31),('labels_accepted',rows-30),('supervised',rows-61)):
            integer(table[key],value,value)
    for key,total in (('scheduled',1319040),('features_accepted',1318792),('labels_accepted',1318800),('supervised',1318552)):
        need(sum(table[key] for table in report['tables'])==total, 'whole-schedule count differs')
    return report

def git(*args, input=None):
    return subprocess.check_output(['git','-C',str(ROOT),*args],input=input,stderr=subprocess.PIPE,timeout=120)

def source_proof():
    need(not git('status','--porcelain'), 'full-size verification requires clean committed source')
    head = git('rev-parse','HEAD').decode().strip()
    names = git('ls-files','-z').decode().split('\0')[:-1]
    source = {};objects = {}
    for name in names:
        path = ordinary(ROOT/name)
        body = path.read_bytes();source[name] = hashlib.sha256(body).hexdigest()
        objects[name] = git('rev-parse','HEAD:'+name).decode().strip()
        need(git('hash-object','--path='+name,'--stdin',input=body).decode().strip()==objects[name], 'Git clean filter differs')
        need(hashlib.sha256(git('cat-file','--filters','HEAD:'+name)).hexdigest()==source[name], 'Git checkout filter differs')
    return dict(complete=True,head=head,tree=git('rev-parse','HEAD^{tree}').decode().strip(),
                source_files=len(source),source_sha256=source,commit_object_ids=objects,all_clean_and_checkout_filters_match=True)

def inventory(directory):
    result = {}
    for path in sorted(Path(directory).rglob('*')):
        ordinary(path,path.is_dir())
        if path.is_file():
            need(path.name!='TERMINAL.json' or path.parent!=directory,'terminal must be written last')
            result[path.relative_to(directory).as_posix()] = dict(bytes=path.stat().st_size,sha256=pin(path))
    return result

def check_inventory(directory, members):
    need(isinstance(members,dict) and 0<len(members)<=10000, 'resource inventory size differs')
    for name,entry in members.items():
        need(isinstance(name,str) and '\\' not in name and not Path(name).is_absolute() and
             all(part not in ('','.','..') for part in name.split('/')), 'resource inventory path differs')
        path = ordinary(directory/name)
        integer(entry['bytes'],0,2*GIB)
        need(path.stat().st_size==entry['bytes'] and pin(path)==entry['sha256'], 'resource artifact bytes differ')
    actual = set()
    for path in directory.rglob('*'):
        ordinary(path,path.is_dir())
        if path.is_file():actual.add(path.relative_to(directory).as_posix())
    need(actual==set(members)|{'TERMINAL.json'}, 'resource artifact fileset differs')

def verify_resource(report_path, report_pin, destination):
    reader = module(ROOT/'tools/point_input_export/bundle_reader.py','full_scale_input_reader')
    primitive = module(ROOT/'tools/point_input_export/resource_values.py','full_scale_primitive_reader')
    report_path=ordinary(report_path);need(pin(report_path)==report_pin,'resource report pin differs')
    report=load(report_path);width=11 if report['layout']=='ground-only' else 8
    need(report['complete'] is True and report['runtime'] in RUNTIMES and report['layout'] in LAYOUTS,'input resource context differs')
    need(report['actual_source_access'] is False and report['scientific_performance_claim'] is False,'input resource permissions differ')
    integer(report['actual_fits'],0,0);integer(report['peak_RSS_bytes'],1,RAM_BYTES)
    primitive_result=primitive.verify_primitive(report_path.parent/'primitive',report['primitive'],width=width)
    case_root=report_path.parent.parent/'output';case_request_path=case_root/'case-request.json'
    case_request=load(case_request_path);case_directory=case_root/'cases'
    need(case_request['role']=='primary' and case_request['layout']==report['layout'] and
         case_request['runtime']==report['runtime'] and case_request['evidence_mode']=='generated','resource source case differs')
    entries=report['exports'];need(isinstance(entries,list) and len(entries)==2,'both full2017stations required')
    results=[]
    for entry,index,station in zip(entries,(1,5),('CMO','FRD')):
        need(entry['table_index']==index and entry['station']==station,'full2017input station/index differs')
        directory=report_path.parent/(station+'-full-year')
        need(Path(entry['directory'])==directory,'full2017export path differs')
        integer(entry['rows'],525539,525539);integer(entry['columns'],width,width)
        integer(entry['chunks'],2,16);integer(entry['peak_RSS_bytes'],1,RAM_BYTES)
        need(entry['entry_sha256']==pin(ROOT/'research/dbdt_completed_source/point_input_export.jl'),'point exporter code pin differs')
        result=reader.verify_bundle(directory,case_directory,case_request_path,
            **{key:entry[key] for key in ('manifest_sha256','complete_sha256','context_sha256','receipt_sha256','input_sha256',
               'entry_sha256','numeric_source_sha256','library_sha256')},
            case_request_sha256=pin(case_request_path),case_manifest_sha256=pin(case_directory/'manifest.json'),
            case_complete_sha256=pin(case_directory/'COMPLETE.json'),table_index=index,
            start='2017-01-01T00:00:00.000Z',stop='2018-01-01T00:00:00.000Z',asof='2018-01-01T00:31:00.000Z')
        need(result['rows']==525539 and result['scheduled_rows_scanned']==525600 and result['columns']==width,
             'independent full2017input count differs')
        need(result['actual_source_access'] is False and result['fit_authority_granted'] is False,'input verifier granted authority')
        results.append(dict(station=station,verification=result))
    need(pin(report_path)==report_pin,'resource report changed during read')
    write_new(destination,dict(complete=True,finished_utc=now(),runtime=report['runtime'],layout=report['layout'],
        primitive=primitive_result,exports=results,verified_primitive_rows=600000,verified_full2017_input_rows=1051078,
        actual_source_access=False,actual_fits=0,software_qualification=False))

def validate_job_evidence(directory, runtime, role, layout, proof):
    directory=ordinary(directory,True)
    terminal=load(directory/'TERMINAL.json',4*GIB//1024)
    need(terminal['complete'] is True and terminal['mode']=='workload' and terminal['head']==proof['head'], 'resource job incomplete or wrong source')
    need((terminal['runtime'],terminal['role'],terminal['layout'])==(runtime,role,layout),'resource job identity differs')
    need(terminal['actual_source_access'] is False and terminal['scientific_performance_claim'] is False,'resource job authority differs')
    integer(terminal['actual_fits'],0,0)
    need(not terminal['changed_sources'] and not terminal['head_changed'] and not terminal['git_status'],'resource job source changed')
    candidate=load(directory/'SOURCE_PROOF.json')
    need(candidate==proof and pin(directory/'SOURCE_PROOF.json')==terminal['source_proof_sha256'],'resource job source proof differs')
    validate_case_report(load(directory/'output/SCALE_RESULT.json'),runtime,role,layout)
    check_inventory(directory,terminal['members'])
    if role=='primary':
        checked=load(directory/'INDEPENDENT_RESOURCE.json')
        need(checked['complete'] is True and checked['runtime']==runtime and checked['layout']==layout,'full-size independent input verification incomplete')
        integer(checked['verified_primitive_rows'],600000,600000)
        integer(checked['verified_full2017_input_rows'],1051078,1051078)
    return dict(directory=str(directory),terminal_sha256=pin(directory/'TERMINAL.json'),source_proof_sha256=pin(directory/'SOURCE_PROOF.json'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('workload','agreement'),required=True)
    parser.add_argument('--evidence-dir',type=Path,required=True)
    parser.add_argument('--runtime',choices=RUNTIMES,required=True)
    parser.add_argument('--role',choices=ROLES);parser.add_argument('--layout',choices=LAYOUTS)
    parser.add_argument('--input-directory',type=Path)
    args=parser.parse_args()
    if args.mode=='workload':need(args.role in ROLES and args.layout in LAYOUTS and args.input_directory is None,'workload arguments differ')
    else:need(args.runtime=='1.13.1' and args.role is None and args.layout is None and args.input_directory is not None,'agreement arguments differ')
    out=fresh_output(args.evidence_dir);out.mkdir()
    owner_path=ROOT/'research/dbdt_completed_source/native_stage.py'
    need(pin(owner_path)==OWNER_SHA,'owned native runner pin differs')
    native=module(owner_path,'full_scale_native_owner')
    H=module(ROOT/'tools/run_canonical_harness.py','full_scale_physical_admission')
    proof=source_proof();write_new(out/'SOURCE_PROOF.json',proof)
    julia=Path(shutil.which('julia') or '').resolve(strict=True);ordinary(julia)
    record=dict(complete=False,started_utc=now(),mode=args.mode,runtime=args.runtime,role=args.role,layout=args.layout,
        head=proof['head'],source_proof_sha256=pin(out/'SOURCE_PROOF.json'),stages={},actual_source_access=False,
        actual_fits=0,scientific_performance_claim=False,software_qualification=False,
        julia_binary_sha256=pin(julia),owner_sha256=OWNER_SHA)
    write_new(out/'PREEXECUTION.json',record)
    def guard():
        changed=[name for name,sha in proof['source_sha256'].items() if pin(ordinary(ROOT/name))!=sha]
        delta=dict(changed_sources=changed,head_changed=git('rev-parse','HEAD').decode().strip()!=proof['head'],
                   git_status=git('status','--porcelain').decode())
        need(not any(delta.values()),'committed source changed during full-size verification')
        return delta
    def run(name,command,timeout=STAGE_SECONDS):
        guard();memory,method=H.available_memory();disk=shutil.disk_usage(out).free;admit(memory,disk)
        stage=out/name;stage.mkdir();tmp=stage/'tmp';tmp.mkdir()
        environment=dict(os.environ,TMP=str(tmp),TEMP=str(tmp),TMPDIR=str(tmp),JULIA_NUM_THREADS='1',
            JULIA_PKG_PRECOMPILE_AUTO='0',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',SOLARSINDY_NO_OPEN='1',PYTHONUTF8='1')
        started=time.monotonic()
        result=native.run_owned(command,cwd=str(ROOT),environment=environment,logpath=stage/'native.log',timeout=timeout)
        record['stages'][name]=dict(native=result,raw_sha256=pin(stage/'native.log'),
            available_physical_bytes=memory,memory_method=method,free_disk_bytes=disk,timeout_seconds=timeout,
            elapsed_seconds=time.monotonic()-started)
        guard();need(result['native_exit']==0 and not result['timed_out'],name+' failed; preserve raw evidence')
        return stage
    command=[str(julia),'--startup-file=no','--pkgimages=no','--threads=1','--project='+str(ROOT)]
    try:
        stage=run('runtime',[str(julia),'--version'],60)
        need((stage/'native.log').read_text(encoding='utf-8').strip()=='julia version '+args.runtime,'native runtime differs')
        if args.mode=='workload':
            run('case-workload',command+[str(ROOT/'examples/archive_case_job_scale_synthetic.jl'),str(out/'output'),args.role,args.layout])
            report=validate_case_report(load(out/'output/SCALE_RESULT.json'),args.runtime,args.role,args.layout)
            record['case_report_sha256']=pin(out/'output/SCALE_RESULT.json')
            if args.role=='primary':
                request=dict(schema='generated_input_export_resource_request_v1',actual_source_access=False,
                    runtime=args.runtime,layout=args.layout,case_request_path=str(out/'output/case-request.json'),
                    case_directory=str(out/'output/cases'),case_request_sha256=pin(out/'output/case-request.json'),
                    case_manifest_sha256=pin(out/'output/cases/manifest.json'),case_complete_sha256=pin(out/'output/cases/COMPLETE.json'))
                request_path=out/'RESOURCE_REQUEST.json';write_new(request_path,request,canonical=True)
                entry=ROOT/'research/dbdt_completed_source/point_input_export.jl'
                run('input-workload',command+[str(ROOT/'examples/point_input_export_resource_synthetic.jl'),
                    str(entry),pin(entry),str(request_path),pin(request_path),str(out/'resource')])
                resource_report=out/'resource/REPORT.json'
                run('independent-input',[sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--verify-resource',
                    str(resource_report),pin(resource_report),str(out/'INDEPENDENT_RESOURCE.json')])
                checked=load(out/'INDEPENDENT_RESOURCE.json')
                need(checked['complete'] is True and checked['verified_full2017_input_rows']==1051078,'full-size independent input verification failed')
                record['independent_input_sha256']=pin(out/'INDEPENDENT_RESOURCE.json')
        else:
            inputs=ordinary(args.input_directory,True)
            need(not inputs.is_relative_to(ROOT) and not ROOT.is_relative_to(inputs),'agreement inputs must be external')
            expected={'full-scale-'+runtime+'-'+role+'-'+layout for runtime in RUNTIMES for role in ROLES for layout in LAYOUTS}
            need({path.name for path in inputs.iterdir()}==expected,'all and only eight job artifacts required')
            evidence=[]
            for runtime in RUNTIMES:
                for role in ROLES:
                    for layout in LAYOUTS:
                        evidence.append(validate_job_evidence(inputs/('full-scale-'+runtime+'-'+role+'-'+layout),runtime,role,layout,proof))
            need(not (out/'NATIVE_AGREEMENT.json').exists(),'agreement report already exists')
            run('native-agreement',command+[str(ROOT/'examples/archive_case_full_scale_agreement.jl'),str(inputs),str(out/'NATIVE_AGREEMENT.json')])
            agreement=load(out/'NATIVE_AGREEMENT.json')
            need(agreement['complete'] is True and agreement['policy_sha256']==POLICY,'native agreement incomplete or wrong policy')
            integer(agreement['table_comparisons'],48,48);integer(agreement['rows_compared'],7914240,7914240)
            integer(agreement['peak_native_process_RSS_bytes'],1,RAM_BYTES)
            need(agreement['actual_source_access'] is False,'agreement gained source access');integer(agreement['actual_fits'],0,0)
            for runtime in RUNTIMES:
                for role in ROLES:
                    for layout in LAYOUTS:
                        validate_job_evidence(inputs/('full-scale-'+runtime+'-'+role+'-'+layout),runtime,role,layout,proof)
            record.update(agreement_sha256=pin(out/'NATIVE_AGREEMENT.json'),input_job_evidence=evidence,
                          full_scale_case_jobs=8,full_scale_input_jobs=4,table_comparisons=48,rows_compared=7914240)
        guard();record['complete']=True
    except BaseException as error:
        record['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        try:record.update(guard())
        except BaseException as error:record.update(complete=False,guard_failure=repr(error))
        record['members']=inventory(out)
        record['finished_utc']=now();write_new(out/'TERMINAL.json',record)
        print(json.dumps({key:value for key,value in record.items() if key not in ('members','stages')}),flush=True)
    return 0 if record['complete'] else 1

if __name__=='__main__':
    if sys.argv[1:2]==['--verify-resource']:
        need(len(sys.argv)==5,'resource worker arguments differ')
        verify_resource(Path(sys.argv[2]),sys.argv[3],fresh_output(Path(sys.argv[4])))
    else:
        raise SystemExit(main())
