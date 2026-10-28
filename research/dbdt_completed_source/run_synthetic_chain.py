"""Fixed pure-synthetic native route. No caller data, model, callback or operation injection."""
from pathlib import Path
import argparse,datetime as dt,hashlib,importlib.util,json,os,stat,struct,subprocess,sys,traceback
sys.dont_write_bytecode=True
if sys.flags.optimize:
 raise RuntimeError('optimized Python is unsupported for synthetic native verification')
def ordinary(path,directory=False):
 value=path.lstat()
 assert not path.is_symlink() and not (getattr(value,'st_file_attributes',0)&0x400),'reparse/symlink input rejected'
 assert stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode)
 return path
pkg=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--layout',choices=['ground11','mixed8'],required=True)
parser.add_argument('--julia',required=True)
parser.add_argument('--destination',required=True)
args=parser.parse_args()
assert sys.version_info>=(3,11),'Python>=3.11 is required'
pythonexe=Path(sys.executable).resolve()
assert pythonexe.is_absolute() and pythonexe.is_file() and not pythonexe.is_symlink()
exe=Path(args.julia)
assert exe.is_absolute(),'Julia must be an absolute ordinary executable'
ordinary(exe)
destination=Path(args.destination)
assert destination.is_absolute() and not destination.exists() and not destination.is_symlink()
ordinary(destination.parent,True)
destination=destination.parent.resolve()/destination.name
assert pkg.resolve() not in destination.parents and destination!=pkg.resolve(),'destination cannot be in the source checkout'
def canon(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode('utf-8')
def sha(data):return hashlib.sha256(data).hexdigest()
def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def save(path,value):
 with Path(path).open('xb') as f:f.write(canon(value))
 return file_sha(path)
checkpointpath=Path(__file__).with_name('synthetic_checkpoint.py')
checkpointspec=importlib.util.spec_from_file_location('fixed_synthetic_source_controls',checkpointpath)
checkpointmodule=importlib.util.module_from_spec(checkpointspec);checkpointspec.loader.exec_module(checkpointmodule)

def git(*argv):return checkpointmodule.git(pkg,*argv)
def utc():
 value=dt.datetime.now(dt.timezone.utc);delta=value-dt.datetime(1970,1,1,tzinfo=dt.timezone.utc)
 return str(delta.days*86400000+delta.seconds*1000+delta.microseconds//1000)
def frame(domain,chunks):
 h=hashlib.sha256(domain.encode()+b'\0'+struct.pack('>Q',len(chunks)))
 for chunk in chunks:h.update(struct.pack('>Q',len(chunk)));h.update(chunk)
 return h.hexdigest()

runner=Path(__file__).with_name('native_stage.py')
assert file_sha(runner)=='0b0f2327841230e5ac87d03bd67d1cbaf10717e0a52ce0134d71169a36146b89'
manifest=ordinary(pkg/'Manifest.toml')
if manifest.stat().st_size>checkpointmodule.MAX_CONTROL_BYTES:
 raise ValueError('Manifest exceeds bounded source size')
assert git('status','--porcelain','--untracked-files=normal')=='','native route requires a clean source checkout'
head=git('rev-parse','HEAD');tree=git('rev-parse','HEAD^{tree}');tracked=git('ls-files').splitlines()
assert 0<len(tracked)<=8192 and tracked==sorted(tracked)
source={}
checked_directories=set()
for name in tracked:
 path=pkg/name
 for parent in path.parents:
  if parent==pkg:break
  if parent not in checked_directories:ordinary(parent,True);checked_directories.add(parent)
 source[name]=file_sha(ordinary(path))
destination.mkdir(exist_ok=False)
factorypath=Path(__file__).with_name('synthetic_sources.py')
factoryspec=importlib.util.spec_from_file_location('fixed_synthetic_sources',factorypath)
factorymodule=importlib.util.module_from_spec(factoryspec);factoryspec.loader.exec_module(factorymodule)
factory=factorymodule.create_sources(pkg,destination,args.layout)
run=destination/'native';run.mkdir(exist_ok=False)
inventory=run/'source-inventory.json';inventorypin=save(inventory,[head,tree,[[name,source[name]] for name in tracked]])
protocolpath=Path(factory['protocol_path']);protocol=json.loads(protocolpath.read_text(encoding='utf-8'));assert file_sha(protocolpath)==factory['protocol_sha256']
pins={factory['protocol_sha256']:str(protocolpath),inventorypin:str(inventory)}
worker=pkg/'research/dbdt_completed_source/run_fit_job.jl';finalizer=pkg/'research/dbdt_completed_source/finalize_artifact.jl'
flags=[str(exe),'--startup-file=no','--compiled-modules=no','--pkgimages=no','--threads=1','--project='+str(pkg)]
environment=dict(os.environ,JULIA_LOAD_PATH=os.pathsep.join(('@','@stdlib')),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
spec=importlib.util.spec_from_file_location('fixed_owned_native_runner',runner);owned=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned)
candidates=sorted({p.resolve() for directory in (exe.parent,exe.parent.parent/'lib/julia',exe.parent.parent/'lib') if directory.is_dir() for p in directory.glob('*openblas*') if p.is_file()},key=str)
assert 0<len(candidates)<=32,'ordinary Julia OpenBLAS candidates are required'
inspection=run/'000-runtime-inspection';inspection.mkdir()
planned=inspection/'planned-argv.json'
inspection_argv=flags+[str(worker),'inspect-runtime',str(planned)]
save(planned,inspection_argv)
def inspection_guards():
 assert git('status','--porcelain','--untracked-files=normal')=='' and git('rev-parse','HEAD')==head and git('rev-parse','HEAD^{tree}')==tree
 result={'source/'+name:file_sha(pkg/name) for name in tracked}
 assert all(result['source/'+name]==pin for name,pin in source.items())
 result.update(protocol=file_sha(protocolpath),source_inventory=file_sha(inventory),runtime_executable=file_sha(exe),python_executable=file_sha(pythonexe),runner=file_sha(runner),planned_argv=file_sha(planned))
 for i,path in enumerate(candidates):result['blas_candidate/'+str(i)]=file_sha(path)
 for permission in protocol['source_permissions']:
  if permission['disposition']=='approved':
   body=Path(permission['path_or_null']);assert body.is_file() and not body.is_symlink() and file_sha(body)==permission['sha256_or_null']
   result['approved_synthetic_body/'+permission['body_id']]=file_sha(body)
 return result
inspection_before=inspection_guards()
inspection_native=owned.run_owned(inspection_argv,cwd=str(pkg),environment=environment,logpath=inspection/'native.log',timeout=120,
 on_launch=lambda pid:print(json.dumps({'event':'runtime_inspection','pid':pid}),flush=True))
inspection_after=inspection_guards()
inspection_receipt={'native':inspection_native,'argv':inspection_argv,'controls_before_sha256':inspection_before,'controls_after_sha256':inspection_after,'log_sha256':file_sha(inspection/'native.log')}
save(inspection/'OWNED_NATIVE_RAW.json',inspection_receipt)
assert inspection_native['native_exit']==0 and not inspection_native['timed_out'] and inspection_before==inspection_after
runtime_base=json.loads((inspection/'native.log').read_text(encoding='utf-8'))
assert runtime_base['julia_version'] in ('1.12.6','1.13.1') and runtime_base['executable_sha256']==file_sha(exe)
assert (runtime_base['architecture'].endswith('/NT'))==(os.name=='nt'),'Python and Julia must use the same host OS ownership contract'
blas=next(path for path in candidates if file_sha(path)==runtime_base['blas_library_sha256'])
stages=[];counter=0
def pin_bundle(path):
 commit=json.loads((path/'COMMIT.json').read_text(encoding='utf-8'))
 assert file_sha(path/'manifest.json')==commit['manifest_sha256']
 pins[commit['manifest_sha256']]=str(path);pins[file_sha(path/'COMMIT.json')]=str(path)
 return commit['manifest_sha256']
def pinfile(path):pins[file_sha(path)]=str(path);return file_sha(path)
def resolver(path):save(path,[[pin,pins[pin]] for pin in sorted(pins)])
def guards(request=None,resolverpath=None):
 assert git('status','--porcelain','--untracked-files=normal')=='' and git('rev-parse','HEAD')==head and git('rev-parse','HEAD^{tree}')==tree
 result={'source/'+name:file_sha(pkg/name) for name in tracked}
 assert all(result['source/'+name]==pin for name,pin in source.items())
 result.update(protocol=file_sha(protocolpath),source_inventory=file_sha(inventory),runtime_executable=file_sha(exe),runtime_blas=file_sha(blas),runner=file_sha(runner),controller=file_sha(__file__),python_executable=file_sha(pythonexe))
 for permission in protocol['source_permissions']:
  if permission['disposition']=='approved':
   body=Path(permission['path_or_null']);assert body.is_absolute() and body.is_file() and not body.is_symlink()
   bodypin=file_sha(body);assert bodypin==permission['sha256_or_null'] and body.stat().st_size==permission['bytes_or_null']
   result['approved_synthetic_body/'+permission['body_id']]=bodypin

 unique=sorted(set(pins.values()))
 for index,text in enumerate(unique):
  path=Path(text);assert not path.is_symlink()
  if path.is_dir():
   names=sorted(path.iterdir());assert len(names)<=256 and all(p.is_file() and not p.is_symlink() for p in names)
   for member in names:result[f'dependency/{index:06d}/{member.name}']=file_sha(member)
  else:result[f'external/{index:06d}']=file_sha(path)
 if request is not None:
  result['request']=file_sha(request)
  value=json.loads(request.read_text(encoding='utf-8'))
  if value['input_manifest_pin_or_null'] is not None:result['input']=value['input_manifest_pin_or_null']
 if resolverpath is not None:result['resolver']=file_sha(resolverpath)
 return result
def run_owned(argv,directory,log,timeout,request=None,resolverpath=None):
 before=guards(request,resolverpath);started=utc()
 raw=owned.run_owned(argv,cwd=str(pkg),environment=environment,logpath=log,timeout=timeout,
     on_launch=lambda pid:print(json.dumps({'event':'native_launch','pid':pid,'stage':directory.name,'argv0':str(exe),'source_head':head}),flush=True))
 finished=utc();after=guards(request,resolverpath)
 result={'argv':argv,'started_utc':started,'finished_utc':finished,'native':raw,'controls_before_sha256':before,'controls_after_sha256':after,'log_sha256':file_sha(log)}
 save(directory/'OWNED_NATIVE_RAW.json',result)
 assert before==after,'guarded bytes changed'
 assert raw['native_exit']==0 and not raw['timed_out'],f'native failure retained: {log}'
 return result
def finalize(action,name,extra):
 global counter
 counter+=1;directory=run/f'{counter:03d}-publish-{name}';directory.mkdir();destination=directory/'artifact'
 resolution=directory/'pins.json';resolver(resolution)
 argv=flags+[str(finalizer),action,str(protocolpath),str(resolution),str(destination)]+list(map(str,extra))
 run_owned(argv,directory,directory/'native.log',600,resolverpath=resolution)
 pin=pin_bundle(destination);stages.append({'kind':action,'directory':str(destination),'manifest_sha256':pin})
 print(json.dumps({'event':'committed_synthetic','kind':action,'manifest_sha256':pin}),flush=True)
 return destination,pin
def job(operation,name,inputpin,datapin,dependencies=None,recipe=None):
 global counter
 counter+=1;directory=run/f'{counter:03d}-job-{name}';directory.mkdir();work=directory/'work';work.mkdir()
 requestpath=directory/'request.json';resolution=directory/'pins.json'
 argv=flags+[str(worker),str(protocolpath),str(requestpath),str(resolution),str(inventory),str(work)]
 runtime=dict(runtime_base,argv=argv)
 request={'schema':'dbdt_completed_job_request_v1','job_id':'synthetic-'+args.layout+'-'+name,'operation':operation,'evidence_mode':'synthetic_only','protocol_pin':factory['protocol_sha256'],'input_manifest_pin_or_null':inputpin,'input_data_pin':datapin,'recipe_pin_or_null':recipe,'source_permission_pin_or_null':factory['source_permission_sha256'] if operation in ('source_audit','source_independent_audit') else None,'required_runtime':runtime,'dependency_pins':[{'role':k,'manifest_sha256':v} for k,v in sorted((dependencies or {}).items())],'work_directory':str(work)}
 pinfile_request=save(requestpath,request);pins[pinfile_request]=str(requestpath);resolver(resolution)
 result=run_owned(argv,directory,directory/'native.log',600,request=requestpath,resolverpath=resolution)
 native=result['native'];internal=json.loads((work/'internal-receipt.json').read_text(encoding='utf-8'))
 assert json.loads((work/'runtime.json').read_text(encoding='utf-8'))==runtime
 payloads=sorted(work.glob('payload*.json'));chunks=[p.read_bytes() for p in payloads]
 payloadpin=sha(chunks[0]) if len(chunks)==1 else frame('DBDTPAYLOAD1',chunks)
 assert internal['payload_sha256']==payloadpin
 external={'schema':'dbdt_owned_native_receipt_v1','operation':operation,'protocol_sha256':factory['protocol_sha256'],'job_id':request['job_id'],'runtime':runtime,'argv_sha256':sha(canon(argv)),'started_utc':result['started_utc'],'finished_utc':result['finished_utc'],'native_exit':native['native_exit'],'timed_out':native['timed_out'],'ownership':native['ownership'],'descendant_cleanup':native['descendant_cleanup'],'controls_before_sha256':[{'name':k,'sha256':v} for k,v in sorted(result['controls_before_sha256'].items())],'controls_after_sha256':[{'name':k,'sha256':v} for k,v in sorted(result['controls_after_sha256'].items())],'input_manifest_sha256_or_null':inputpin,'input_data_sha256':datapin,'payload_sha256':payloadpin,'internal_receipt_sha256':file_sha(work/'internal-receipt.json'),'log_sha256':file_sha(directory/'native.log'),'runner_sha256':file_sha(runner)}
 externalpath=directory/'NATIVE_OWNERSHIP_RECEIPT.json';save(externalpath,external);pinfile(externalpath)
 bundle,pin=finalize('native',name,[requestpath,work,externalpath,directory/'native.log'])
 data=json.loads(chunks[0]);metadata=data.get('metadata') if isinstance(data,dict) else None
 return bundle,pin,metadata
terminal={'schema':'synthetic_chain_development_terminal_v1','complete':False,'source_head':head,'source_tree':tree,'layout':args.layout,'runtime':runtime_base['julia_version'],'scope':'synthetic development chain; no actual-source or full qualification claim','source_bound_provenance_verified':False,'controller_sha256':file_sha(__file__),'runtime_inspection_raw_sha256':file_sha(inspection/'OWNED_NATIVE_RAW.json'),'python_executable_sha256':file_sha(pythonexe)}
try:
 primary,primarypin,metadata=job('source_audit','source-primary',None,factory['source_permission_sha256'])
 checker,checkerpin,_=job('source_independent_audit','source-check',primarypin,metadata['source_audit_data_sha256'])
 sourcepath,sourcepin=finalize('source','source',[primary,checker])
 feature_request=run/'feature-request.json';save(feature_request,{'schema':'dbdt_completed_feature_request_v1','ordinal':1,'chunks':1,'metadata':{'source_manifest_sha256':sourcepin},'rows':factory['issues']});featurepin=pinfile(feature_request)
 primary,primarypin,metadata=job('feature_schedule','schedule-primary',sourcepin,featurepin)
 checker,_,_=job('feature_schedule','schedule-check',primarypin,metadata['schedule_data_sha256'])
 schedulepath,schedulepin=finalize('schedule','schedule',[primary,checker])
 inputs={}
 for split in ('fit','october'):
  request=run/('split-'+split+'.json');save(request,{'schema':'dbdt_completed_split_request_v1','schedule_manifest_sha256':schedulepin,'split':split});requestpin=pinfile(request)
  primary,primarypin,metadata=job('input_audit',split+'-input-primary',schedulepin,requestpin)
  checker,_,_=job('input_independent_audit',split+'-input-check',primarypin,metadata['input_data_sha256'])
  inputs[split]=(*finalize('input',split+'-input',[primary,checker]),metadata['input_data_sha256'])
 primary,primarypin,_=job('fit_ridge','ridge-primary',inputs['fit'][1],inputs['fit'][2],recipe=factory['recipe_sha256'])
 repeated,repeatedpin,_=job('fit_ridge','ridge-repeat',inputs['fit'][1],inputs['fit'][2],recipe=factory['recipe_sha256'])
 assert (repeated/'payload.json').read_bytes()==(primary/'payload.json').read_bytes(),'identical-runtime actual ridge payload did not reproduce'
 terminal['deterministic_fit_payload_sha256']=file_sha(primary/'payload.json')

 checker,_,_=job('fit_independent_check','ridge-check',inputs['fit'][1],inputs['fit'][2],{'producer':primarypin},recipe=factory['recipe_sha256'])
 pointpath,pointpin=finalize('point','point',[primary,checker])
 primary,primarypin,_=job('calibrate','cal-primary',inputs['october'][1],inputs['october'][2],{'point':pointpin})
 checker,_,_=job('cal_independent_check','cal-check',inputs['october'][1],inputs['october'][2],{'point':pointpin,'producer':primarypin})
 calpath,calpin=finalize('calibration','calibration',[primary,checker])
 loadpath,loadpin,_=job('load_cal_check','cal-load',calpin,inputs['october'][2])
 terminal.update(complete=True,point_manifest_sha256=pointpin,calibration_manifest_sha256=calpin,outer_load_check_manifest_sha256=loadpin)
except BaseException as error:
 terminal.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
 terminal['stages']=stages;terminal['final_source_head']=git('rev-parse','HEAD');terminal['final_clean']=git('status','--porcelain','--untracked-files=normal')==''
 save(run/'TERMINAL.json',terminal)
 print(json.dumps(terminal,indent=2),flush=True)
