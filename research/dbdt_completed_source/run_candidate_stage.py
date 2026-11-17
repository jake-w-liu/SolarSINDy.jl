"""Run one pinned completed-source operation with owned timeout and cleanup.

Request schema completed_source_candidate_request_v2 contains the ordinary case
artifact pins, operation, output directory and fixed candidate inputs. Julia
performs source/target/model validation. No CI evidence or permission files.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,shutil,sys,time
ROOT=Path(__file__).resolve().parents[2]
ENTRY=Path(__file__).with_suffix('.jl')
SECONDS=7200
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
def main(required_operation=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--request-sha256',required=True)
    parser.add_argument('--julia',default='julia')
    args=parser.parse_args()
    with args.request.open('rb') as stream:raw=stream.read(1048577)
    if len(raw)>1048576 or hashlib.sha256(raw).hexdigest()!=args.request_sha256:
        raise ValueError('candidate request pin/size differs')
    request=json.loads(raw);root=Path(request['execution_directory'])
    if required_operation is not None and request.get('operation')!=required_operation:
        raise ValueError('this entry requires operation='+required_operation)
    if request['schema']!='completed_source_candidate_request_v2' or not root.is_absolute() or not root.is_dir():
        raise ValueError('ordinary candidate request/output directory required')
    log=root/'julia.log';result_path=root/'PROCESS_RESULT.json'
    if any(p.exists() or p.is_symlink() for p in (log,result_path,root/'JULIA_STAGE_RESULT.json')):
        raise ValueError('experiment output already exists')
    # Reuse the platform-native resource observation and existing process owner.
    memory,method=load_module(ROOT/'tools/run_canonical_harness.py','candidate_memory').available_memory()
    disk=shutil.disk_usage(root).free
    if memory<8*1024**3:raise ValueError('8GiB available physical memory required before Julia')
    executable=shutil.which(args.julia)
    if executable is None:raise FileNotFoundError(args.julia)
    argv=[executable,'--startup-file=no','--threads=1','--project='+str(ROOT),str(ENTRY),
        '--request',str(args.request.resolve()),'--request-sha256',args.request_sha256]
    environment=os.environ.copy()
    for key in ('JULIA_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        environment[key]='1'
    owner=load_module(ENTRY.with_name('candidate_native_owner.py'),'candidate_process_owner')
    started=time.monotonic()
    result=owner.run_owned(argv,cwd=ROOT,environment=environment,logpath=log,timeout=SECONDS)
    result.update(schema='completed_source_process_result_v2',request_sha256=args.request_sha256,
        elapsed_seconds=time.monotonic()-started,available_physical_bytes=memory,memory_method=method,free_disk_bytes=disk,
        julia_threads=1,blas_threads=1,timeout_seconds=SECONDS)
    with result_path.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,sort_keys=True,allow_nan=False);stream.write('\n')
    if args.request.read_bytes()!=raw:raise ValueError('candidate request changed during execution')
    return 0 if result['native_exit']==0 and not result['timed_out'] else 1
if __name__=='__main__':raise SystemExit(main())
