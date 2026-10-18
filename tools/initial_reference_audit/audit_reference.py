"""Run bounded independent artifact replay; no fitting or scientific verdict."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
from artifact_io import Limits,need,is_digest,plain_root,member,relative_parts,read_file,json_file,inventory
from prepare_artifact_audit import prepare,write_new
from owned_process import run_owned

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]


def now():return datetime.now(timezone.utc).isoformat()


def audit_reference(*,reference,reference_sha256,source,source_sha256,mode,output,julia,julia_sha256,timeout=1800):
    need(type(timeout) is int and 1<=timeout<=21600,'timeout must be an integer in 1:21600 seconds')
    need(all(is_digest(x) for x in (reference_sha256,source_sha256,julia_sha256)),'caller-pinned lowercase SHA256 digests required')
    reference=plain_root(reference);source=plain_root(source)
    julia=Path(julia)
    need(julia.is_absolute(),'explicit absolute Julia executable required')
    julia=member(plain_root(julia.parent),julia.name)
    need(read_file(julia,Limits().file_bytes)['sha256']==julia_sha256,'Julia executable digest differs')
    tools=plain_root(HERE)
    controls={str(p):read_file(member(tools,p.name),Limits().file_bytes)['sha256']
              for p in tools.iterdir() if p.suffix in ('.py','.jl')}
    for name in ('Project.toml','Manifest.toml'):
        path=member(plain_root(PROJECT),name)
        controls[str(path)]=read_file(path,Limits().file_bytes)['sha256']
    controls[str(julia)]=julia_sha256
    output=Path(output)
    need(output.is_absolute() and not output.exists(),'exclusive absolute audit output required')
    parent=plain_root(output.parent);relative_parts(output.name);output=parent/output.name
    for root in (reference,source,PROJECT):
        need(not output.is_relative_to(root) and not root.is_relative_to(output),'audit output overlaps input or code tree')
    output.mkdir()
    invocation={'schema':'independent_initial_reference_invocation_v1','complete':False,'started_utc':now(),
        'expected_mode':mode,'reference_completion_sha256':reference_sha256,'source_completion_sha256':source_sha256,
        'controls_sha256':controls,'timeout_seconds':timeout,'new_fitting':False,'author_or_ml_imported':False,
        'scientific_verdict':False,'python_version':sys.version}
    write_new(output/'invocation.json',json.dumps(invocation,allow_nan=False).encode())
    try:
        plan=prepare(reference,reference_sha256,source,source_sha256,mode,output/'prepared')
        plan_path=output/'prepared/audit-plan.json'
        plan_sha=read_file(plan_path,Limits().json_bytes)['sha256']
        argv=[str(julia),'--startup-file=no','--pkgimages=no','--threads=1','--depwarn=error','--project='+str(PROJECT),
              str(HERE/'audit_prepared_reference.jl'),str(plan_path),plan_sha]
        environment=dict(os.environ,JULIA_LOAD_PATH='@'+os.pathsep+'@stdlib',JULIA_PKG_OFFLINE='true',
            JULIA_PKG_PRECOMPILE_AUTO='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        native=run_owned(argv,cwd=str(output),environment=environment,logpath=output/'numerical.log',timeout=timeout)
        need(native['native_exit']==0 and not native['timed_out'],'independent Julia numerical replay failed')
        numerical_path=output/'prepared/numerical-audit.json'
        numerical_sha=read_file(numerical_path,Limits().json_bytes)['sha256']
        numerical=json_file(numerical_path,numerical_sha)
        need(numerical['complete'] is True and numerical['expected_mode']==mode and
             numerical['input_plan_sha256']==plan_sha and numerical['author_or_ml_imported'] is False and
             numerical['new_fitting'] is False and numerical['scientific_verdict'] is False,'numerical receipt scope/binding differs')
        for path,expected in {**plan['input_files_sha256'],**controls}.items():
            need(read_file(path,Limits().file_bytes)['sha256']==expected,'input or reader control changed during replay')
        need(read_file(plan_path,Limits().json_bytes)['sha256']==plan_sha,'prepared plan changed during replay')
        outputs=inventory(output)
        receipt={'schema':'independent_initial_reference_artifact_audit_v1','complete':True,'completed_utc':now(),
            'expected_mode':mode,'reference_completion_sha256':reference_sha256,'source_completion_sha256':source_sha256,
            'controls_sha256':controls,'output_sha256':outputs,'native':native,'numerical':numerical,
            'new_fitting':False,'author_or_ml_imported':False,'scientific_verdict':False,
            'execution_lineage_qualification':False,'serving_promotion':False}
        write_new(output/'completion.json',json.dumps(receipt,allow_nan=False,separators=(',',':')).encode())
        return receipt
    except BaseException as error:
        failure={'schema':'independent_initial_reference_artifact_audit_failure_v1','complete':False,
                 'completed_utc':now(),'type':type(error).__name__,'message':str(error),'scientific_verdict':False}
        write_new(output/'failure.json',json.dumps(failure,allow_nan=False).encode())
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','reference-sha256','source','source-sha256','output','julia','julia-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--mode',required=True,choices=('synthetic_only','exposed_development_reconstruction'))
    parser.add_argument('--timeout',type=int,default=1800)
    args=parser.parse_args()
    receipt=audit_reference(**vars(args))
    print(json.dumps({'complete':receipt['complete'],'expected_mode':receipt['expected_mode'],
                     'checks':receipt['numerical']['checks'],'scientific_verdict':False}))


if __name__=='__main__':main()
