"""Bounded independent controls and native generated metadata reconstruction."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import argparse, importlib.util, json, os, subprocess, sys, time, traceback

ROOT=Path(__file__).resolve().parents[2]
def pin(path):return sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    with Path(path).open('xb') as stream:stream.write((json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--report-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('separate_evaluation_metadata_oracle',Path(__file__).with_name('evaluation_binding.py'))
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    report_path=oracle._ordinary(args.report.resolve(strict=True));output=args.output.resolve()
    oracle.need(not output.exists() and output.parent.is_dir() and not output.is_relative_to(ROOT),'independent output path differs')
    body=oracle.V.bounded_read(report_path,131072)
    oracle.need(sha256(body).hexdigest()==args.report_sha256,'external report pin differs')
    report=oracle.V.decoded(body)
    oracle._keys(report,('schema','runtime','entries','actual_source_access','actual_fits'))
    oracle.need(report['schema']=='generated_separate_evaluation_contexts_v1' and report['runtime'] in ('1.12.6','1.13.1') and
        report['actual_source_access'] is False and type(report['actual_fits']) is int and report['actual_fits']==0 and
        isinstance(report['entries'],list) and len(report['entries'])==4,'generated evaluation report differs')
    controls={str(p):pin(p) for p in (Path(__file__),report_path,*[ROOT/name for name in oracle.FILES],
        Path(__file__).with_name('evaluation_binding.py'),Path(__file__).with_name('bounded_json.py'),
        ROOT/'test/point_input_export/test_evaluation_binding.py')}
    output.mkdir();record=dict(complete=False,started_utc=now(),controls_sha256=controls,checks=[],actual_source_access=False,actual_fits=0)
    write(output/'PREEXECUTION.json',record)
    try:
        log=output/'controls.log';started=time.monotonic()
        with log.open('xb') as stream:
            # Fixed stdlib-only child inherits the surrounding native owner's
            # process group/Windows Job and is killed and waited on at timeout.
            child=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'test/point_input_export/test_evaluation_binding.py')],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=900,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        record['controls_native']=dict(native_exit=child.returncode,timed_out=False,wall_milliseconds=int((time.monotonic()-started)*1000),raw_sha256=pin(log))
        oracle.need(child.returncode==0,'independent evaluation controls failed')
        seen=set()
        for entry in report['entries']:
            oracle._keys(entry,('directory','pins','table_index','start','stop','asof','role','layout'))
            key=entry['role'],entry['layout'];oracle.need(key not in seen and key in {(r,l) for r in ('primary','independent') for l in ('ground-only','mixed')},'evaluation context inventory differs');seen.add(key)
            checked=oracle.verify_binding(entry['directory'],ROOT,**{k:entry[k] for k in ('pins','table_index','start','stop','asof')})
            context=oracle.V.decoded(oracle.V.bounded_read(Path(entry['directory'])/'evaluation-binding.json',131072))
            oracle.need((context['role'],context['layout'],context['runtime'])==(*key,report['runtime']),'native evaluation context differs')
            record['checks'].append(dict(role=key[0],layout=key[1],verification=checked))
        record.update(complete=len(seen)==4,contexts=len(seen))
    except BaseException as error:
        record['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        record['changed_controls']=[p for p,value in controls.items() if pin(p)!=value]
        record['complete']=record['complete'] and not record['changed_controls'];record['finished_utc']=now()
        write(output/'terminal.json',record)
        print(json.dumps({key:value for key,value in record.items() if key not in ('controls_sha256','checks')}),flush=True)
    return 0 if record['complete'] else 1

if __name__=='__main__':raise SystemExit(main())
