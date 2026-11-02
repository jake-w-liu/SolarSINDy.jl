"""Fixed toy processes for native ownership mechanics; never a Julia fallback."""
from pathlib import Path
import argparse,importlib.util,json,os,subprocess,sys,time
p=argparse.ArgumentParser();p.add_argument('--mode',choices=('emit','block','flood','descendant','pump-wrapper'),required=True)
p.add_argument('--root',type=Path,required=True);a=p.parse_args()
if a.mode=='emit':sys.stdout.buffer.write(b'native-byte-oracle\n');sys.stdout.buffer.flush()
elif a.mode=='block':time.sleep(3600)
elif a.mode=='flood':
    while True:sys.stdout.buffer.write(b'x'*65536);sys.stdout.buffer.flush()
elif a.mode=='descendant':
    child=subprocess.Popen([sys.executable,'-B','-X','utf8',__file__,'--mode','block','--root',str(a.root)],stdin=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    (a.root/'descendant-pid.json').write_text(json.dumps(dict(pid=child.pid))+'\n',encoding='utf8')
else:
    source=Path(__file__).resolve().parents[2]/'research/dbdt_completed_source/run_candidate_stage.py'
    spec=importlib.util.spec_from_file_location('owned_fixture_candidate_driver',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    result=m.pump_process([sys.executable,'-B','-X','utf8',__file__,'--mode','descendant','--root',str(a.root)],str(a.root),dict(os.environ),
        a.root/'inner.log',limit=128,timeout=5)
    m.write(a.root/'inner-wait.json',result)
    if result.get('reader_joined') is False:
        # This intentionally negative fixture is itself under the real owner;
        # exiting lets that boundary close the still-inherited pipe writer.
        os._exit(0)
