
import os,subprocess,sys,threading,time
if sys.flags.optimize:raise RuntimeError("optimized Python capture is unsupported")
limit=8388608
overflow=threading.Event()
failed=threading.Event()
def report_failure(stage,error):
    failed.set()
    try:
        text=("capture "+stage+" failure: "+type(error).__name__+": "+str(error)+"\n").encode("utf-8",errors="replace")[:1024]
        os.write(2,text)
    except BaseException:pass
def write_exact(saved,data):
    if saved.write(data)!=len(data):raise OSError("short control-output write")
def drain(pipe,path):
    try:
        remaining=limit
        with open(path,"xb",buffering=0) as saved:
            while True:
                chunk=pipe.read(min(65536,remaining+1))
                if not chunk:break
                if len(chunk)>remaining:
                    write_exact(saved,chunk[:remaining]);overflow.set();break
                write_exact(saved,chunk);remaining-=len(chunk)
            saved.flush()
    except BaseException as error:report_failure("stream",error)
    finally:
        try:pipe.close()
        except BaseException as error:report_failure("reader close",error)
def main():
    process=subprocess.Popen(sys.argv[3:],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    threads=[threading.Thread(target=drain,args=(process.stdout,sys.argv[1]),daemon=True),
             threading.Thread(target=drain,args=(process.stderr,sys.argv[2]),daemon=True)]
    for thread in threads:thread.start()
    while process.poll() is None:
        if failed.is_set():os._exit(88)
        if overflow.is_set():os._exit(86)
        time.sleep(0.01)
    for thread in threads:thread.join(5)
    if failed.is_set():os._exit(88)
    if overflow.is_set():os._exit(86)
    if any(thread.is_alive() for thread in threads):os._exit(87)
    return process.returncode
try:result=main()
except BaseException as error:
    report_failure("main",error);os._exit(88)
sys.exit(result)
