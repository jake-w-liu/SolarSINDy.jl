"""Versioned candidate owner: preserve Windows; observe POSIX exit without reaping.

Boundary cleanup is distinct from observing/reaping every descendant. Only the
direct leader is reaped here; per-descendant OS reaping is explicitly unobserved.
"""
from pathlib import Path
import hashlib,importlib.util,os,signal,subprocess,time
LEGACY=Path(__file__).with_name('native_stage.py')
LEGACY_SHA='0b0f2327841230e5ac87d03bd67d1cbaf10717e0a52ce0134d71169a36146b89'
def posix_features():
    if os.name!='posix' or not all(hasattr(os,k) for k in ('waitid','P_PID','WEXITED','WNOHANG','WNOWAIT')):
        raise RuntimeError('nonreaping POSIX waitid/WNOWAIT is required; no unsafe fallback')
    try:os.waitid(os.P_PID,os.getpid(),os.WEXITED|os.WNOHANG|os.WNOWAIT)
    except ChildProcessError:pass  # Valid operation; this process is not its own child.
    except OSError as error:raise RuntimeError('nonreaping waitid feature unavailable before launch') from error
def run_owned(argv,*,cwd,environment,logpath,timeout,on_launch=None):
    if os.name=='nt':
        with LEGACY.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=LEGACY_SHA:raise RuntimeError('preserved Windows native owner changed')
        spec=importlib.util.spec_from_file_location('candidate_preserved_windows_owner',LEGACY)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        started=time.monotonic()
        result=old.run_owned(argv,cwd=cwd,environment=environment,logpath=logpath,timeout=timeout,on_launch=on_launch)
        elapsed=time.monotonic()-started
        result['timed_out']=result['timed_out'] or elapsed>=timeout or result.get('wall_seconds',elapsed)>=timeout
        result['wrapper_observed_wall_seconds']=elapsed
        result.update(leader_reaped=True,ownership_boundary_closed=True,all_descendants_os_reaped=False,
            descendant_evidence='private kill-on-close Job boundary closed; individual descendant exit/reaping not observed')
        return result
    posix_features();started=time.monotonic();expired=False;native=None;process=None
    with Path(logpath).open('xb',buffering=0) as log:
        process=subprocess.Popen(argv,cwd=cwd,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            if on_launch:on_launch(process.pid)
            while True:
                observed=os.waitid(os.P_PID,process.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
                expired=time.monotonic()-started>=timeout
                if observed is not None or expired:break
                time.sleep(min(.02,max(0.,timeout-(time.monotonic()-started))))
        finally:
            signal_error=None
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            except BaseException as error:signal_error=error;process.kill()
            # This is the FIRST reaping operation. The group signal always
            # precedes it, including exceptions from the launch callback/waitid.
            native=process.wait(timeout=20)
            if signal_error is not None:raise signal_error
    return dict(pid=process.pid,native_exit=native,timed_out=expired,leader_reaped=True,
        ownership='start_new_session=True',descendant_cleanup='owned POSIX session killed after stage',
        ownership_boundary_closed=True,all_descendants_os_reaped=False,wall_seconds=time.monotonic()-started,
        observation_method='waitid(WEXITED|WNOHANG|WNOWAIT) before killpg, then wait',
        descendant_evidence='SIGKILL applied to private group before leader reaping; individual descendant reaping not observed')
