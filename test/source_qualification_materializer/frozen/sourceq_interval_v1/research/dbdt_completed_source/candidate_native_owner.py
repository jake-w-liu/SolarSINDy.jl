"""Versioned candidate owner: preserve Windows; observe POSIX exit without reaping.

Boundary cleanup is distinct from observing/reaping every descendant. Only the
direct leader is reaped here; per-descendant OS reaping is explicitly unobserved.
"""
from pathlib import Path
import hashlib,importlib.util,os,signal,subprocess,time
LEGACY=Path(__file__).with_name('native_stage.py')
LEGACY_SHA='30c7fc96042c3d84814f0ea728c0415d657d90c02b5de4cf516da517c7f9a21c'
def _posix_module():
    with LEGACY.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=LEGACY_SHA:raise RuntimeError('current native POSIX owner changed')
    spec=importlib.util.spec_from_file_location('candidate_checked_posix_owner',LEGACY)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def posix_features():
    return _posix_module()._posix_features()

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
    started=time.monotonic()
    old=_posix_module()
    result,proof=old._run_posix(argv,cwd,environment,logpath,timeout,on_launch,started=started)
    elapsed=time.monotonic()-started
    if proof['session_zero_live_verified'] is not True:
        raise RuntimeError('native private-session closure was not observed')
    result['timed_out']=result['timed_out'] or elapsed>=timeout or result['wall_seconds']>=timeout
    result['wall_seconds']=elapsed
    result.update(leader_reaped=True,ownership_boundary_closed=True,all_descendants_os_reaped=False,
        observation_method='terminal-code waitid with WNOWAIT; Darwin stop-only consumption; original-group signal; zero-live-SID census; then terminal wait',
        descendant_evidence='original-group SIGKILL '+proof['group_signal']+'; native private session zero live members observed while leader held; individual descendant reaping not observed')
    return result
