"""Independent literal process-state spies; never send a signal or start a child."""
import errno
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ORACLE=HERE/'fixtures/posix_cleanup_oracles_v2.json'
ORACLE_SHA='a4acc005f785c84b48d3d56539edbbbed53fd31fc4cccb573e545fd62e00fc73'

def literals():
    raw=ORACLE.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==ORACLE_SHA
    return json.loads(raw)

def exercise(module,mode,root):
    expected=literals()[mode];events=[]
    state=SimpleNamespace(pid=424242,now=0.,exitcode=None,reaped=False,signals=0,launched=False)
    class Process:
        pid=424242
        def poll(self):raise AssertionError('reaping poll before signal is forbidden')
        def wait(self,timeout):
            assert state.signals==1 and not state.reaped,'first reap must follow the single group signal'
            events.append(['first_reap',timeout]);state.reaped=True
            if mode=='cleanup_late_zero':state.now=5.
            return state.exitcode
        def kill(self):
            assert state.signals==1 and not state.reaped
            events.append(['direct_kill',self.pid])
            if mode=='direct_kill_error':raise RuntimeError('controlled direct-kill failure')
            if state.exitcode is None:state.exitcode=-9
    process=Process()
    def popen(argv,**options):
        assert argv==['fixture'] and options['start_new_session'] is True
        assert events==[literals()['feature_event']]
        state.launched=True;events.append(['spawn',True]);return process
    def waitid(kind,pid,flags):
        assert kind==1 and flags==14 and not state.reaped
        if pid==999999:
            events.append(['feature',pid,flags])
            if mode=='unsupported_feature':raise OSError(errno.ENOSYS,'controlled unsupported waitid')
            raise ChildProcessError(errno.ECHILD,'self is not an owned child')
        assert pid==process.pid;events.append(['observe',pid,flags])
        if mode=='observation_error':raise OSError(errno.EIO,'controlled observation failure')
        state.now=5. if mode in ('timeout','late_zero','disappeared_group') else .25
        if mode in ('timeout','disappeared_group'):return None
        state.exitcode=0 if mode in ('late_zero','cleanup_late_zero') else 7
        return SimpleNamespace(si_pid=pid)
    def killpg(pid,sig):
        assert pid==process.pid and sig==9 and not state.reaped
        state.signals+=1;assert state.signals==1
        events.append(['killpg',pid,sig])
        if mode in ('permission_denied','direct_kill_error'):raise PermissionError(errno.EPERM,'controlled group denial')
        if mode=='disappeared_group':
            state.exitcode=0;raise ProcessLookupError(errno.ESRCH,'controlled missing group')
        if state.exitcode is None:state.exitcode=-9
    def callback(pid):
        events.append(['callback',pid]);raise RuntimeError('controlled launch callback failure')
    fake_os=SimpleNamespace(name='posix',getpid=lambda:999999,waitid=waitid,P_PID=1,WEXITED=2,WNOHANG=4,WNOWAIT=8,killpg=killpg)
    if mode=='missing_feature':del fake_os.WNOWAIT
    fake_subprocess=SimpleNamespace(Popen=popen,STDOUT=-2)
    outcome=None;error=None;log=Path(root)/'native-v2.log'
    with patch.object(module,'os',fake_os),patch.object(module,'signal',SimpleNamespace(SIGKILL=9)),\
            patch.object(module,'subprocess',fake_subprocess),\
            patch.object(module,'time',SimpleNamespace(monotonic=lambda:state.now,sleep=lambda _:None)):
        try:
            outcome=module.run_owned(['fixture'],cwd=str(root),environment={},logpath=log,timeout=5.,
                on_launch=callback if mode=='callback_error' else None)
        except BaseException as caught:error=caught
    assert events==expected['events'],(mode,events,expected['events'])
    if 'error' in expected:
        assert error is not None and type(error).__name__==expected['error'] and outcome is None
    else:
        assert error is None and outcome['native_exit']==expected['native_exit']
        assert outcome['timed_out'] is expected['timed_out'] and outcome['ownership']=='start_new_session=True'
    if expected.get('launched') is False:assert not state.launched and not log.exists()
    else:assert state.reaped and state.signals==1
    return outcome,error,events

def verify_windows_deadlines(module):
    cases=((.25,.25,0,False,False),(2.,2.,0,False,True),(1.,1.,0,False,True),
           (2.,2.,7,False,True),(.25,.25,57005,True,True),(.25,2.,0,False,True))
    for elapsed,native_wall,code,old_timeout,expected in cases:
        ticks=iter((0.,elapsed))
        raw=dict(pid=424242,native_exit=code,timed_out=old_timeout,wall_seconds=native_wall,
                 ownership='mock only',descendant_cleanup='mock only')
        with patch.object(module,'os',SimpleNamespace(name='nt')),\
                patch.object(module,'time',SimpleNamespace(monotonic=lambda:next(ticks))),\
                patch.object(module,'_run_windows',return_value=raw.copy()) as delegate:
            result=module.run_owned(['fixture'],cwd='unused',environment={},logpath='never-opened',timeout=1.)
        delegate.assert_called_once()
        assert result['timed_out'] is expected
        assert {k:v for k,v in result.items() if k!='timed_out'}=={k:v for k,v in raw.items() if k!='timed_out'}
