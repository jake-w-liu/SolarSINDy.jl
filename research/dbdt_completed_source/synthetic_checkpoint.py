"""Bounded owned Git capture shared by the standalone synthetic research driver."""
from pathlib import Path
import hashlib,importlib.util,json,os,stat,subprocess,sys,tempfile
sys.dont_write_bytecode=True
if sys.flags.optimize:
    raise RuntimeError('optimized Python is unsupported for synthetic source verification')
MAX_CONTROL_BYTES=8388608
_CAPTURE_ENTRY=Path(__file__).with_name('capture_control_output.py')
_CAPTURE_SHA256='9d3a578bbd707969982feee22e5143984625a655d638f37b32a347abf2c76f7e'

def bounded_file(path,limit=MAX_CONTROL_BYTES):
    path=ordinary(Path(path))
    if path.stat().st_size>limit:raise ValueError('control file exceeds bounded read limit')
    with path.open('rb') as stream:data=stream.read(limit+1)
    if len(data)>limit:raise ValueError('control file grew beyond bounded read limit')
    return data

def captured(arguments,*,timeout=120,environment=None):
    # A fixed capture child drains both streams concurrently with bounded
    # chunks. The unchanged owner kills its entire child tree/session on every
    # exit, including overflow and timeout; no unbounded communicate fallback.
    directory=Path(tempfile.mkdtemp(prefix='ss-r1-git-capture-')).resolve()
    capture=ordinary(_CAPTURE_ENTRY)
    if hashlib.sha256(bounded_file(capture)).hexdigest()!=_CAPTURE_SHA256:
        raise RuntimeError('fixed capture entry bytes differ')
    runner=Path(__file__).with_name('native_stage.py')
    if file_sha(runner)!='30c7fc96042c3d84814f0ea728c0415d657d90c02b5de4cf516da517c7f9a21c':
        raise RuntimeError('owned Git capture runner bytes differ')
    spec=importlib.util.spec_from_file_location('fixed_git_capture_owner',runner)
    owned=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned)
    argv=[str(Path(sys.executable).resolve()),'-B','-X','utf8',str(capture),str(directory/'stdout.log'),str(directory/'stderr.log'),*map(str,arguments)]
    result=owned.run_owned(argv,cwd=str(directory),environment=environment or dict(os.environ),logpath=directory/'capture.log',timeout=timeout)
    (directory/'OWNED_NATIVE_RAW.json').write_text(json.dumps(result,sort_keys=True),encoding='utf-8')
    if hashlib.sha256(bounded_file(ordinary(capture))).hexdigest()!=_CAPTURE_SHA256:
        raise RuntimeError('fixed capture entry bytes differ after native return; preserved owner receipt: '+str(directory))
    if result['timed_out'] or result['native_exit'] in (86,87,88):
        raise RuntimeError('bounded Git capture failed; preserved output prefixes and owner receipt: '+str(directory))
    stdout=bounded_file(directory/'stdout.log');stderr=bounded_file(directory/'stderr.log')
    return subprocess.CompletedProcess(arguments,result['native_exit'],stdout,stderr)

def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()

def ordinary(path,directory=False):
    path=Path(path);value=path.lstat()
    assert not path.is_symlink() and not (getattr(value,'st_file_attributes',0)&0x400),'reparse/symlink input rejected'
    assert stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode)
    return path

def git(path,*arguments,strip_output=True):
    result=captured(['git','-C',str(path),*arguments],
        environment=dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_TERMINAL_PROMPT='0'))
    if result.returncode:raise RuntimeError('Git control command failed: '+repr(arguments))
    text=result.stdout.decode('utf-8')
    return text.strip() if strip_output else text
