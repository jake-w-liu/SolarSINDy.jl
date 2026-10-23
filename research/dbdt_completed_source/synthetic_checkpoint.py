"""Explicit private synthetic test descendant; never changes the original checkout."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,runpy,stat,subprocess,sys,tempfile
sys.dont_write_bytecode=True
if sys.flags.optimize:
    raise RuntimeError('optimized Python is unsupported for synthetic source verification')
MAX_CONTROL_BYTES=8388608
_CAPTURE_CODE=r'''
import os,subprocess,sys,threading,time
if sys.flags.optimize:raise RuntimeError("optimized Python capture is unsupported")
limit=8388608
process=subprocess.Popen(sys.argv[3:],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
overflow=threading.Event()
def drain(pipe,path):
    remaining=limit
    with open(path,"xb",buffering=0) as saved:
        while True:
            chunk=pipe.read(min(65536,remaining+1))
            if not chunk:break
            if len(chunk)>remaining:
                saved.write(chunk[:remaining]);overflow.set();break
            saved.write(chunk);remaining-=len(chunk)
    pipe.close()
threads=[threading.Thread(target=drain,args=(process.stdout,sys.argv[1]),daemon=True),
         threading.Thread(target=drain,args=(process.stderr,sys.argv[2]),daemon=True)]
for thread in threads:thread.start()
while process.poll() is None:
    if overflow.is_set():os._exit(86)
    time.sleep(0.01)
for thread in threads:thread.join(5)
if overflow.is_set():os._exit(86)
if any(thread.is_alive() for thread in threads):os._exit(87)
sys.exit(process.returncode)
'''

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
    runner=Path(__file__).with_name('native_stage.py')
    if file_sha(runner)!='1c593d90793f6f70e05622772aec83b53919aa2c22575ab738a4ee1968f116f0':
        raise RuntimeError('owned Git capture runner bytes differ')
    spec=importlib.util.spec_from_file_location('fixed_git_capture_owner',runner)
    owned=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned)
    argv=[str(Path(sys.executable).resolve()),'-B','-c',_CAPTURE_CODE,str(directory/'stdout.log'),str(directory/'stderr.log'),*map(str,arguments)]
    result=owned.run_owned(argv,cwd=str(directory),environment=environment or dict(os.environ),logpath=directory/'capture.log',timeout=timeout)
    (directory/'OWNED_NATIVE_RAW.json').write_text(json.dumps(result,sort_keys=True),encoding='utf-8')
    if result['timed_out'] or result['native_exit'] in (86,87):
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

def canonical(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode('utf-8')

def state(source):
    ordinary(source,True)
    manifest=ordinary(source/'Manifest.toml')
    if manifest.stat().st_size>MAX_CONTROL_BYTES:raise ValueError('Manifest exceeds bounded source size')
    names=git(source,'ls-files').splitlines()
    assert 0<len(names)<=8192 and names==sorted(names) and len(set(names))==len(names)
    raw={}
    for name in names:
        assert not Path(name).is_absolute() and '..' not in Path(name).parts and '\n' not in name and '\r' not in name
        path=source
        for part in Path(name).parts[:-1]:path=ordinary(path/part,True)
        raw[name]=file_sha(ordinary(source/name))
    assert 'Project.toml' in raw and 'Manifest.toml' in raw
    others=git(source,'ls-files','--others','--exclude-standard').splitlines()
    assert not others,'untracked unignored input rejected'
    changed=set(git(source,'diff','HEAD','--name-only').splitlines())
    changed.update(git(source,'diff','--cached','HEAD','--name-only').splitlines())
    changed.update(git(source,'diff','--name-only').splitlines())
    assert changed<= {'Manifest.toml'},'only an explicit tracked Manifest delta is allowed'
    return {'head':git(source,'rev-parse','HEAD'),'tree':git(source,'rev-parse','HEAD^{tree}'),
        'files_sha256':raw,'changed_paths':sorted(changed),'manifest_diff':git(source,'diff','HEAD','--','Manifest.toml',strip_output=False),
        'status':git(source,'status','--porcelain','--untracked-files=normal')}

def bounded_projection(proof):
    result=dict(proof)
    for key in ('source_before','source_after','child'):
        value=dict(proof[key])
        value['files_sha256']=[[name,pin] for name,pin in sorted(value['files_sha256'].items())]
        value.pop('manifest_diff')
        result[key]=value
    result.pop('exact_manifest_diff')
    return result

def prepare_snapshot(source,destination):
    source=Path(source);destination=Path(destination)
    assert source.is_absolute() and destination.is_absolute()
    ordinary(source,True);ordinary(destination.parent,True)
    assert not destination.exists() and not destination.is_symlink() and source.resolve() not in destination.resolve().parents
    before=state(source)
    cloned=captured(['git','clone','--no-local','--no-checkout',str(source),str(destination)],
        environment=dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_TERMINAL_PROMPT='0'))
    if cloned.returncode:raise RuntimeError('explicit source clone failed')
    assert not (destination/'.git/objects/info/alternates').exists(),'shared mutable Git objects rejected'
    for option in ('core.autocrlf','core.eol'):
        result=captured(['git','-C',str(source),'config','--get',option],timeout=30)
        assert result.returncode in (0,1)
        if result.returncode==0:git(destination,'config','--local',option,result.stdout.decode().strip())
    git(destination,'checkout','--detach',before['head'])
    for name,pin in before['files_sha256'].items():
        if name!='Manifest.toml':assert file_sha(ordinary(destination/name))==pin,'non-Manifest checkout byte differs'
    original_manifest=destination/'Manifest.toml'
    parent_manifest_sha256=file_sha(ordinary(original_manifest))
    actual_manifest=bounded_file(source/'Manifest.toml')
    original_manifest.write_bytes(actual_manifest)
    assert set(git(destination,'diff','--name-only').splitlines())<= {'Manifest.toml'}
    git(destination,'config','--local','user.name','SolarSINDy synthetic tests')
    git(destination,'config','--local','user.email','synthetic-tests@invalid')
    # A copied local origin is never a publication target. Keep the owned test
    # descendant associated with the same repository; no network/push is used.
    remote=captured(['git','-C',str(source),'remote','get-url','origin'],timeout=30)
    if remote.returncode==0:
        url=remote.stdout.decode().strip()
        assert 'github.com' in url and '/PT' not in url and '@' not in url.removeprefix('git@github.com:')
        git(destination,'remote','set-url','origin',url)
    if parent_manifest_sha256!=file_sha(original_manifest):
        git(destination,'add','--','Manifest.toml')
        git(destination,'commit','-m','Record the resolved dependency lock for synthetic tests')
    assert git(destination,'status','--porcelain','--untracked-files=normal')==''
    child=state(destination);after=state(source)
    git(destination,'merge-base','--is-ancestor',before['head'],child['head'])
    assert before==after,'original checkout changed during snapshot'
    assert before['files_sha256']==child['files_sha256'],'test descendant does not preserve actual tested bytes'
    return {'schema':'explicit_synthetic_test_source_projection_v1','source_directory':str(source),'child_directory':str(destination),
        'parent_head':before['head'],'parent_tree':before['tree'],'child_head':child['head'],'child_tree':child['tree'],
        'parent_manifest_sha256':parent_manifest_sha256,'actual_resolved_manifest_sha256':file_sha(original_manifest),
        'exact_manifest_diff':before['manifest_diff'],'source_before':before,'source_after':after,'child':child,
        'original_checkout_unchanged':True,'child_checkpoint_truthful':True,'source_bound_provenance_verified':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--julia',required=True);parser.add_argument('--destination',required=True)
    args=parser.parse_args();assert sys.version_info>=(3,11)
    package=Path(__file__).resolve().parents[2];output=Path(args.destination)
    assert output.is_absolute() and not output.exists() and not output.is_symlink()
    ordinary(output.parent,True);assert package not in output.resolve().parents
    output.mkdir(exist_ok=False)
    proof=prepare_snapshot(package,output/'source')
    # The exact diff remains a separate bounded ordinary text file. The small
    # canonical projection uses the existing bounded positional byte-map form.
    diff=proof['exact_manifest_diff'].encode('utf-8')
    assert len(diff)<=8388608
    (output/'Manifest.diff').write_bytes(diff)
    projected=bounded_projection(proof)
    projected['manifest_diff_sha256']=hashlib.sha256(diff).hexdigest()
    (output/'SOURCE_PROJECTION.json').write_bytes(canonical(projected))
    routes={}
    for layout in ('ground11','mixed8'):
        target=output/layout
        entry=output/'source/research/dbdt_completed_source/run_synthetic_chain.py'
        argv=[str(entry),
            '--julia',args.julia,'--layout',layout,'--destination',str(target)]
        # Execute only the byte-verified fixed entry in this controller. Every
        # numerical Julia invocation still has its own unmodified owned runner.
        previous_arguments=sys.argv
        try:
            sys.argv=argv
            runpy.run_path(str(entry),run_name='__main__')
        finally:
            sys.argv=previous_arguments
        terminal=target/'native/TERMINAL.json';value=json.loads(terminal.read_text())
        assert value['complete'] and value['source_head']==proof['child_head'] and value['final_clean']
        routes[layout]={'terminal_path':str(terminal),'terminal_sha256':file_sha(terminal),'fixed_entry_sha256':file_sha(entry),
            'controller_pid':os.getpid(),'python_executable_sha256':file_sha(Path(sys.executable).resolve())}
    assert state(package)==proof['source_before']
    result={'schema':'explicit_synthetic_test_descendant_routes_v1','complete':True,'source_projection_sha256':file_sha(output/'SOURCE_PROJECTION.json'),
        'parent_head':proof['parent_head'],'actual_native_child_head':proof['child_head'],'routes':routes,
        'parent_source_qualification_claim':False,'source_bound_provenance_verified':False}
    (output/'TERMINAL.json').write_bytes(canonical(result))
    print(json.dumps({'complete':True,'directory':str(output),'actual_native_child_head':proof['child_head']}),flush=True)

if __name__=='__main__':main()
