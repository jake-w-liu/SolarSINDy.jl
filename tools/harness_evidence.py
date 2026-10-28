"""Retain ordinary audit bytes and encode special test fixtures without following them."""
from pathlib import Path
from contextlib import nullcontext
import hashlib,json,os,stat

MAX_MEMBERS=100000
MAX_BYTES=8*1024**3
BLOCK_BYTES=1024**2

def need(value,message):
    if not value:raise ValueError(message)

def identity(info):
    return (info.st_dev,info.st_ino,info.st_mode,info.st_size,info.st_mtime_ns,info.st_ctime_ns,
            getattr(info,'st_file_attributes',0))

def opened_identity(info):
    # Windows path-stat and descriptor-stat can expose different ctime values.
    # Compare file ID/mode/size/mtime/attributes across APIs;check full ctime
    # identity separately within each API before and after the read.
    value=identity(info)
    return value[:5]+value[6:]

def linked(info):
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info,'st_file_attributes',0)&0x400)

def ordinary(path,directory=False):
    path=Path(path);need(path.is_absolute() and '..' not in path.parts,'absolute ordinary evidence path required')
    for part in (path,*path.parents):need(not linked(part.lstat()),'linked evidence ancestor')
    info=path.lstat()
    need(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode),'ordinary evidence kind differs')
    return info

def _checked_bytes(source,destination=None):
    """Check the opened identity before reading through a bounded buffer."""
    source=Path(source);before=ordinary(source)
    if destination is not None:
        destination=Path(destination)
        ordinary(destination.parent,True);need(not destination.exists() and not destination.is_symlink(),'fresh snapshot member required')
    flags=os.O_RDONLY|getattr(os,'O_BINARY',0)|getattr(os,'O_NOFOLLOW',0)
    fd=os.open(source,flags);hasher=hashlib.sha256();total=0
    try:
        with os.fdopen(fd,'rb') as stream:
            fd=None;opened=os.fstat(stream.fileno())
            need(opened_identity(opened)==opened_identity(before) and stat.S_ISREG(opened.st_mode) and not linked(opened),
                 'evidence open identity differs: before='+repr(identity(before))+';opened='+repr(identity(opened)))
            with (destination.open('xb') if destination is not None else nullcontext()) as output:
                while True:
                    body=stream.read(BLOCK_BYTES)
                    if not body:break
                    total+=len(body);need(total<=before.st_size<=MAX_BYTES,'evidence member byte bound')
                    hasher.update(body)
                    if output is not None:output.write(body)
            need(total==before.st_size and identity(os.fstat(stream.fileno()))==identity(opened),'evidence changed while copying')
        need(identity(ordinary(source))==identity(before),'evidence source replaced after copy')
    finally:
        if fd is not None:os.close(fd)
    return dict(bytes=total,sha256=hasher.hexdigest())

def checked_copy(source,destination):
    return _checked_bytes(source,destination)

def walk(roots):
    records={};files={};fixtures=[]
    for prefix,root in roots:
        root=Path(root);ordinary(root,True);stack=[root]
        while stack:
            parent=stack.pop();ordinary(parent,True)
            with os.scandir(parent) as scanner:entries=sorted(scanner,key=lambda entry:entry.name)
            for entry in entries:
                # DirEntry stat data can be cached by the directory scan.
                path=parent/entry.name;info=path.lstat()
                name=prefix+'/'+path.relative_to(root).as_posix()
                need(name not in records and len(records)<MAX_MEMBERS,'evidence member count/path collision')
                records[name]=identity(info)
                if linked(info):
                    target=os.readlink(path)
                    need(isinstance(target,str) and len(target.encode('utf-8'))<=4096,'fixture link target text bound')
                    fixtures.append(dict(path=name,kind='link',mode=info.st_mode,target=target))
                elif stat.S_ISDIR(info.st_mode):stack.append(path)
                elif stat.S_ISREG(info.st_mode):files[name]=path
                elif stat.S_ISFIFO(info.st_mode):fixtures.append(dict(path=name,kind='fifo',mode=info.st_mode))
                elif stat.S_ISSOCK(info.st_mode):fixtures.append(dict(path=name,kind='socket',mode=info.st_mode))
                else:raise ValueError('unsupported special evidence object:'+name)
    return records,files,fixtures

def snapshot_members(raw,operational,destination):
    """Copy all regular bytes, including hidden files; preserve special-object metadata."""
    destination=Path(destination);ordinary(destination.parent,True)
    need(not destination.exists() and not destination.is_symlink(),'fresh artifact snapshot required')
    roots=(('raw',Path(raw)),('operational',Path(operational)))
    for _,root in roots:
        need(not destination.is_relative_to(root) and not root.is_relative_to(destination),'snapshot and source roots must be disjoint')
    records,files,fixtures=walk(roots)
    need(sum(path.stat().st_size for path in files.values())<=MAX_BYTES,'total evidence byte bound')
    destination.mkdir();members={};total=0
    for name,source in sorted(files.items()):
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        members[name]=checked_copy(source,target);total+=members[name]['bytes']
        need(total<=MAX_BYTES,'total copied evidence byte bound')
    after,after_files,after_fixtures=walk(roots)
    need(records==after and set(files)==set(after_files) and fixtures==after_fixtures,'evidence tree changed during snapshot')
    # Directory metadata may be cached or have coarse timestamp resolution.
    # Rehash every current ordinary source, rather than trusting metadata alone.
    for name,source in sorted(after_files.items()):
        need(_checked_bytes(source)==members[name],'evidence contents changed after snapshot copy')
    manifest=dict(schema='canonical_harness_special_fixture_metadata_v1',complete=True,
        ordinary_file_members=len(members),ordinary_file_bytes=total,fixtures=fixtures,
        targets_followed=False,fixture_objects_loaded=False)
    body=(json.dumps(manifest,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
    need(len(body)<=8*1024**2,'fixture metadata byte bound')
    with (destination/'FIXTURE_METADATA.json').open('xb') as stream:stream.write(body)
    return dict(members=members,fixture_metadata_sha256=hashlib.sha256(body).hexdigest(),
                special_fixture_objects=len(fixtures),ordinary_bytes=total,source_identities_unchanged=True)
