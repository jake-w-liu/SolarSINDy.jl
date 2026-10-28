"""Bounded independent snapshot binding; no model or source-number parsing."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import json,re,struct
MEMBER_BYTES=8_388_608
SNAPSHOT_BYTES=1_073_741_824
MAX_MEMBERS=1024
SHARD_ROWS=8192
class SnapshotError(ValueError):pass
def need(condition,message):
    if not condition:raise SnapshotError(message)
def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
def _pairs(pairs):
    value={}
    for key,item in pairs:
        need(key not in value,'duplicate JSON key');value[key]=item
    return value
def _no_constant(value):raise SnapshotError('nonfinite JSON atom')
def _shape(value,depth=0):
    need(depth<=16,'metadata nesting bound')
    if isinstance(value,dict):
        need(len(value)<=32 and all(isinstance(k,str) for k in value),'metadata object bound')
        for key,item in value.items():_shape(key,depth+1);_shape(item,depth+1)
    elif isinstance(value,list):
        need(len(value)<=8192,'metadata array bound')
        for item in value:_shape(item,depth+1)
    elif isinstance(value,str):need(len(value.encode('utf-8'))<=4096,'metadata string bound')
    else:need(value is None or isinstance(value,(bool,int)),'metadata atom type')
def _ordinary(path,directory=False):
    path=Path(path)
    need(path.is_absolute(),'absolute path required')
    need(all(not part.is_symlink() for part in (path,*path.parents)),'link path component')
    need(path.is_dir() if directory else path.is_file(),'ordinary path missing')
    return path
def _identity(path):
    s=path.stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns
def bounded_read(path,limit):
    path=_ordinary(path);before=_identity(path);need(0<=before[2]<=limit,'metadata byte bound')
    with path.open('rb') as stream:data=stream.read(limit+1)
    need(len(data)==before[2] and _identity(path)==before,'file changed during read')
    return data
def decoded(data):
    try:value=json.loads(data,object_pairs_hook=_pairs,parse_constant=_no_constant)
    except (UnicodeDecodeError,json.JSONDecodeError) as error:raise SnapshotError('invalid JSON') from error
    _shape(value);need(canonical(value)==data,'noncanonical JSON bytes')
    return value
def _integer(value,maximum=2**63-1):
    need(type(value) is int and 0<=value<=maximum,'bounded integer required');return value
def _sha(value):
    need(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None,'SHA256 syntax');return value
def _hash(path,limit=MEMBER_BYTES):
    path=_ordinary(path);before=_identity(path);need(before[2]<=limit,'member byte bound')
    value=sha256();_feed(value,path,before[2])
    need(_identity(path)==before,'member identity changed');return value.hexdigest(),before[2]
def _feed(digest,path,size):
    path=_ordinary(path);before=_identity(path);need(before[2]==size<=MEMBER_BYTES,'streamed member size differs')
    with path.open('rb') as stream:
        remaining=size
        while remaining:
            chunk=stream.read(min(1_048_576,remaining));need(bool(chunk),'truncated streamed member')
            digest.update(chunk);remaining-=len(chunk)
        need(stream.read(1)==b'','extended streamed member')
    need(_identity(path)==before,'streamed member identity changed')
def _minute(text):
    need(isinstance(text,str) and re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:00\.000Z',text) is not None,'minute declaration syntax')
    try:stamp=datetime.fromisoformat(text.replace('Z','+00:00'))
    except ValueError as error:raise SnapshotError('invalid minute declaration') from error
    delta=stamp-datetime(1970,1,1,tzinfo=timezone.utc)
    return delta.days*86_400_000+delta.seconds*1000+delta.microseconds//1000
