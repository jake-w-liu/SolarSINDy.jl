"""Bounded, read-only artifact I/O. Concurrent hostile replacement is out of scope."""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat


class InvalidArtifact(ValueError):
    pass


def need(ok,message):
    if not ok:
        raise InvalidArtifact(message)


def is_integer(value,minimum=0):
    return type(value) is int and value>=minimum


def is_digest(value):
    return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None


@dataclass(frozen=True)
class Limits:
    file_bytes: int = 512*1024**2
    total_bytes: int = 2*1024**3
    files: int = 512
    directories: int = 128
    json_bytes: int = 64*1024**2
    depth: int = 64
    nodes: int = 4_000_000
    string_characters: int = 1024**2

    def validate(self):
        need(all(is_integer(value,1) for value in vars(self).values()),'positive integer artifact budgets required')
        need(self.json_bytes<=self.file_bytes<=self.total_bytes,'inconsistent artifact byte budgets')


def ordinary(metadata):
    return not stat.S_ISLNK(metadata.st_mode) and not getattr(metadata,'st_file_attributes',0)&0x400


def spelling(path):
    return str(Path(path).resolve(strict=True))


def plain_root(path):
    root=Path(path)
    need(root.is_absolute(),'explicit absolute artifact root required')
    root=root.absolute()
    need(root.is_dir(),'existing artifact directory required')
    for current in (root,*root.parents):
        need(ordinary(current.lstat()),'artifact root has a link/reparse ancestor')
    need(spelling(root)==str(root),'artifact root case or location alias')
    return root


def relative_parts(name):
    need(isinstance(name,str) and name and '\\' not in name and ':' not in name,'invalid relative artifact name')
    path=PurePosixPath(name)
    parts=name.split('/')
    need(not path.is_absolute() and all(part and part not in ('.','..') and
         all(ord(char)>=32 and char not in '\\/:"<>|?*' for char in part) and
         not part.endswith(('.', ' ')) and not re.fullmatch(r'(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?',part,re.I)
         for part in parts),'unsafe artifact path component')
    return parts


def member(root,name):
    path=plain_root(root)
    for part in relative_parts(name):
        path=path/part
        need(ordinary(path.lstat()),'artifact member is a link/reparse point')
        need(spelling(path)==str(path),'artifact member case or location alias')
    need(stat.S_ISREG(path.lstat().st_mode),'ordinary artifact file required')
    return path


def read_file(path,limit,materialize=False):
    need(is_integer(limit,1),'positive artifact byte cap required')
    path=Path(path);before=path.lstat()
    need(ordinary(before) and stat.S_ISREG(before.st_mode) and before.st_size<=limit,'artifact type or byte budget')
    context=hashlib.sha256();count=0;chunks=[]
    with path.open('rb') as stream:
        opened=os.fstat(stream.fileno())
        need((opened.st_dev,opened.st_ino)==(before.st_dev,before.st_ino),'artifact identity changed before read')
        while True:
            chunk=stream.read(min(65536,limit-count+1))
            if not chunk:
                break
            count+=len(chunk)
            need(count<=limit,'artifact grew beyond byte budget')
            context.update(chunk)
            if materialize:
                chunks.append(chunk)
    after=path.lstat()
    need(ordinary(after) and stat.S_ISREG(after.st_mode) and
         (before.st_dev,before.st_ino,before.st_size)==(after.st_dev,after.st_ino,after.st_size) and
         count==before.st_size,'artifact changed during read')
    return {'sha256':context.hexdigest(),'bytes':count,'data':b''.join(chunks) if materialize else None}


def lexical_json(text,limits):
    stack=[];quoted=False;escaped=False;characters=0;nodes=0;scalar=False
    for char in text:
        if quoted:
            characters+=1
            need(characters<=limits.string_characters,'JSON string budget')
            if escaped:
                escaped=False
            elif char=='\\':
                escaped=True
            elif char=='"':
                quoted=False
            continue
        if char=='"':
            quoted=True;escaped=False;characters=0;nodes+=1;scalar=False
        elif char in '[{':
            stack.append(char);nodes+=1;scalar=False
            need(len(stack)<=limits.depth,'JSON depth budget')
        elif char in ']}':
            need(stack and stack.pop()==('[' if char==']' else '{'),'mismatched JSON container')
            scalar=False
        elif char in ',:' or char.isspace():
            scalar=False
        elif not scalar:
            scalar=True;nodes+=1
        need(nodes<=limits.nodes,'JSON node budget before parsing')
    need(not quoted and not stack,'unterminated JSON token/container')


def unique_pairs(items):
    result={}
    for key,value in items:
        need(key not in result,'duplicate decoded JSON key')
        result[key]=value
    return result


def decode_json(data,limits=Limits()):
    limits.validate()
    need(isinstance(data,bytes) and len(data)<=limits.json_bytes,'JSON byte budget')
    text=data.decode('utf-8','strict')
    lexical_json(text,limits)
    def invalid_constant(value):
        raise InvalidArtifact('nonfinite JSON constant: '+value)
    result=json.loads(text,object_pairs_hook=unique_pairs,parse_constant=invalid_constant)
    pending=[result]
    while pending:
        value=pending.pop()
        if isinstance(value,dict):pending.extend(value.values())
        elif isinstance(value,list):pending.extend(value)
        elif isinstance(value,float):need(math.isfinite(value),'overflowed JSON numeric literal')
    return result


def json_file(path,expected,limits=Limits()):
    limits.validate()
    need(is_digest(expected),'caller-pinned JSON digest required')
    record=read_file(path,limits.json_bytes,materialize=True)
    need(record['sha256']==expected,'JSON file digest mismatch')
    return decode_json(record['data'],limits)


def inventory(root,limits=Limits()):
    limits.validate();root=plain_root(root)
    pending=[root];files=[];directories=1;total=0;aliases=set()
    # Preflight the full node/byte inventory before hashing file bodies.
    while pending:
        current=pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                path=current/entry.name;relative=path.relative_to(root).as_posix()
                relative_parts(relative)
                key=relative.casefold();need(key not in aliases,'case-aliased artifact members');aliases.add(key)
                metadata=path.lstat();need(ordinary(metadata),'linked artifact inventory member')
                if stat.S_ISDIR(metadata.st_mode):
                    directories+=1;need(directories<=limits.directories,'artifact directory-count budget')
                    pending.append(path)
                else:
                    need(stat.S_ISREG(metadata.st_mode),'special artifact inventory member')
                    files.append((relative,path,metadata.st_size))
                    need(len(files)<=limits.files,'artifact file-count budget')
                    need(metadata.st_size<=limits.file_bytes and metadata.st_size<=limits.total_bytes-total,'artifact inventory byte budget')
                    total+=metadata.st_size
    result={}
    for relative,path,size in files:
        record=read_file(path,limits.file_bytes)
        need(record['bytes']==size,'artifact changed after inventory preflight')
        result[relative]=record['sha256']
    return result
