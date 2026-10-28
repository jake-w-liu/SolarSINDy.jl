"""Bounded data-only verification of a retained hosted canonical audit ZIP."""
from pathlib import Path,PurePosixPath
import hashlib,json,re,stat,zipfile

AUDIT_SHA='c7cb2ad1d05a77cf0462c9af6f82f17394d2fc37476fe8c68e57bd4a625c2b28'
OWNER_SHA='0b0f2327841230e5ac87d03bd67d1cbaf10717e0a52ce0134d71169a36146b89'

def digest(body):return hashlib.sha256(body).hexdigest()
def strict_json(body):
    def mapping(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate JSON key')
            result[key]=value
        return result
    return json.loads(body,object_pairs_hook=mapping,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON literal')))

def verify(path,expected):
    assert Path(path).is_file() and Path(path).stat().st_size<=2*1024**3
    with zipfile.ZipFile(path) as archive:
        info=archive.infolist();assert len(info)<=100000
        files={};folded=set();expanded=0
        for entry in info:
            name=entry.filename;parts=PurePosixPath(name).parts
            assert name and not name.startswith('/') and '\\' not in name and ':' not in name and all(p not in ('..','.') for p in parts)
            assert name.rstrip('/')==PurePosixPath(name).as_posix()
            assert name.casefold() not in folded;folded.add(name.casefold())
            kind=stat.S_IFMT(entry.external_attr>>16)
            assert kind in (0,stat.S_IFREG,stat.S_IFDIR) and not (entry.flag_bits&1)
            if entry.is_dir():assert kind in (0,stat.S_IFDIR);continue
            assert kind in (0,stat.S_IFREG) and entry.file_size>=0
            expanded+=entry.file_size;assert expanded<=8*1024**3
            files[name]=entry
        required={'SOURCE_PROOF.json','PREEXECUTION.json','TERMINAL.json','harness.log','run-harness.sh','FIXTURE_METADATA.json'}
        assert required<=set(files)
        def read(name,limit=4*1024**2):
            assert files[name].file_size<=limit
            return archive.read(name)
        source_bytes=read('SOURCE_PROOF.json');source=strict_json(source_bytes)
        before=strict_json(read('PREEXECUTION.json'));terminal=strict_json(read('TERMINAL.json'))
        assert source['complete'] is True and source['head']==expected['head'] and source['tree']==expected['tree']
        assert source['source_files']==expected['source_files'] and source['source_sha256']==expected['source_sha256']
        assert source['commit_object_ids']==expected['commit_object_ids'] and source['all_clean_and_checkout_filters_match'] is True
        for record in (before,terminal):
            assert record['head']==expected['head'] and record['source_sha256']==expected['source_sha256']
            assert record['source_proof_sha256']==digest(source_bytes) and record['canonical_audit_sha256']==AUDIT_SHA and record['owner_sha256']==OWNER_SHA
            assert record['actual_source_access'] is False and record['actual_fits']==0 and record['scientific_performance_claim'] is False
        assert terminal['complete'] is True and terminal['unfiltered_Pkg_test_and_complete_experiments_passed'] is True
        assert terminal['native']['native_exit']==0 and terminal['native']['timed_out'] is False
        assert terminal['available_physical_bytes_at_admission']>=8*1024**3 and terminal['available_disk_bytes_at_admission']>=8*1024**3
        assert terminal['memory_admission_method']=='Linux /proc/meminfo MemAvailable'
        assert terminal['changed_sources']==terminal['changed_controls']==terminal['guard_read_errors']==terminal['inventory_errors']==[]
        assert terminal['git_status']=='' and terminal['head_changed'] is False
        assert terminal['controls_sha256']==before['controls_sha256']
        log=read('harness.log',16*1024**2);text=log.decode('utf8',errors='strict')
        assert digest(log)==terminal['harness_log_sha256']
        counts={key:int(re.search(r'^'+key+r': (\d+)$',text,re.M).group(1)) for key in ('PASS','WARN','FAIL')}
        assert counts==terminal['audit_counts'] and counts['PASS']>0 and counts['FAIL']==0
        assert '[PASS] Pkg.test() passed' in text and '[PASS] examples/experiments.jl executed' in text
        members=terminal['raw_and_operational_members'];assert isinstance(members,dict) and members
        assert len([name for name in members if re.fullmatch(r'raw/dev-harness-pkg-test\.[^/]+',name)])==1
        assert len([name for name in members if re.fullmatch(r'raw/dev-harness-experiments\.[^/]+',name)])==1

        metadata_bytes=read('FIXTURE_METADATA.json',8*1024**2);metadata=strict_json(metadata_bytes)
        assert terminal['artifact_snapshot_ordinary_files_only'] is True and terminal['fixture_targets_followed'] is False
        assert terminal['snapshot_source_identities_and_bytes_reverified'] is True
        assert digest(metadata_bytes)==terminal['fixture_metadata_sha256']
        assert set(metadata)=={'schema','complete','ordinary_file_members','ordinary_file_bytes','fixtures','targets_followed','fixture_objects_loaded'}
        assert metadata['schema']=='canonical_harness_special_fixture_metadata_v1' and metadata['complete'] is True
        assert metadata['targets_followed'] is False and metadata['fixture_objects_loaded'] is False
        assert type(metadata['ordinary_file_members']) is int and metadata['ordinary_file_members']==len(members)
        assert type(metadata['ordinary_file_bytes']) is int and metadata['ordinary_file_bytes']==sum(row['bytes'] for row in members.values())
        fixtures=metadata['fixtures'];assert isinstance(fixtures,list) and len(fixtures)<=100000
        assert type(terminal['special_fixture_objects']) is int and terminal['special_fixture_objects']==len(fixtures)
        fixture_names=set()
        for item in fixtures:
            assert isinstance(item,dict) and item.get('kind') in ('link','fifo','socket')
            assert set(item)==({'path','kind','mode','target'} if item['kind']=='link' else {'path','kind','mode'})
            name=item['path'];assert isinstance(name,str) and 0<len(name.encode('utf-8'))<=4096
            parts=PurePosixPath(name).parts
            assert name.startswith(('raw/','operational/')) and '\\' not in name and ':' not in name
            assert all(part not in ('.','..') for part in parts) and PurePosixPath(name).as_posix()==name
            assert name.casefold() not in fixture_names;fixture_names.add(name.casefold())
            assert not any(member==name or member.startswith(name+'/') for member in members)
            assert type(item['mode']) is int and 0<=item['mode']<=0o177777
            expected_kind={'link':stat.S_IFLNK,'fifo':stat.S_IFIFO,'socket':stat.S_IFSOCK}[item['kind']]
            assert stat.S_IFMT(item['mode'])==expected_kind
            if item['kind']=='link':assert isinstance(item['target'],str) and len(item['target'].encode('utf-8'))<=4096

        actual_raw={name for name in files if name.startswith(('raw/','operational/'))};assert actual_raw==set(members)
        allowed=required|set(members)|{'WAITING_ADMISSION.json'};assert set(files)<=allowed
        retained={}
        for name,row in members.items():
            assert files[name].file_size==row['bytes']
            with archive.open(name) as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
            assert sha==row['sha256'];retained[name]=row
        return dict(complete=True,head=expected['head'],source_files=expected['source_files'],canonical_audit_sha256=AUDIT_SHA,
            audit_counts=counts,unfiltered_Pkg_test_and_complete_experiments_passed=True,
            physical_and_disk_admission_at_least8GiB=True,all_raw_and_operational_members_verified=len(retained),special_fixture_objects_verified=len(fixtures),fixture_metadata_sha256=digest(metadata_bytes),
            expanded_bytes=expanded,source_proof_sha256=digest(source_bytes),harness_log_sha256=digest(log),
            terminal_sha256=digest(read('TERMINAL.json')),software_gate_only=True,actual_source_access=False,actual_fits=0,
            main_promotion_authorized=False,scientific_performance_claim=False)
