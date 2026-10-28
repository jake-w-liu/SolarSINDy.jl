"""Independent full-coordinate checks for the generated numeric resource fixture."""
from pathlib import Path
from datetime import datetime
import hashlib,importlib.util,struct
BASE=Path(__file__).with_name('bundle_reader.py')
spec=importlib.util.spec_from_file_location('point_resource_bundle_reader',BASE)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
V=M.V;C=M.C;need=V.need

def verify_primitive(directory,expected_report,*,width):
    directory=C.ordinary(Path(directory),True)
    raw=C.read(directory/'REPORT.json',65536);report=C.decoded(raw)
    C.keys(report,('complete','numeric_primitive_only','rows','columns','input_sha256','input_bytes','members','elapsed_nanoseconds','actual_source_access','actual_fits'))
    need(report==expected_report and report['complete'] is True and report['numeric_primitive_only'] is True and
         report['actual_source_access'] is False and C.integer(report['actual_fits'],0)==0,'numeric resource report differs')
    n=C.integer(report['rows'],600000,600000);need(C.integer(report['columns'],11)==width and width in (8,11),'numeric resource geometry differs')
    C.integer(report['elapsed_nanoseconds'],2**63-1)
    total=C.integer(report['input_bytes'],V.MAX_BYTES,1);C.V._sha(report['input_sha256'])
    entries=report['members'];count=(total+M.CHUNK_BYTES-1)//M.CHUNK_BYTES
    need(isinstance(entries,list) and len(entries)==count and 1<=count<=16,'numeric resource chunks differ')
    bodies=[];identities={};names={'REPORT.json'}
    for index,entry in enumerate(entries,1):
        C.keys(entry,('name','bytes','sha256'));name=f'input-{index:04d}.bin';need(entry['name']==name,'numeric resource chunk order differs')
        size=C.integer(entry['bytes'],M.CHUNK_BYTES,1);need(size==min(M.CHUNK_BYTES,total-(index-1)*M.CHUNK_BYTES),'numeric resource chunk size differs')
        path=directory/name;before=C.V._identity(C.ordinary(path));body=C.read(path,M.CHUNK_BYTES)
        need(len(body)==size and hashlib.sha256(body).hexdigest()==entry['sha256'],'numeric resource chunk hash differs')
        names.add(name);identities[path]=before;bodies.append(body)
    need({path.name for path in directory.iterdir()}==names,'numeric resource fileset differs')
    data=b''.join(bodies);need(len(data)==total and hashlib.sha256(data).hexdigest()==report['input_sha256'],'numeric resource full stream differs')
    first=(datetime(2020,1,1)-datetime(1970,1,1)).days*86400000
    def records():
        for i in range(1,n+1):
            issue=first+(i-1)*60000;ready=issue+1845000
            feature=bytearray(160);label=bytearray(64)
            struct.pack_into('<qqBB',feature,0,issue,ready,1,0);struct.pack_into('<qqBB',label,0,issue,ready,1,0)
            struct.pack_into('<dd',feature,24,(i%7)/8,1+(i%13)/16)
            for j in range(1,width+1):struct.pack_into('<d',feature,40+8*(j-1),(i*(2*j+1)%257)/256)
            struct.pack_into('<d',label,24,(i%19+1)/4)
            yield i,bytes(feature),bytes(label)
    stop=first+(n+31)*60000
    result=V.check_input_bytes(data,records(),start_ms=first,stop_ms=stop,asof_ms=stop,width=width)
    need(result['rows']==result['scheduled_rows_scanned']==600000 and result['columns']==width,'numeric resource independent count differs')
    need(all(C.V._identity(C.ordinary(path))==before for path,before in identities.items()) and C.read(directory/'REPORT.json',65536)==raw,'numeric resource input changed')
    result.update(numeric_primitive_only=True,source_case_claim=False,profile_report_sha256=hashlib.sha256(raw).hexdigest(),actual_source_access=False,actual_fits=0)
    return result
