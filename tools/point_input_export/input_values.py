"""Independent bounded comparison of native fit input with sealed case records.

No source-cache path is opened, no model is fitted and no permission is granted.
The imported case reader verifies all artifact members independently first.
"""
from pathlib import Path
from datetime import datetime,timezone
from hashlib import sha256
import hashlib,importlib.util,math,re,struct

CASE_READER=Path(__file__).with_name('case_records.py')
assert sha256(CASE_READER.read_bytes()).hexdigest()=='70c4f8076c63098b791a7e43bac246648315f56f066b86057906cc167d399e38'
spec=importlib.util.spec_from_file_location('point_input_independent_case_reader',CASE_READER)
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
need=C.need
PREFIX=b'SOURCE_BOUND_POINT_INPUT_V1\n'
MAX_ROWS=600000
MAX_BYTES=134217728
# Julia DateTime stores Rata Die milliseconds (Dates/src/conversions.jl).
# Python ordinal days use the same day count; derive the Unix epoch without
# reading or trusting a native exported timestamp.
JULIA_UNIX_OFFSET=datetime(1970,1,1).toordinal()*86400000
LOG_ABS=2e-15
LOG_REL=2e-15

def pin(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def integer(value,low,high,message):
    need(type(value) is int and low<=value<=high,message)
    return value

def minute(value):
    need(isinstance(value,str) and re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:00\.000Z',value) is not None,'split clock syntax differs')
    return C._millis(value)

def check_input_bytes(data,records,*,start_ms,stop_ms,asof_ms,width):
    """Verify a complete canonical input stream against independent row records.

    Each record is (one-based row id,160-byte feature record,64-byte label record).
    The generator must cover the complete scheduled table in exact row order.
    """
    need(isinstance(data,bytes) and len(data)<=MAX_BYTES and data.startswith(PREFIX),'native input header or size differs')
    integer(width,8,11,'input width type differs');need(width in (8,11),'input width differs')
    need(type(start_ms) is type(stop_ms) is type(asof_ms) is int and start_ms<stop_ms<=asof_ms,'split clocks differ')
    base=len(PREFIX);need(len(data)>=base+16,'native input geometry missing')
    n,p=struct.unpack_from('<QQ',data,base);base+=16
    integer(n,2,MAX_ROWS,'native input row bound differs');need(p==width,'native input matrix width differs')
    need(len(data)==base+8*n*(p+6),'native input complete byte count differs')
    row_at=base;time_at=row_at+8*n;matrix_at=time_at+8*n
    targets_at=matrix_at+8*n*p;logs_at=targets_at+8*n;now_at=logs_at+8*n;scales_at=now_at+8*n
    previous_row=0;previous_issue=None;kept=0;scheduled=0;first=None;last=None;worst=0.
    for row,feature,label in records:
        integer(row,1,MAX_ROWS,'case row id type differs')
        need(row==previous_row+1 and isinstance(feature,bytes) and len(feature)==160 and isinstance(label,bytes) and len(label)==64,'complete case record sequence differs')
        previous_row=row;scheduled+=1
        issue,ready=struct.unpack_from('<qq',feature)
        need(previous_issue is None or issue==previous_issue+60000,'case issue cadence differs')
        previous_issue=issue
        need(issue%60000==0 and ready==issue+1845000,'case completion clock differs')
        need(struct.unpack_from('<qq',label)==(issue,ready),'paired case clocks differ')
        accepted=feature[16];label_ok=label[16]
        need(accepted in (0,1) and label_ok in (0,1),'case admission flags differ')
        if not (start_ms<=issue<stop_ms and ready<stop_ms and ready<=asof_ms and accepted and label_ok):continue
        need(kept<n,'native input omitted case rows')
        got_row=struct.unpack_from('<Q',data,row_at+8*kept)[0]
        got_issue=struct.unpack_from('<q',data,time_at+8*kept)[0]
        need(got_row==row and got_issue==issue+JULIA_UNIX_OFFSET,'native row membership or timestamp epoch differs')
        for column in range(p):
            expected=feature[40+8*column:48+8*column]
            actual=data[matrix_at+8*(column*n+kept):matrix_at+8*(column*n+kept+1)]
            need(actual==expected,'native feature column bytes differ')
            need(math.isfinite(struct.unpack('<d',actual)[0]),'nonfinite native feature')
        for offset,expected in ((targets_at,label[24:32]),(now_at,feature[24:32]),(scales_at,feature[32:40])):
            actual=data[offset+8*kept:offset+8*(kept+1)]
            need(actual==expected,'native target/raw-now/scale bytes differ')
        target=struct.unpack_from('<d',label,24)[0];scale=struct.unpack_from('<d',feature,32)[0]
        raw=struct.unpack_from('<d',feature,24)[0]
        need(math.isfinite(target) and target>=0 and math.isfinite(raw) and raw>=0 and math.isfinite(scale) and scale>0,'native scalar domain differs')
        transformed=struct.unpack_from('<d',data,logs_at+8*kept)[0];oracle=math.log1p(target)
        error=abs(transformed-oracle)
        need(math.isfinite(transformed) and error<=LOG_ABS+LOG_REL*abs(oracle),'native log1p target differs from independent libm oracle')
        worst=max(worst,error)
        if first is None:first=row
        last=row;kept+=1
    need(kept==n,'native training membership count differs')
    return dict(complete=True,rows=n,columns=p,scheduled_rows_scanned=scheduled,first_row=first,last_row=last,
        input_sha256=sha256(data).hexdigest(),native_input_bytes=len(data),all_source_input_coordinates_byte_exact=True,
        transformed_targets_independently_checked=n,max_independent_log_absolute_error=worst,
        transformed_target_comparison=dict(absolute=LOG_ABS,relative=LOG_REL),
        source_paths_opened=False,actual_model_fits=0,fit_authority_granted=False)

def verify_training_input(directory,request_path,input_path,*,request_sha256,manifest_sha256,complete_sha256,
                          input_sha256,table_index,start,stop,asof,expected_authority=None):
    """Verify full artifact provenance and all native input values, without fitting."""
    directory=Path(directory);request_path=Path(request_path);input_path=Path(input_path)
    report=C.verify_cases(directory,request_path,request_sha256=request_sha256,manifest_sha256=manifest_sha256,
        complete_sha256=complete_sha256,expected_authority=expected_authority)
    request=C.decoded(C.read(request_path,131072));manifest=C.decoded(C.read(directory/'manifest.json',1048576))
    count=len(request['tables']);integer(table_index,1,count,'input table index differs')
    table=manifest['tables'][table_index-1];declaration=request['tables'][table_index-1]
    begin,end,cutoff=map(minute,(start,stop,asof))
    need(minute(declaration['start_inclusive'])<=begin<end<=minute(declaration['stop_exclusive']) and cutoff<=minute(declaration['label_asof']),'input split exceeds sealed table')
    if request['evidence_mode']=='actual_archival_cache':
        need(minute('2017-01-01T00:00:00.000Z')<=begin<end<=minute('2018-01-01T00:00:00.000Z'),'actual point input outside strict2017 domain')
    need(pin(request_path)==request_sha256 and pin(directory/'manifest.json')==manifest_sha256,'case metadata changed before input scan')
    before=C.V._identity(C.ordinary(input_path));data=C.read(input_path,MAX_BYTES)
    C.V._sha(input_sha256);need(sha256(data).hexdigest()==input_sha256,'native input external pin differs')
    entries={entry['name']:entry for entry in manifest['members']};identities={}
    def records():
        row=0
        for fn,ln in zip(table['feature_shards'],table['label_shards']):
            bodies=[]
            for name,width in ((fn,160),(ln,64)):
                path=directory/name;identity=C.V._identity(C.ordinary(path));body=C.read(path,8192*width)
                need(len(body)==entries[name]['bytes'] and sha256(body).hexdigest()==entries[name]['sha256'],'case shard changed before input scan')
                identities[path]=identity;bodies.append(body)
            fb,lb=bodies;need(len(fb)//160==len(lb)//64,'paired input shard geometry differs')
            for local in range(len(fb)//160):
                row+=1;yield row,fb[160*local:160*(local+1)],lb[64*local:64*(local+1)]
    result=check_input_bytes(data,records(),start_ms=begin,stop_ms=end,asof_ms=cutoff,width=11 if request['layout']=='ground-only' else 8)
    need(result['scheduled_rows_scanned']==table['scheduled'],'input schedule count differs')
    need(C.V._identity(C.ordinary(input_path))==before and pin(input_path)==input_sha256,'native input changed during verification')
    need(all(C.V._identity(C.ordinary(path))==identity for path,identity in identities.items()),'case shard changed during input verification')
    need(pin(request_path)==request_sha256 and pin(directory/'manifest.json')==manifest_sha256 and pin(directory/'COMPLETE.json')==complete_sha256,'case metadata changed during input verification')
    result.update(case_verification=report,table_index=table_index,station=declaration['station'],runtime=request['runtime'],role=request['role'],layout=request['layout'],
        start_inclusive=start,stop_exclusive=stop,target_asof=asof,input_path=str(input_path),verifier_sha256=pin(__file__))
    return result
