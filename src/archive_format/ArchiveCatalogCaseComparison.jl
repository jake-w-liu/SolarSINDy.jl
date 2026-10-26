"""Pure comparison of completed-source archival case records across native jobs.

The caller supplies already source-bound bytes. This reader checks encoding,
schedule, source-support agreement and a fixed generated-tested floating rule.
It performs no I/O and grants no source, fitting or scientific qualification.
"""
module ArchiveCatalogCaseComparison
using Dates,SHA
import ..ArchiveCatalogCases, ..CompletedMinuteReplay, ..CompletedSourceDBDT
const C=ArchiveCatalogCases
const M=CompletedMinuteReplay
const R=CompletedSourceDBDT.CompletedRefitArtifacts
export comparison_policy, policy_digest, compare_case_bytes
const POLICY_SCHEMA="archive_catalog_case_comparison_policy_v1"
const U=2.0^-53
const COMPONENT_LIMIT=90000.0
# This is the parser's admitted coordinate range, not a physical maximum.
const RATE_LIMIT=nextfloat(hypot(2COMPONENT_LIMIT,2COMPONENT_LIMIT))
const LOG_LIMIT=nextfloat(nextfloat(log1p(RATE_LIMIT)))
const SCALE_LIMIT=nextfloat(nextfloat(1+LOG_LIMIT))
need(ok,message)=ok ? nothing : throw(ArgumentError(message))

"""Return an immutable description of the fixed comparison criterion.

Summary rounding uses128*u*peak, propagated through log1p and positive-denominator
ratios. Comparison adds both native error bounds. Calendar terms have their own
64*u*(1+2*pi) bound; direct wind fields remain bit exact. Generated independent
oracles support this regression criterion; it is not a universal libm proof.
"""
comparison_policy()=(;schema=POLICY_SCHEMA,unit_roundoff_bits=string(reinterpret(UInt64,U);base=16,pad=16),summary_multiplier=128,
    unary_multiplier=32,calendar_multiplier=64,peak_reconstruction_factor=2,
    admitted_ground_component_limit=Int(COMPONENT_LIMIT),maximum_rows=C.MAX_ROWS,
    feature_record_bytes=C.FEATURE_BYTES,label_record_bytes=C.LABEL_BYTES,
    timing="unchanged_default_completed_source_policy",direct_wind_fields_bit_exact=true,
    qualification="pure_comparison_only")

"""SHA256 of the canonical fixed comparison description; no file is read."""
policy_digest()=bytes2hex(sha256(R.canonical_bytes(Dict(string(key)=>value for (key,value) in pairs(comparison_policy()));limit=4096)))

function _owned(value,limit)
    need(value isa String && ncodeunits(value)<=limit,"case comparison bounded binary string required")
    String(copy(codeunits(value)))
end
function _same(a,at,b,bt,count)
    all(codeunit(a,at+offset)==codeunit(b,bt+offset) for offset in 0:count-1)
end
_zero(value,at,count)=all(codeunit(value,at+offset)==0 for offset in 0:count-1)
function _header(bytes,at,issue,ready,reasons)
    need(C._word(bytes,at,8)==reinterpret(UInt64,C._stamp(issue)) &&
         C._word(bytes,at+8,8)==reinterpret(UInt64,C._stamp(ready)),"case comparison issue/readiness differs")
    status=codeunit(bytes,at+16);reason=codeunit(bytes,at+17)
    need(status in (0,1) && Int(reason)<length(reasons) && (status==1)==(reason==0),"case comparison status/reason differs")
    need(_zero(bytes,at+18,6),"case comparison padding differs")
    status==1
end
_values(bytes,at)=ntuple(index->C._number(bytes,at+24+8(index-1)),13)
function _numeric(mode,v)
    need(all(isfinite,v) && 0<=v[1]<=RATE_LIMIT && 1<=v[2]<=SCALE_LIMIT,"case comparison invalid accepted summary")
    if mode===:ground11
        need(all(index->0<=v[index]<=LOG_LIMIT,3:6) && all(index->v[index]>=0,7:9) &&
             v[7]<=1 && v[8]<=1 && all(index->-1<=v[index]<=1,10:13),"case comparison invalid ground features")
        need(reinterpret(UInt64,v[2])==reinterpret(UInt64,1+v[5]),"case comparison ground scale differs")
    else
        need(all(index->0<=v[index]<=RATE_LIMIT,3:6) && 100<v[7]<90000 && abs(v[8])<9000 &&
             0<=v[9]<9000 && 0<=v[10]<810000000 && v[9]==max(-v[8],0.0) && v[10]==v[7]*v[9],
             "case comparison invalid mixed features")
        need(reinterpret(UInt64,v[1])==reinterpret(UInt64,v[3]),"case comparison latest rate differs")
    end
end
function _bounds(mode,left,right)
    peak_left=mode===:ground11 ? expm1(left[5]) : left[5]
    peak_right=mode===:ground11 ? expm1(right[5]) : right[5]
    # Ground peaks have already passed their log-domain admission. Inverting
    # the rounded logarithm can exceed the raw limit by a few rounding units.
    need(isfinite(peak_left) && isfinite(peak_right) && peak_left>=0 && peak_right>=0,
         "case comparison reconstructed peak differs")
    peak_upper=min(RATE_LIMIT,2max(peak_left,peak_right))
    delta=128U*peak_upper
    scale=2*(delta/(1+max(0.0,min(peak_left,peak_right)-delta))+32U*max(abs(left[2]),abs(right[2])))
    if mode===:mixed8
        return (2delta,scale,2delta,2delta,2delta,2delta,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    end
    raw_left=ntuple(index->expm1(left[index+2]),4)
    raw_right=ntuple(index->expm1(right[index+2]),4)
    logbounds=ntuple(4) do index
        2*(delta/(1+max(0.0,min(raw_left[index],raw_right[index])-delta))+
           32U*max(abs(left[index+2]),abs(right[index+2])))
    end
    function ratio_bound(numerator_left,numerator_right,denominator_left,denominator_right,index)
        low=1+max(0.0,min(denominator_left,denominator_right)-delta)
        2*(delta/low+max(abs(numerator_left),abs(numerator_right))*delta/(low*(1+min(denominator_left,denominator_right)))+
           32U*max(abs(left[index]),abs(right[index])))
    end
    a=ratio_bound(left[1],right[1],peak_left,peak_right,7)
    b=ratio_bound(raw_left[2],raw_right[2],peak_left,peak_right,8)
    c=ratio_bound(raw_left[4],raw_right[4],raw_left[2],raw_right[2],9)
    calendar=2*64U*(1+2pi)
    (2delta,scale,logbounds...,a,b,c,calendar,calendar,calendar,calendar)
end
function _compare_number(left,right,bound)
    need(isfinite(bound) && bound>=0,"case comparison invalid floating bound")
    allowance=bound==0 ? 0.0 : nextfloat(bound)
    error=abs(left-right)
    need(error<=allowance,"case comparison numerical disagreement")
    allowance==0 ? 0.0 : error/allowance
end

"""Compare two owned160-byte feature and64-byte label streams.

`start` and `stop` delimit a complete whole-minute issue schedule under the
unchanged default completion policy. Exclusions, clocks, padding, unused fields
and support hashes must agree exactly. Accepted numerical values must pass the
fixed comparison policy; supplied direct wind values are bit exact.

This function does not establish provenance of caller-supplied bytes or replace
native `ArchiveCatalogCaseJob.read_verified` checks. Its result contains counts
and diagnostics, never fitting arrays or an actual-source permission.
"""
function compare_case_bytes(feature_left,label_left,feature_right,label_right;mode,start,stop)
    need(mode in (:ground11,:mixed8),"case comparison layout differs")
    need(start isa DateTime && stop isa DateTime && floor(start,Minute)==start && floor(stop,Minute)==stop,
         "case comparison whole-minute domain required")
    span=Int128(Dates.value(stop))-Int128(Dates.value(start))
    need(span>0 && mod(span,60000)==0 && span÷60000<=C.MAX_ROWS,"case comparison domain bound")
    rows=Int(span÷60000)
    fl=_owned(feature_left,C.FEATURE_BYTES*C.MAX_ROWS);fr=_owned(feature_right,C.FEATURE_BYTES*C.MAX_ROWS)
    ll=_owned(label_left,C.LABEL_BYTES*C.MAX_ROWS);lr=_owned(label_right,C.LABEL_BYTES*C.MAX_ROWS)
    need(ncodeunits(fl)==ncodeunits(fr)==C.FEATURE_BYTES*rows && ncodeunits(ll)==ncodeunits(lr)==C.LABEL_BYTES*rows,
         "case comparison payload width differs")
    features_accepted=0;labels_accepted=0;supervised=0;changed_features=0;changed_labels=0;maximum_fraction=0.0
    for row in 1:rows
        issue=M.shift(start,Int64(60000)*(row-1));ready=M.issue_timing(issue).label_assumed_ready
        fa=C.FEATURE_BYTES*(row-1)+1;la=C.LABEL_BYTES*(row-1)+1
        fleft=_header(fl,fa,issue,ready,C.FEATURE_REASONS);fright=_header(fr,fa,issue,ready,C.FEATURE_REASONS)
        lleft=_header(ll,la,issue,ready,C.LABEL_REASONS);lright=_header(lr,la,issue,ready,C.LABEL_REASONS)
        need(fleft==fright && lleft==lright && _same(fl,fa+16,fr,fa+16,8) && _same(ll,la+16,lr,la+16,8),
             "case comparison exclusion partition differs")
        need(_same(fl,fa+128,fr,fa+128,32) && _same(ll,la+32,lr,la+32,32),"case comparison source support differs")
        features_accepted+=fleft;labels_accepted+=lleft;supervised+=fleft&&lleft
        if !fleft
            need(_zero(fl,fa+24,104) && _zero(fr,fa+24,104),"case comparison rejected feature payload differs")
        else
            if mode===:mixed8
                need(_zero(fl,fa+104,24) && _zero(fr,fa+104,24),"case comparison unused feature columns differ")
            end
            left=_values(fl,fa);right=_values(fr,fa);_numeric(mode,left);_numeric(mode,right)
            bounds=_bounds(mode,left,right)
            for index in 1:13
                if mode===:mixed8 && 7<=index<=13
                    need(reinterpret(UInt64,left[index])==reinterpret(UInt64,right[index]),"case comparison direct wind bits differ")
                else
                    maximum_fraction=max(maximum_fraction,_compare_number(left[index],right[index],bounds[index]))
                end
                changed_features+=reinterpret(UInt64,left[index])!=reinterpret(UInt64,right[index])
            end
        end
        if !lleft
            need(_zero(ll,la+24,8) && _zero(lr,la+24,8),"case comparison rejected target differs")
        else
            left=C._number(ll,la+24);right=C._number(lr,la+24)
            need(isfinite(left) && isfinite(right) && 0<=left<=RATE_LIMIT && 0<=right<=RATE_LIMIT,
                 "case comparison invalid target")
            bound=2*128U*min(RATE_LIMIT,2max(left,right))
            maximum_fraction=max(maximum_fraction,_compare_number(left,right,bound))
            changed_labels+=reinterpret(UInt64,left)!=reinterpret(UInt64,right)
        end
    end
    (;complete=true,scheduled=rows,features_accepted,labels_accepted,supervised,
      features_rejected=rows-features_accepted,labels_rejected=rows-labels_accepted,
      changed_feature_numbers=changed_features,changed_label_numbers=changed_labels,
      maximum_fraction_of_bound=maximum_fraction,policy_sha256=policy_digest(),
      source_provenance_qualified=false,actual_source_authority_granted=false,actual_fit_performed=false)
end
end
