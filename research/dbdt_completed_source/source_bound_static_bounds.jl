"""Four locked static bounds on generated source-bound point/calibration inputs.

The Gaussian uncertainty parameter fit is distinct from score calibration.
This entry grants no actual-data authority or scientific coverage qualification.
"""
module SourceBoundStaticBounds
using Dates,SHA,Statistics,LinearAlgebra,SolarSINDy
import ..SourceBoundCalibration
const K=SourceBoundCalibration
const F=SolarSINDy.ArchiveSourceBoundPointFit
const Q=SolarSINDy.ArchiveCatalogCaseJob
const C=SolarSINDy.ArchiveCatalogCases
const J=SolarSINDy.ArchiveCatalogSourceJob
const D=SolarSINDy.CompletedSourceDBDT
const R=D.CompletedRefitArtifacts
const NAME="research/dbdt_completed_source/source_bound_static_bounds.jl"
const PIN=bytes2hex(sha256(read(@__FILE__)))
const MAGIC=Vector{UInt8}(codeunits("DBDT_STATIC_BOUNDS_V1\0"))
const METHODS=(:het_gaussian,:raw_split,:normalized_split,:regime_split)
const CONTEXT_FIELDS=Set(("schema","evidence_mode","runtime","job_sha256","point_context_sha256","point_model_sha256",
    "point_receipt_sha256","point_source_sha256","calibration_context_sha256","calibration_input_sha256",
    "case_request_sha256","case_manifest_sha256","case_complete_sha256","table_index","role","layout","table",
    "training_input_sha256","training_rows","calibration_rows","calibration_stop_exclusive","first_evaluation_issue",
    "numeric_sha256","regime_bin_counts","empty_regime_bins","convention"))
need(ok,message)=ok || throw(ArgumentError(message))
digest(bytes)=bytes2hex(sha256(bytes))
export StaticCalibration,ExpectedStatic,calibrate_generated,predict_upper,check_calibration
struct _Token end
struct ExpectedStatic
    context_sha256::String
    numeric_sha256::String
    function ExpectedStatic(context,numeric)
        new(J._sha(context),J._sha(numeric))
    end
end
struct StaticCalibration
    context::String
    numeric::String
    receipt::String
    function StaticCalibration(context,numeric,receipt,::_Token)
        owned=numeric isa AbstractString ? String(copy(codeunits(numeric))) : String(copy(numeric))
        new(String(context),owned,String(receipt))
    end
end
function job_identity()
    need(digest(read(@__FILE__))==PIN && J._same(realpath(@__FILE__),joinpath(realpath(pkgdir(SolarSINDy)),split(NAME,'/')...)),
        "static bound job outside unchanged loaded package")
    PIN
end
function _quantile(sorted,alpha)
    need(!isempty(sorted) && issorted(sorted) && all(isfinite,sorted),"static quantile finite sorted scores")
    n=length(sorted);k=ceil(Int,(n+1)*(1-alpha))
    k>n ? Inf : k<1 ? -Inf : sorted[k]
end
# Retained Acklam polynomial from the locked comparator. Endpoint extensions
# handle binary64 rounding of1-alpha;interior polynomial arithmetic is unchanged.
function _normal_quantile(p)
    need(isfinite(p) && 0<=p<=1,"normal probability domain")
    p==0 && return -Inf;p==1 && return Inf
    a=(-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00)
    b=(-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01)
    c=(-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00)
    d=(7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00);pl=0.02425
    if p<pl;q=sqrt(-2log(p));return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6])/((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    elseif p<=1-pl;q=p-0.5;r=q*q;return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q/(((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
    else;q=sqrt(-2log(1-p));return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6])/((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1);end
end
binof(edges,x)=clamp(searchsortedlast(edges,x),1,4)
function _fit_numeric(train::AbstractMatrix{Float64},features::AbstractMatrix{Float64},residuals::AbstractVector{Float64},
                      scores::AbstractVector{Float64},peaks::AbstractVector{Float64},alpha)
    Base.require_one_based_indexing(train,features,residuals,scores,peaks)
    n,p=size(features)
    need(1<=p<=11 && size(train,2)==p && 2<=size(train,1)<=600000 && 2<=n<=600000 &&
        length(residuals)==length(scores)==length(peaks)==n,"static calibration dimensions")
    need(all(isfinite,train) && all(isfinite,features) && all(isfinite,residuals) && all(isfinite,scores) &&
        all(x->isfinite(x)&&x>=0,peaks),"static calibration numeric domain")
    alpha=Float64(D.nominal_value(alpha));need(0<alpha<1,"static nominal must be interior")
    means=vec(mean(train;dims=1));scales=vec(std(train;dims=1)).+1e-9
    need(all(isfinite,means) && all(x->isfinite(x)&&x>0,scales),"static training standardizer")
    Z=hcat(ones(n),(features.-means')./scales')
    need(all(isfinite,Z),"static standardized calibration overflow")
    # The same penalty1 objective,including intercept. Forming Z'Z can round
    # away the unit penalty when a constant training feature shifts at calibration.
    # An exactlyzero design column contributes only its ridge penalty. Its
    # unique coefficient is exactlyzero,including when that feature later
    # changes at evaluation. Remove these columns before the numerical solve
    # so roundoff cannot create a spurious coefficient on an inactive feature.
    active=[j for j in axes(Z,2) if any(x->!iszero(x),view(Z,:,j))]
    weights=zeros(p+1)
    augmented=vcat(Z[:,active],Matrix{Float64}(I,length(active),length(active)))
    weights[active]=qr(augmented)\vcat(abs.(residuals),zeros(length(active)))
    need(all(isfinite,weights),"Gaussian uncertainty ridge overflow")
    raw=_quantile(sort(residuals),alpha);normalized=_quantile(sort(scores),alpha)
    edges=quantile(peaks,[0.,.5,.8,.95,1.]);bins=binof.(Ref(edges),peaks)
    counts=[count(==(b),bins) for b in 1:4]
    quantiles=[counts[b]==0 ? raw : _quantile(sort(residuals[bins.==b]),alpha) for b in 1:4]
    (;means,scales,weights,raw,normalized,edges,quantiles,alpha,normal=_normal_quantile(1-alpha),counts)
end
function _encode(v)
    io=IOBuffer();write(io,MAGIC,htol(UInt32(length(v.means))))
    for x in vcat(v.means,v.scales,v.weights,[v.raw,v.normalized],v.edges,v.quantiles,[v.alpha,v.normal])
        write(io,htol(reinterpret(UInt64,Float64(x))))
    end
    take!(io)
end
function _decode(bytes)
    need(bytes isa Vector{UInt8} && length(bytes)<=4096 && length(bytes)>=length(MAGIC)+4 && bytes[1:length(MAGIC)]==MAGIC,"static numeric frame")
    io=IOBuffer(bytes);skip(io,length(MAGIC));p=Int(ltoh(read(io,UInt32)))
    need(1<=p<=11 && length(bytes)==length(MAGIC)+4+8*(3p+14),"static numeric geometry")
    take(n)=[reinterpret(Float64,ltoh(read(io,UInt64))) for _ in 1:n]
    means=take(p);scales=take(p);weights=take(p+1);raw,normalized=take(2);edges=take(5);quantiles=take(4);alpha,normal=take(2)
    need(eof(io) && all(isfinite,means) && all(x->isfinite(x)&&x>0,scales) && all(isfinite,weights) &&
        all(x->isfinite(x)&&x>=0,edges) && issorted(edges) && 0<alpha<1 &&
        all(x->!isnan(x)&&x!=-Inf,vcat([raw,normalized],quantiles)) && !isnan(normal),"static numeric values")
    need(normal===_normal_quantile(1-alpha),"static Gaussian probability differs")
    (;means,scales,weights,raw,normalized,edges,quantiles,alpha,normal)
end
function _case_binding(value,index,point,kc)
    need(value.actual_source_authority_granted===false && value.actual_fit_performed===false,"static generated entry rejects actual case authority")
    pc=F._check_capsule(point);need(!pc.actual && !kc.actual,"static generated entry rejects actual fitted inputs")
    need(F.isint(index) && 1<=index<=length(value.features) && kc.context["table_index"]==index &&
        pc.context["table_index"]==index && pc.context["case_request_sha256"]==value.request_sha256 &&
        pc.context["case_manifest_sha256"]==value.manifest_sha256 && pc.context["case_complete_sha256"]==value.complete_sha256,
        "static calibration case/model binding differs")
    pc
end
function _raw_peak(ground,issue,policy)
    anchor=D.CMR.issue_timing(issue,policy).latest_ground_center
    need(D.whole(issue) && anchor<issue,"static peak feature clock")
    peak=0.
    for k in 29:-1:0
        tag=anchor-Minute(k);before=C._locate(ground,tag-Minute(1));after=C._locate(ground,tag)
        need(before!==nothing && after!==nothing && before.valid_mask==after.valid_mask==UInt8(3),"static peak source support missing")
        rate=hypot(after.values[1]-before.values[1],after.values[2]-before.values[2])
        need(isfinite(rate) && rate>=0,"static peak source rate invalid");peak=max(peak,rate)
    end
    peak
end
"""Compute locked values after the caller has admitted matching case/model inputs."""
function _calibration_values(value::Q.VerifiedCases,index,point,pc,kc;nominal)
    c=kc.context;model=pc.context
    cal=Q.split_arrays(value,index;start=F._fit_clock(c["start_inclusive"]),stop=F._fit_clock(c["stop_exclusive"]),asof=F._fit_clock(c["target_asof"]))
    input,_=F._data_sha(cal)
    need(input==c["input_sha256"] && cal.rows==kc.arrays.rows && cal.issues==kc.arrays.issues &&
        cal.targets==kc.arrays.targets && cal.scales==kc.arrays.scales,"static calibration cohort differs")
    train=Q.split_arrays(value,index;start=F._fit_clock(model["start_inclusive"]),stop=F._fit_clock(model["stop_exclusive"]),asof=F._fit_clock(model["target_asof"]))
    train_pin,_=F._data_sha(train);need(train_pin==model["input_sha256"],"static standardizer point-training cohort differs")
    features=value.features[index];ground=getproperty(value.source.catalogs,Symbol("ground-"*features.station))
    wind=features.mode===:mixed8 ? getproperty(value.source.catalogs,Symbol("wind-"*features.station)) : nothing
    need(C._pins(ground,wind,features.mode)==features.source_pins,"static source catalog differs")
    peaks=Float64[]
    for row in cal.rows
        saved=C.feature_row(features,row);checked=C._prepare(ground,saved.issue;policy=features.policy,mode=features.mode,wind)
        need(checked.accepted && checked.features==saved.features && saved.scale==1+checked.ground11[3],
            "static calibration feature/source support differs")
        peak=_raw_peak(ground,saved.issue,features.policy)
        need(log1p(peak)==checked.ground11[3],"static raw peak differs from source-bound feature")
        push!(peaks,peak)
    end
    residuals=log1p.(kc.arrays.targets).-kc.arrays.predictions
    need(residuals./kc.arrays.scales==kc.arrays.scores,"static signed score definition differs")
    numeric=_fit_numeric(train.features,cal.features,residuals,kc.arrays.scores,peaks,nominal);bytes=_encode(numeric)
    (;numeric,bytes,training_input_sha256=train_pin,training_rows=length(train.rows),calibration_rows=length(cal.rows))
end
"""Fit the locked uncertainty comparisons on generated calibration rows only.

All four methods use the same fixed point model and score cohort. Actual-source
uncertainty fitting requires a distinct qualified authority not granted here.
"""
function calibrate_generated(value::Q.VerifiedCases,index,point,calibration,expected;nominal=.1)
    kc=K._checked(calibration,point,expected);pc=_case_binding(value,index,point,kc)
    values=_calibration_values(value,index,point,pc,kc;nominal)
    numeric=values.numeric;bytes=values.bytes;c=kc.context
    context=Dict("schema"=>"generated_source_bound_static_calibration_v1","evidence_mode"=>"generated","runtime"=>string(VERSION),"job_sha256"=>job_identity(),
        "point_context_sha256"=>point.expected.context_sha256,"point_model_sha256"=>c["point_model_sha256"],"point_receipt_sha256"=>c["point_receipt_sha256"],
        "point_source_sha256"=>point.expected.source_sha256,"calibration_context_sha256"=>expected.context_sha256,"calibration_input_sha256"=>expected.input_sha256,
        "case_request_sha256"=>value.request_sha256,"case_manifest_sha256"=>value.manifest_sha256,"case_complete_sha256"=>value.complete_sha256,
        "table_index"=>Int(index),"role"=>c["role"],"layout"=>c["layout"],"table"=>c["table"],"training_input_sha256"=>values.training_input_sha256,
        "training_rows"=>values.training_rows,"calibration_rows"=>values.calibration_rows,"calibration_stop_exclusive"=>c["stop_exclusive"],
        "first_evaluation_issue"=>c["first_evaluation_issue"],"numeric_sha256"=>digest(bytes),"regime_bin_counts"=>numeric.counts,
        "empty_regime_bins"=>findall(==(0),numeric.counts),"convention"=>"locked_heteroGaussian_raw_normalized_fixedCalibrationEdge_split_v1")
    context_bytes=R.canonical_bytes(context;limit=32768)
    receipt=R.canonical_bytes(Dict("schema"=>"generated_static_calibration_receipt_v1","complete"=>true,"context_sha256"=>digest(context_bytes),
        "numeric_sha256"=>digest(bytes),"generated_uncertainty_parameter_fits"=>1,"actual_source_access"=>false,"actual_point_model_fits"=>0,
        "software_qualification"=>false,"scientific_coverage_qualified"=>false);limit=4096)
    # String(Vector{UInt8}) consumes its buffer; bind hashes before transfer.
    static_expected=ExpectedStatic(digest(context_bytes),digest(bytes))
    (;capsule=StaticCalibration(String(context_bytes),String(bytes),String(receipt),_Token()),
      expected=static_expected)
end
function check_calibration(value::StaticCalibration,point,calibration,expected;expected_static::ExpectedStatic)
    kc=K._checked(calibration,point,expected);pc=F._check_capsule(point)
    need(!kc.actual && !pc.actual,"static generated capsule rejects actual fitted inputs")
    cb=Vector{UInt8}(codeunits(value.context));rb=Vector{UInt8}(codeunits(value.receipt));bytes=Vector{UInt8}(codeunits(value.numeric))
    need(digest(cb)==expected_static.context_sha256 && digest(bytes)==expected_static.numeric_sha256,"static external expected pins differ")
    c=R.decode_canonical(cb;limit=32768);r=R.decode_canonical(rb;limit=4096);J._object(c,CONTEXT_FIELDS)
    need(c["schema"]=="generated_source_bound_static_calibration_v1" && c["evidence_mode"]=="generated" && c["runtime"]==string(VERSION) &&
        c["job_sha256"]==job_identity() && c["point_context_sha256"]==point.expected.context_sha256 &&
        c["point_source_sha256"]==point.expected.source_sha256 && c["calibration_context_sha256"]==expected.context_sha256 &&
        c["calibration_input_sha256"]==expected.input_sha256 && c["training_input_sha256"]==pc.context["input_sha256"] &&
        c["numeric_sha256"]==digest(bytes),"static capsule external bindings differ")
    for key in ("point_model_sha256","point_receipt_sha256","case_request_sha256","case_manifest_sha256","case_complete_sha256","table_index","role","layout","table")
        need(c[key]==kc.context[key],"static point/calibration case identity differs")
    end
    need(c["calibration_stop_exclusive"]==kc.context["stop_exclusive"] && c["first_evaluation_issue"]==kc.context["first_evaluation_issue"] &&
        F.isint(c["training_rows"]) && 2<=c["training_rows"]<=600000 && F.isint(c["calibration_rows"]) && c["calibration_rows"]==kc.context["rows"] &&
        c["convention"]=="locked_heteroGaussian_raw_normalized_fixedCalibrationEdge_split_v1","static cohort/convention differs")
    counts=c["regime_bin_counts"];need(counts isa Vector && length(counts)==4 && all(x->F.isint(x)&&0<=x<=c["calibration_rows"],counts) &&
        sum(counts)==c["calibration_rows"] && c["empty_regime_bins"]==findall(==(0),counts),"static regime counts differ")
    J._object(r,Set(("schema","complete","context_sha256","numeric_sha256","generated_uncertainty_parameter_fits","actual_source_access",
        "actual_point_model_fits","software_qualification","scientific_coverage_qualified")))
    need(r["schema"]=="generated_static_calibration_receipt_v1" && r["complete"]===true && r["context_sha256"]==digest(cb) &&
        r["numeric_sha256"]==digest(bytes) && F.isint(r["generated_uncertainty_parameter_fits"]) && r["generated_uncertainty_parameter_fits"]==1 &&
        r["actual_source_access"]===false && F.isint(r["actual_point_model_fits"]) && r["actual_point_model_fits"]==0 &&
        r["software_qualification"]===false && r["scientific_coverage_qualified"]===false,"static receipt differs")
    numeric=_decode(bytes);need(length(numeric.means)==(c["layout"]=="ground-only" ? 11 : 8),"static layout width differs")
    (;context=c,numeric,context_sha256=digest(cb),numeric_sha256=digest(bytes))
end
function _predict(v,X::AbstractMatrix{Float64},centers::AbstractVector{Float64},scales::AbstractVector{Float64},peaks::AbstractVector{Float64})
    Base.require_one_based_indexing(X,centers,scales,peaks);n,p=size(X)
    need(0<=n<=600000 && p==length(v.means) && length(centers)==length(scales)==length(peaks)==n,"static prediction dimensions")
    need(all(isfinite,X) && all(isfinite,centers) && all(x->isfinite(x)&&x>=1,scales) && all(x->isfinite(x)&&x>=0,peaks),"static prediction domain")
    Z=hcat(ones(n),(X.-v.means')./v.scales');need(all(isfinite,Z),"static prediction standardization overflow")
    sigma=max.(sqrt(pi/2).*(Z*v.weights),1e-3);need(all(isfinite,sigma),"static Gaussian scale overflow")
    bounds=(het_gaussian=expm1.(centers.+v.normal.*sigma),raw_split=expm1.(centers.+v.raw),
        normalized_split=expm1.(centers.+v.normalized.*scales),regime_split=expm1.([centers[i]+v.quantiles[binof(v.edges,peaks[i])] for i in 1:n]))
    need(all(x->all(y->!isnan(y),x),values(bounds)),"static bound became NaN")
    counts=map(x->(;issued=length(x),full=count(==(Inf),x),empty=count(y->y<0,x),nonfinite=count(y->!isfinite(y),x)),bounds)
    (;bounds,counts,all_bounds_finite_nonempty=all(c->c.nonfinite==0 && c.empty==0,values(counts)))
end
"""Issue all four fixed bounds without consuming evaluation targets or labels."""
function predict_upper(value::StaticCalibration,X,centers,scales,peaks;point,calibration,expected,expected_static,start)
    v=check_calibration(value,point,calibration,expected;expected_static)
    need(start isa DateTime && D.whole(start) && F._fit_clock(v.context["calibration_stop_exclusive"])<=start &&
        F._fit_clock(v.context["first_evaluation_issue"])<=start,"static replay precedes frozen evaluation")
    result=_predict(v.numeric,X,centers,scales,peaks)
    (;result...,methods=METHODS,context_sha256=v.context_sha256,numeric_sha256=v.numeric_sha256,
        calibration_rows=v.context["calibration_rows"],regime_bin_counts=copy(v.context["regime_bin_counts"]),
        empty_regime_bins=copy(v.context["empty_regime_bins"]),actual_source_authority_granted=false,
        actual_point_model_fits=0,scientific_coverage_qualified=false)
end
end
