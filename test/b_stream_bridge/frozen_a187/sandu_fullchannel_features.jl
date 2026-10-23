module SanduFullChannelFeatures

using Dates, SHA, Statistics

export RAW_NAMES, RAW_COLUMNS, HourlyInputs, FeatureBatch, read_author_hourly,
       forecast_clock, censor_fills, lag, trailing_stat, wasserstein1,
       completed_cdf, causal_convolution, trajectory_features, recovery_state,
       source_energy_input, wang2014_energy_input, build_features, features_at,
       raw_lag_features, SOURCE_COMMIT, IMPLEMENTATION_SHA256

const SOURCE_COMMIT = "5bb0b6adb3e77a6ade5f84d298589849862bd219"
const IMPLEMENTATION_SHA256 = bytes2hex(sha256(read(@__FILE__)))
const RAW_NAMES = ["Vector B Magnitude,nT", "Lat. Angle of B (GSE)",
    "Long. Angle of B (GSE)", "BX, nT (GSE, GSM)", "BY, nT (GSM)",
    "BZ, nT (GSM)", "Scalar B, nT", "SW Plasma Speed, km/s",
    "SW Proton Density, N/cm^3", "SW Plasma Temperature, K", "Plasma Beta",
    "Flow pressure", "E elecrtric field", "Magnetosonic Much num.",
    "Dst-index, nT", "AE-index, nT", "Alpha/Prot. ratio"]
const RAW_COLUMNS = [10,11,12,13,16,17,9,25,24,23,37,35,36,39,43,46,28]
const FILL_LIMITS = [999.,999.,999.,999.,999.,999.,999.,9999.,999.,9999999.,
    999.,99.99,999.,99.,99999.,9999.,9.]
const DROP_AFTER_PHYSICS = RAW_NAMES[[1,2,3,4,14,16]]
const ENERGY = "energy_loading_state"
const TAIL = "magnetotail_loading_state"
const COUPLING = "coupling_efficiency_state"

"Raw provider-hour tags and seventeen real channels, with explicit retrospective availability."
struct HourlyInputs
    time::Vector{DateTime}
    values::Matrix{Float64} # channel × hour, in RAW_NAMES order
    source_sha256::String
    availability_kind::Symbol
    function HourlyInputs(time, values; source_sha256="", availability_kind=:assumed_completed_hour)
        times = collect(DateTime, time); x = Matrix{Float64}(values)
        size(x)==(17,length(times)) || throw(ArgumentError("expected seventeen channels × hourly tags"))
        isempty(times) && throw(ArgumentError("empty hourly frame"))
        allunique(times) && issorted(times) || throw(ArgumentError("hour tags must be unique and ordered"))
        all(t->floor(t,Hour)==t,times) || throw(ArgumentError("raw tags must be exact hours"))
        all(diff(times).==Hour(1)) || throw(ArgumentError("calendar gaps must remain explicit missing rows"))
        all(v->isfinite(v)||isnan(v),x) || throw(ArgumentError("infinite raw channel"))
        availability_kind==:assumed_completed_hour || throw(ArgumentError(
            "this development frame supports explicit hypothetical completion only; measured receipts use features_at"))
        new(times,x,String(source_sha256),availability_kind)
    end
end

struct FeatureBatch
    origin::Vector{DateTime}
    issue::Vector{DateTime}
    names::Vector{String}
    matrix::Matrix{Float64} # row × feature
    variant::Symbol
    source_sha256::String
    availability_kind::Symbol
end

function _hour_add(time::DateTime, hours::Integer)
    hours isa Bool && throw(ArgumentError("Boolean is not a lead"))
    count=try Int64(hours) catch; throw(ArgumentError("hour count outside Int64")) end
    milliseconds=try Base.Checked.checked_mul(count,Int64(3_600_000)) catch
        throw(ArgumentError("hour duration overflows milliseconds"))
    end
    total=try Base.Checked.checked_add(Dates.value(time),milliseconds) catch
        throw(ArgumentError("timestamp addition overflows"))
    end
    DateTime(Dates.UTInstant{Millisecond}(Millisecond(total)))
end

"h is lead to the END of the target mean: raw o, issue o+1, target o+h, end issue+h."
function forecast_clock(origin::DateTime, horizon::Integer)
    horizon>0 || throw(ArgumentError("positive completed-target lead required"))
    floor(origin,Hour)==origin || throw(ArgumentError("origin must be an hour tag"))
    issue=_hour_add(origin,1); target=_hour_add(origin,horizon)
    (origin_tag_utc=origin,issue_time_utc=issue,target_record_tag_utc=target,
     target_hour_complete_utc=_hour_add(target,1),completed_target_lead_hours=Int(horizon))
end

"Parse a hash-bound immutable byte snapshot; reject future year before any physical field."
function read_author_hourly(path::AbstractString; expected_sha256::AbstractString)
    bytes=read(path); digest=bytes2hex(sha256(bytes))
    digest==expected_sha256 || throw(ArgumentError("author data SHA-256 mismatch"))
    rows=count(==(0x0a),bytes)+(isempty(bytes)||last(bytes)==0x0a ? 0 : 1)
    x=Matrix{Float64}(undef,17,rows);times=Vector{DateTime}(undef,rows)
    count_rows=0
    for line in eachline(IOBuffer(bytes))
        fields=split(line)
        isempty(fields) && throw(ArgumentError("blank source row"))
        year=parse(Int,fields[1])
        1<=year<=2025 || throw(ArgumentError("future/invalid year rejected before outcome parsing"))
        length(fields)==57 || throw(ArgumentError("author format requires exactly57 fields"))
        doy=parse(Int,fields[2]);hour=parse(Int,fields[3])
        1<=doy<=daysinyear(Date(year)) && 0<=hour<=23 || throw(ArgumentError("invalid raw time"))
        count_rows+=1;times[count_rows]=DateTime(year)+Day(doy-1)+Hour(hour)
        for (j,col) in enumerate(RAW_COLUMNS);x[j,count_rows]=parse(Float64,fields[col]);end
    end
    count_rows==rows || error("source line-count disagreement")
    HourlyInputs(times,x;source_sha256=digest)
end

"Causal fill censoring, without centered replacement or target imputation."
function censor_fills(values::AbstractMatrix)
    size(values,1)==17 || throw(ArgumentError("expected seventeen raw channels"))
    result=Matrix{Float64}(values)
    for j in axes(result,1), i in axes(result,2)
        value=result[j,i]
        (!isfinite(value)||value>=FILL_LIMITS[j]) && (result[j,i]=NaN)
    end
    result
end

function lag(values::AbstractVector, hours::Integer)
    hours>=0 || throw(ArgumentError("future lag forbidden"))
    result=fill(NaN,length(values))
    hours<length(values) && copyto!(result,Int(hours)+1,values,1,length(values)-Int(hours))
    result
end

function _heap_push!(heap::Vector{Float64},value::Float64,maximum_heap::Bool)
    push!(heap,value);i=length(heap)
    precedes(a,b)=maximum_heap ? a>b : a<b
    while i>1
        parent=i÷2;precedes(heap[i],heap[parent]) || break
        heap[i],heap[parent]=heap[parent],heap[i];i=parent
    end
end
function _heap_pop!(heap::Vector{Float64},maximum_heap::Bool)
    isempty(heap) && throw(ArgumentError("empty median heap"))
    result=first(heap);last_value=pop!(heap);isempty(heap) && return result
    heap[1]=last_value;i=1
    precedes(a,b)=maximum_heap ? a>b : a<b
    while 2i<=length(heap)
        child=2i
        child<length(heap) && precedes(heap[child+1],heap[child]) && (child+=1)
        precedes(heap[child],heap[i]) || break
        heap[i],heap[child]=heap[child],heap[i];i=child
    end
    result
end

function trailing_stat(values::AbstractVector, window::Integer, statistic::Symbol; min_valid=3)
    window>=1 && min_valid>=1 || throw(ArgumentError("invalid trailing window"))
    statistic in (:std,:median) || throw(ArgumentError("unknown trailing statistic"))
    output=fill(NaN,length(values));lower=Float64[];upper=Float64[]
    count=0;average=0.;m2=0.
    for i in eachindex(values)
        value=Float64(values[i])
        if isfinite(value)
            count+=1;delta=value-average;average+=delta/count;m2+=delta*(value-average)
            if statistic==:median
                if isempty(lower)||value<=first(lower);_heap_push!(lower,value,true)
                else;_heap_push!(upper,value,false);end
                length(lower)>length(upper)+1 && _heap_push!(upper,_heap_pop!(lower,true),false)
                length(upper)>length(lower) && _heap_push!(lower,_heap_pop!(upper,false),true)
            end
        end
        sample=filter(isfinite,@view values[max(1,i-window+1):i])
        if length(sample)>=min_valid
            output[i]=statistic==:std ? std(sample;corrected=true) : median(sample)
        elseif count>=min_valid
            # Literal pandas rolling.fillna(expanding) behavior also fills late gaps.
            output[i]=statistic==:std ? sqrt(max(0.,m2/(count-1))) :
                length(lower)==length(upper) ? .5*first(lower)+.5*first(upper) : first(lower)
        end
    end
    output
end

"Exact empirical one-dimensional Wasserstein distance, including unequal sample counts."
function wasserstein1(a::AbstractVector,b::AbstractVector)
    x=sort(filter(isfinite,a));y=sort(filter(isfinite,b))
    isempty(x)||isempty(y) ? throw(ArgumentError("empty empirical distribution")) : nothing
    grid=sort!(unique!(vcat(x,y)));length(grid)<2 && return 0.
    total=0.
    for k in 1:length(grid)-1
        fx=searchsortedlast(x,grid[k])/length(x);fy=searchsortedlast(y,grid[k])/length(y)
        total+=abs(fx-fy)*(grid[k+1]-grid[k])
    end
    total
end

"Only complete windows publish a distance; no future endpoint interpolation or partial tail."
function completed_cdf(values::AbstractVector,window::Integer,overlap::Real;min_valid=3)
    window>=1 && 0<=overlap<1 && min_valid>=1 || throw(ArgumentError("invalid CDF window"))
    step=max(1,trunc(Int,window*(1-overlap)));output=fill(NaN,length(values))
    previous=nothing;held=NaN
    for i in eachindex(values)
        if i>=window && (i-window)%step==0
            current=filter(isfinite,@view values[i-window+1:i])
            if length(current)>=min_valid
                previous===nothing || (held=wasserstein1(previous,current))
                previous=current
            end
        end
        output[i]=held
    end
    output
end

const KERNELS=Dict(
    "storm_onset_short"=>[-1.,-.5,0.,.5,1.]./2,
    "storm_onset_long"=>[-1.,-1.,-.5,-.2,0.,.2,.5,1.,1.,1.,1.]./6,
    "compression_short"=>[.2,1.,.2]./1.4,
    "compression_long"=>[.1,.2,.4,.8,1.,1.,.8,.4,.2,.1]./5,
    "enhancement_short"=>[.1,.3,.6,1.]./2,
    "enhancement_long"=>[.05,.1,.15,.2,.3,.4,.5,.6,.7,.8,.9,1.]./6)

function causal_convolution(values::AbstractVector,kernel::AbstractVector)
    isempty(kernel) && throw(ArgumentError("empty convolution kernel"))
    all(isfinite,kernel) || throw(ArgumentError("invalid convolution kernel"))
    output=fill(NaN,length(values))
    for i in eachindex(values)
        total=0.;valid=true
        for delay in 0:min(length(kernel)-1,i-1)
            value=values[i-delay]
            if !isfinite(value);valid=false;break;end
            total+=value*kernel[end-delay]
        end
        valid && (output[i]=isfinite(total) ? total : NaN)
    end
    output
end

function trajectory_features(values::AbstractVector,m::Integer,tau::Integer)
    m>=3 && tau>=1 || throw(ArgumentError("invalid trajectory embedding"))
    output=fill(NaN,length(values),3)
    for i in 1+(m-1)*tau:length(values)
        coordinates=[Float64(values[i-j*tau]) for j in 0:m-1]
        all(isfinite,coordinates) || continue
        points=[(coordinates[j],coordinates[j+1]) for j in 1:m-1]
        vectors=[(points[j+1][1]-points[j][1],points[j+1][2]-points[j][2]) for j in 1:m-2]
        distances=[hypot(v...) for v in vectors];path=sum(distances)
        direct=hypot(points[end][1]-points[1][1],points[end][2]-points[1][2])
        angles=Float64[]
        for j in 1:length(vectors)-1
            a,b=vectors[j],vectors[j+1];den=hypot(a...)*hypot(b...)
            cosine=(hypot(a...)>1e-10 && hypot(b...)>1e-10) ? (a[1]*b[1]+a[2]*b[2])/den : 0.
            push!(angles,acos(clamp(cosine,-1.,1.)))
        end
        output[i,:]=[maximum(distances),path>0 ? direct/path : 0.,isempty(angles) ? 0. : var(angles;corrected=false)]
    end
    output
end

"Original Sandu algebra, retaining negative-fractional-power NaN and its Bx/By dependence."
function source_energy_input(n::Real,v::Real,bx::Real,by::Real,bz::Real; absolute_sine=false)
    all(isfinite,(n,v,bx,by,bz)) || return NaN
    sine=sin(atan(Float64(by),Float64(bz))/2)
    sine=absolute_sine ? abs(sine) : sine
    sine<0 && return NaN
    3.78e7*max(Float64(n),1e-6)^.24*max(Float64(v),1.)^1.47*
        max(hypot(Float64(bx),Float64(by)),1e-6)^.86*(sine^2.70+.25)
end

"Wang2014 Eq13: n cm^-3, V km/s, transverse GSM field nT; result watts."
function wang2014_energy_input(n::Real,v::Real,by::Real,bz::Real)
    all(isfinite,(n,v,by,bz)) || return NaN
    n>=0 && v>=0 || return NaN
    bt=hypot(Float64(by),Float64(bz));bt==0 && return 0.
    angle=mod2pi(atan(Float64(by),Float64(bz)))
    3.78e7*Float64(n)^.24*Float64(v)^1.47*bt^.86*(max(0.,sin(angle/2))^2.70+.25)
end

function _accumulate(input::AbstractVector,tau::Real)
    tau>0 || throw(ArgumentError("positive memory time required"))
    alpha=-expm1(-1/Float64(tau));output=fill(NaN,length(input))
    for i in eachindex(input)
        isfinite(input[i]) || continue
        previous=i==1 ? NaN : output[i-1]
        output[i]=alpha*input[i]+(isfinite(previous) ? (1-alpha)*previous : 0.)
    end
    output
end

"Online confirmed minimum, preserving past emissions; thresholds/confirmation are declared adaptations."
function recovery_state(dst::AbstractVector,lookback::Integer,threshold::Real;
                        rise_nt=15.,confirmation_hours=2)
    lookback>0 && threshold<0 && rise_nt>0 && confirmation_hours>=1 || throw(ArgumentError("invalid recovery rule"))
    output=fill(NaN,length(dst));minimum_value=Inf;candidate=0;last_peak=0
    confirmed_minimum=Inf;counter=0;tracking=false
    for i in eachindex(dst)
        value=dst[i]
        if !isfinite(value);counter=0;tracking=false;minimum_value=Inf;candidate=0;confirmed_minimum=Inf;continue;end
        if value<threshold && !tracking && (last_peak==0 || value<confirmed_minimum || !isfinite(confirmed_minimum))
            tracking=true;minimum_value=value;candidate=i;counter=0
        elseif tracking && value<minimum_value
            minimum_value=value;candidate=i;counter=0
        end
        if tracking
            counter=value>=minimum_value+rise_nt ? counter+1 : 0
            if counter>=confirmation_hours
                last_peak=candidate;confirmed_minimum=minimum_value;tracking=false;counter=0
            end
        end
        # A pending minimum must survive crossing the storm threshold: a -55 nT
        # minimum cannot gain15 nT before crossing-50. Reset only after confirmation.
        if value>=threshold && !tracking
            minimum_value=Inf;candidate=0;counter=0;confirmed_minimum=Inf
        end
        if last_peak>0 && last_peak<i && i-last_peak<=lookback
            rate=i>=3 && all(isfinite,@view dst[i-2:i]) ?
                .5*(max(0.,dst[i]-dst[i-1])+max(0.,dst[i-1]-dst[i-2])) : 0.
            output[i]=min((i-last_peak)/24.,1.)*(1+rate)
        end
    end
    output
end

mutable struct _Columns
    names::Vector{String}
    vectors::Vector{Vector{Float64}}
    index::Dict{String,Int}
end
_Columns()=_Columns(String[],Vector{Float64}[],Dict{String,Int}())
_get(c::_Columns,name)=c.vectors[c.index[name]]
function _put!(c::_Columns,name,values)
    vector=collect(Float64,values)
    if haskey(c.index,name);c.vectors[c.index[name]]=vector
    else;push!(c.names,name);push!(c.vectors,vector);c.index[name]=length(c.names);end
    c
end
function _drop!(c::_Columns,names)
    keep=findall(n->!(n in names),c.names)
    c.names=c.names[keep];c.vectors=c.vectors[keep];c.index=Dict(n=>i for (i,n) in enumerate(c.names));c
end
function _lags!(c::_Columns,name,lags)
    for h in lags;_put!(c,h==0 ? name : name*"_lag_$(h)",lag(_get(c,name),h));end
end

const TRANSFORMS=Dict(
    RAW_NAMES[6]=>[(:log,0.,0.)],RAW_NAMES[5]=>[(:log,0.,0.)],RAW_NAMES[7]=>[(:log,0.,0.)],
    RAW_NAMES[8]=>[(:sas,.1,.5)],RAW_NAMES[9]=>[(:log,0.,0.)],
    RAW_NAMES[10]=>[(:log,0.,0.),(:sas,0.,.4)],
    RAW_NAMES[11]=>[(:log,0.,0.),(:sas,0.,.5)],
    RAW_NAMES[12]=>[(:log,0.,0.),(:sas,0.,.5)],RAW_NAMES[13]=>[(:sas,.1,.5)],
    RAW_NAMES[17]=>[(:sas,.1,.5),(:log,0.,0.)],
    ENERGY=>[(:sas,.1,.5),(:log,0.,0.)],TAIL=>[(:sas,.1,.5),(:log,0.,0.)],
    COUPLING=>[(:sas,.1,.5),(:log,0.,0.)])

function _transform(value,steps)
    isfinite(value) || return NaN
    result=Float64(value)
    for (kind,epsilon,delta) in steps
        result=kind==:log ? sign(result)*log1p(abs(result)) :
            sinh(delta*asinh(clamp(result,-1e100,1e100))-epsilon)
    end
    isfinite(result) ? result : NaN
end

"Preserve the actual source's repeated global substring passes, not its unscaled-data comments."
function _transform_pass!(c::_Columns)
    bases=sort(collect(keys(TRANSFORMS));by=length,rev=true)
    for name in c.names
        base=findfirst(b->occursin(b,name),bases)
        base===nothing || _put!(c,name,[_transform(v,TRANSFORMS[bases[base]]) for v in _get(c,name)])
    end
end

function _rolling!(c::_Columns,name,window,overlap)
    std_name=name*"_std_$(window)r";cdf_name=name*"_cdf_wass_trend_$(window)r"
    median_name=name*"_rolling_median_$(window)r"
    values=_get(c,name)
    _put!(c,std_name,trailing_stat(values,window,:std))
    _put!(c,cdf_name,completed_cdf(values,window,overlap))
    _put!(c,median_name,trailing_stat(values,window,:median))
    [std_name,cdf_name,median_name]
end
function _domain_rolling!(c::_Columns,names,window,overlap,lags)
    derived=String[]
    for name in names;append!(derived,_rolling!(c,name,window,overlap));end
    for name in derived;_lags!(c,name,lags);end
end
function _trajectory!(c::_Columns,name,m,tau)
    values=trajectory_features(_get(c,name),m,tau)
    for (j,suffix) in enumerate(("jump_mag","path_eff","turn_var"))
        _put!(c,name*"_embedding_2d_"*suffix,values[:,j])
    end
end
function _convolve!(c::_Columns,name,kind)
    sanitized=replace(name," "=>"_",","=>"","("=>"",")"=>"")
    _put!(c,"conv_"*kind*"_"*sanitized,causal_convolution(_get(c,name),KERNELS[kind]))
end

function _prepared(input::HourlyInputs,imputed)
    clean=censor_fills(input.values)
    imputed===nothing && return clean
    result=Matrix{Float64}(imputed)
    size(result)==size(clean) || throw(ArgumentError("imputed shape differs"))
    all(v->isfinite(v)||isnan(v),result) || throw(ArgumentError("invalid imputed value"))
    for j in (15,16,17)
        isequal(result[j,:],clean[j,:]) || throw(ArgumentError("Dst/AE/alpha are excluded from imputation"))
    end
    for j in axes(clean,1),i in axes(clean,2)
        isfinite(clean[j,i]) && result[j,i]!=clean[j,i] && throw(ArgumentError("imputation may only replace missing predictor cells"))
    end
    result
end

"Full-channel causal adaptation. This API does not claim original saved-model or offline-source parity."
function build_features(input::HourlyInputs;variant::Symbol=:causal_primaryphysics_v2,imputed=nothing)
    variant in (:causal_sourcephysics_v1,:causal_primaryphysics_v2) || throw(ArgumentError("unknown causal variant"))
    raw=_prepared(input,imputed);n=size(raw,2);c=_Columns()
    for (j,name) in enumerate(RAW_NAMES);_put!(c,name,raw[j,:]);end
    # Three source lookbacks, with a separately declared prefix-confirmed detector.
    for (lookback,threshold) in ((8,-50),(24,-100),(48,-150))
        name="recovery_state_lookback_$(lookback)_thresh_$(threshold)"
        _put!(c,name,recovery_state(raw[15,:],lookback,threshold));_lags!(c,name,(0,4,8))
    end
    energy=fill(NaN,n);coupling=fill(NaN,n);tail=fill(NaN,n)
    for i in 1:n
        density,speed,bx,by,bz=raw[9,i],raw[8,i],raw[4,i],raw[5,i],raw[6,i]
        if variant==:causal_sourcephysics_v1
            energy[i]=source_energy_input(density,speed,bx,by,bz)
            denominator=source_energy_input(density,speed,bx,by,bz;absolute_sine=true)
        else
            energy[i]=wang2014_energy_input(density,speed,by,bz)
            denominator=energy[i]
        end
        if all(isfinite,(raw[15,i],raw[16,i],denominator))
            coupling[i]=(abs(raw[15,i])+raw[16,i])/(denominator+1.)
        end
        all(isfinite,(density,speed,bz)) &&
            (tail[i]=max(0.,-bz+3.)*max(speed,1.)*max(density,1e-6)*1e-4)
    end
    _put!(c,ENERGY,_accumulate(energy,8.5));_put!(c,TAIL,_accumulate(tail,2.));_put!(c,COUPLING,coupling)
    _transform_pass!(c)
    for name in (ENERGY,TAIL,COUPLING);_lags!(c,name,(0,4,8));end
    _drop!(c,DROP_AFTER_PHYSICS)
    # Source call order is significant: each global transform pass also revisits previous domains.
    _domain_rolling!(c,RAW_NAMES[[8,10,9,11]],24,0.,(0,8,12))
    for (name,m,tau) in zip(RAW_NAMES[[8,10,9,11]],(5,6,5,6),(1,3,2,2));_trajectory!(c,name,m,tau);end
    for (name,kind) in ((RAW_NAMES[8],"enhancement_short"),(RAW_NAMES[8],"enhancement_long"),
            (RAW_NAMES[9],"compression_short"),(RAW_NAMES[9],"compression_long"),
            (RAW_NAMES[10],"compression_short"),(RAW_NAMES[10],"enhancement_long"));_convolve!(c,name,kind);end
    _transform_pass!(c)
    for name in RAW_NAMES[[11,9,10,8]];_lags!(c,name,(0,8,12));end
    _domain_rolling!(c,RAW_NAMES[[6,7]],12,.3,(0,4,8))
    for name in RAW_NAMES[[6,7]];_trajectory!(c,name,4,1);end
    for (name,kind) in ((RAW_NAMES[6],"storm_onset_short"),(RAW_NAMES[6],"storm_onset_long"),
            (RAW_NAMES[7],"compression_short"),(RAW_NAMES[7],"enhancement_long"));_convolve!(c,name,kind);end
    _transform_pass!(c)
    for name in RAW_NAMES[[6,5,7]];_lags!(c,name,(0,4,8));end
    _domain_rolling!(c,RAW_NAMES[[12,13,17]],24,.3,(0,4,8))
    for (name,m,tau) in zip(RAW_NAMES[[12,13,17]],(5,4,5),(2,1,2));_trajectory!(c,name,m,tau);end
    for kind in ("compression_short","compression_long");_convolve!(c,RAW_NAMES[12],kind);end
    _transform_pass!(c)
    for name in RAW_NAMES[[12,13]];_lags!(c,name,(0,4,8));end
    _lags!(c,RAW_NAMES[17],(0,8,12))
    _put!(c,RAW_NAMES[15]*"_current",raw[15,:])
    for h in 1:4;_put!(c,RAW_NAMES[15]*"_past_$(h)h",lag(raw[15,:],h));end
    _drop!(c,[RAW_NAMES[15]]) # the source's target column is not a predictor
    allunique(c.names) || error("duplicate engineered feature")
    matrix=reduce(hcat,c.vectors)
    all(v->isfinite(v)||isnan(v),matrix) || error("infinite engineered feature")
    FeatureBatch(copy(input.time),[_hour_add(t,1) for t in input.time],copy(c.names),matrix,
                 variant,input.source_sha256,input.availability_kind)
end

"An independent prefix evaluation route, optionally gated by explicitly supplied receipt clocks."
function features_at(input::HourlyInputs,issue::DateTime;receipts=nothing,variant=:causal_primaryphysics_v2)
    floor(issue,Hour)==issue || throw(ArgumentError("development issue must be an exact hour"))
    count=searchsortedlast(input.time,_hour_add(issue,-1));count>0 || throw(ArgumentError("no completed input hour"))
    input.time[count]==_hour_add(issue,-1) || throw(ArgumentError(
        "issue requires its explicit preceding raw-hour row; materialize missing hours rather than relabel stale input"))
    raw=copy(input.values[:,1:count])
    if receipts!==nothing
        size(receipts)==size(input.values) || throw(ArgumentError("receipt shape differs"))
        eltype(receipts)<:DateTime || throw(ArgumentError("receipt clocks must be explicit UTC DateTime values"))
        for j in axes(raw,1),i in axes(raw,2);receipts[j,i]<=issue || (raw[j,i]=NaN);end
    end
    prefix=HourlyInputs(input.time[1:count],raw;source_sha256=input.source_sha256)
    batch=build_features(prefix;variant=variant)
    (origin=batch.origin[end],issue=issue,names=batch.names,values=copy(batch.matrix[end,:]),variant=variant,
     receipt_policy=receipts===nothing ? :assumed_completed_hour : :provided_receipt_times)
end

function raw_lag_features(input::HourlyInputs;lags=(0,1,2,3,4,8,12,24),imputed=nothing)
    length(unique(lags))==length(lags) && all(>=(0),lags) || throw(ArgumentError("invalid raw-lag schema"))
    raw=_prepared(input,imputed);names=String[];columns=Vector{Float64}[]
    for (j,name) in enumerate(RAW_NAMES),h in lags
        push!(names,name*"_raw_lag_$(h)");push!(columns,lag(raw[j,:],h))
    end
    FeatureBatch(copy(input.time),[_hour_add(t,1) for t in input.time],names,reduce(hcat,columns),
        :fullchannel_xgb_lags_v1,input.source_sha256,input.availability_kind)
end

end # module
