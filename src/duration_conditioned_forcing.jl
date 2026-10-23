"""Research-only duration-conditioned forcing with exact path masses and A interval rollouts."""
module DurationConditionedForcing

using Dates, SHA, JSON3, LinearAlgebra
using ..IntervalMeanForecaster
const A = IntervalMeanForecaster
const K = A.DstIntervalAdapter.Kernel
const R = A.DstIntervalAdapter.Replay
const LOADED_SELF = bytes2hex(sha256(read(@__FILE__)))
const LOADED_A = A.source_identity()
const Cells = NTuple{49,NTuple{5,Int}}
const Basis = NTuple{2,NTuple{2,Float64}}
const HOUR_MS = Int64(3_600_000)
const MAX_HOURS = 500_000
const MAX_ARTIFACT_BYTES = 2_097_152
export WindHour, DurationModel, fit_duration, current_state, hazard,
       affine_rollout, evaluate_rollout, save_model, load_model, source_identity

_need(ok, message) = ok || throw(ArgumentError(message))
_sha(bytes) = bytes2hex(sha256(bytes))
_hex(x) = x isa AbstractString && occursin(r"\A[0-9a-f]{64}\z",x)
_json(x) = Vector{UInt8}(codeunits(JSON3.write(x)))
_hash(x) = _sha(_json(x))
function _shift(t::DateTime, hours::Integer)
    value = Int128(Dates.value(t)) + Int128(hours)*HOUR_MS
    typemin(Int64)<=value<=typemax(Int64) || throw(OverflowError("hour shift exceeds DateTime"))
    DateTime(Dates.UTInstant{Millisecond}(Millisecond(Int64(value))))
end

"""Verify this loaded source and all eight unchanged A implementation dependencies."""
function source_identity()
    _need(_sha(read(@__FILE__))==LOADED_SELF,"loaded duration source changed")
    _need(A.source_identity()==LOADED_A,"loaded A dependencies changed")
    (LOADED_A...,"duration_conditioned_forcing.jl"=>LOADED_SELF)
end
_source_signature() = _hash([(name=first(p),sha256=last(p)) for p in source_identity()])

"""One tagged hourly wind input with explicit completion/assumed receipt and supplied q.
NaN marks a missing Bz or q. No measured receipt or raw-source verification is inferred.
"""
struct WindHour
    tag::DateTime
    completed::DateTime
    received::DateTime
    bz::Float64
    q::Float64
    function WindHour(tag::DateTime,completed::DateTime,received::DateTime,bz::Real,q::Real)
        _need(mod(Dates.value(tag),HOUR_MS)==0 && completed==_shift(tag,1),"exact tagged hourly completion required")
        _need(received>=completed,"receipt predates source completion")
        _need(!(bz isa Bool) && !(q isa Bool),"Boolean wind value")
        b,z=Float64(bz),Float64(q)
        _need((isfinite(b)||isnan(b)) && (isfinite(z)||isnan(z)),"infinite wind value")
        _need(isnan(z)||z>=0,"negative injection")
        _need((bz==0||isnan(bz)||b!=0) && (q==0||isnan(q)||z!=0),"wind conversion underflow")
        new(tag,completed,received,b,z)
    end
end
function _known(row::WindHour)
    isfinite(row.bz) && isfinite(row.q) || return false
    _need(row.bz<0 || row.q==0,"North injection must be zero")
    true
end
_state(row::WindHour) = row.bz<0 ? :S : :N
function _ordered(rows)
    _need(length(rows)<=MAX_HOURS && all(x->x isa WindHour,rows),"bounded WindHour sequence required")
    _need(issorted(getproperty.(rows,:tag)) && allunique(getproperty.(rows,:tag)),"ordered unique source hours required")
end
function _trace(rows)
    out=Tuple{Symbol,Int}[];previous=nothing;previous_state=:missing;age=0;run=0
    for row in rows
        if !_known(row)
            push!(out,(:missing,0));previous=nothing;previous_state=:missing;age=0;run=0
            continue
        end
        state=_state(row)
        contiguous=previous!==nothing && row.tag==_shift(previous,1)
        if contiguous && state==previous_state
            run+=1;age=age>0 ? min(age+1,49) : run>=49 ? 49 : 0
        elseif contiguous
            run=1;age=1
        else
            run=1;age=0
        end
        push!(out,(state,age));previous=row.tag;previous_state=state
    end
    out
end
function _prefix_digest(rows)
    context=SHA.SHA2_256_CTX()
    for row in rows
        bits(x)=isnan(x) ? "missing" : string(reinterpret(UInt64,x))
        SHA.update!(context,codeunits(join((Dates.value(row.tag),Dates.value(row.completed),
            Dates.value(row.received),bits(row.bz),bits(row.q)),"|")*"\n"))
    end
    bytes2hex(SHA.digest!(context))
end
function _edges(values)
    isempty(values) && return nothing
    ordered=sort(collect(values);alg=MergeSort)
    ntuple(k->ordered[cld(k*length(ordered),4)],3)
end
_bin(q,edges) = something(findfirst(e->q<=e,edges),4)
_partition_counts(n) = ntuple(k->cld(k*n,4)-cld((k-1)*n,4),4)
function _marks(values)
    counts=_partition_counts(length(values))
    length(values)>=4 || return nothing,counts
    ordered=sort(collect(values);alg=MergeSort)
    marks=ntuple(4) do k
        first=cld((k-1)*length(ordered),4)+1;last=cld(k*length(ordered),4)
        exact=sum(Rational{BigInt}(ordered[i]) for i in first:last)/(last-first+1)
        value=Float64(exact)
        _need(isfinite(value) && (exact==0||value!=0),"mark centroid range unsupported")
        value
    end
    marks,counts
end

"""Immutable, source-bound hazard counts and first-S marks; unsupported states stay explicit."""
struct DurationModel
    beta::Int
    boundary::DateTime
    cutoff::DateTime
    edges::Union{Nothing,NTuple{3,Float64}}
    risks::Cells
    flips::Cells
    global_risks::NTuple{2,Int}
    global_flips::NTuple{2,Int}
    marks::Union{Nothing,NTuple{4,Float64}}
    mark_counts::NTuple{4,Int}
    training_prefix_sha256::String
    source_sha256::String
    source_signature::String
    model_id::String
end
function _data(m::DurationModel)
    (;schema="duration_forcing_research_v1",availability_kind="assumed_completed_hour",
      beta=m.beta,boundary=string(m.boundary),cutoff=string(m.cutoff),edges=m.edges,
      risks=m.risks,flips=m.flips,global_risks=m.global_risks,global_flips=m.global_flips,
      marks=m.marks,mark_counts=m.mark_counts,training_prefix_sha256=m.training_prefix_sha256,
      source_sha256=m.source_sha256,source_signature=m.source_signature,
      quantile_rule="nearest_rank_lower_bin_v1",mark_rule="four_ceil_rank_blocks_v1")
end
function _validate(m::DurationModel;check_identity=true)
    _need(m.beta in (5,20) && m.cutoff==_shift(m.boundary,-168),"duration configuration/cutoff differs")
    _need(all(p->all(x->0<=x<=MAX_HOURS,p),m.risks) && all(p->all(x->0<=x<=MAX_HOURS,p),m.flips),"invalid risk counts")
    _need(all(m.flips[a][b]<=m.risks[a][b] for a in 1:49,b in 1:5),"flips exceed known transitions")
    _need(m.global_risks==(sum(r[1] for r in m.risks),sum(sum(r[2:5]) for r in m.risks)) &&
          m.global_flips==(sum(r[1] for r in m.flips),sum(sum(r[2:5]) for r in m.flips)),"global/cell counts differ")
    _need(sum(m.global_risks)<=MAX_HOURS-2,"risk counts exceed bounded known-next source transitions")
    _need((m.edges===nothing)==(m.global_risks[2]==0),"South edge support differs")
    m.edges===nothing || _need(all(isfinite,m.edges) && all(>=(0),m.edges) && issorted(m.edges),"invalid South edges")
    _need(m.mark_counts==_partition_counts(m.global_flips[1]),"first-S rank masses differ")
    _need((m.marks===nothing)==(m.global_flips[1]<4),"first-S mark support differs")
    m.marks===nothing || _need(all(isfinite,m.marks) && all(>=(0),m.marks) && issorted(m.marks),"invalid first-S marks")
    _need(all(_hex,(m.training_prefix_sha256,m.source_sha256,m.source_signature,m.model_id)),"invalid model digest")
    _need(m.source_signature==_source_signature(),"duration model/loaded source differs")
    check_identity && _need(_hash(_data(m))==m.model_id,"duration model identity differs")
    m
end
function _model(beta,boundary,edges,risks,flips,globals,globalflips,marks,masses,prefix,source)
    fields=(beta,boundary,_shift(boundary,-168),edges,risks,flips,globals,globalflips,marks,masses,prefix,source,_source_signature())
    provisional=DurationModel(fields...,repeat("0",64))
    model=DurationModel(fields...,_hash(_data(provisional)))
    _validate(model)
end

"""Fit beta5/20 hazards using only completed-and-received transitions before B−168h.
No Dst target, network, actual-fit permit or serving interface is consumed.
"""
function fit_duration(rows,boundary::DateTime;beta::Integer,source_sha256::AbstractString)
    source_identity();_ordered(rows)
    _need(!(beta isa Bool)&&beta in (5,20)&&_hex(source_sha256),"beta5/20 and source digest required")
    cutoff=_shift(boundary,-168)
    eligible=[r for r in rows if r.completed<=cutoff && r.received<=cutoff]
    trace=_trace(eligible);pairs=Tuple{Int,Symbol,Int,Bool}[]
    for i in 1:length(eligible)-1
        s,age=trace[i];next,_=trace[i+1]
        age>0 && next in (:N,:S) && eligible[i+1].tag==_shift(eligible[i].tag,1) || continue
        push!(pairs,(i,s,age,s!=next))
    end
    edges=_edges([eligible[i].q for (i,s,_,_) in pairs if s==:S])
    risks=zeros(Int,49,5);flips=zeros(Int,49,5);mark_values=Float64[]
    for (i,s,age,flip) in pairs
        column=s==:N ? 1 : 1+_bin(eligible[i].q,edges)
        risks[age,column]+=1;flips[age,column]+=flip
        s==:N && flip && push!(mark_values,eligible[i+1].q)
    end
    marks,masses=_marks(mark_values)
    cells(a)=ntuple(i->ntuple(j->a[i,j],5),49)
    globals=(sum(risks[:,1]),sum(risks[:,2:5]));globalflips=(sum(flips[:,1]),sum(flips[:,2:5]))
    _model(Int(beta),boundary,edges,cells(risks),cells(flips),globals,globalflips,marks,masses,
        _prefix_digest(eligible),String(source_sha256))
end

"""Derive the exact latest completed wind state and known/capped run age at a nominal issue."""
function current_state(rows,issue::DateTime)
    source_identity();_ordered(rows)
    eligible=[r for r in rows if r.completed<=issue && r.received<=issue]
    trace=_trace(eligible)
    reason=isempty(eligible)||last(eligible).tag!=_shift(issue,-1) ? :latest_driver_unavailable :
        last(trace)[1]==:missing ? :latest_driver_missing : last(trace)[2]==0 ? :unknown_initial_age : :available
    state,age=isempty(trace) ? (:missing,0) : last(trace)
    q=reason==:available ? last(eligible).q : nothing
    data=(;status=reason==:available ? :available : :unavailable,reason,issue,state,age,q,
          input_prefix_sha256=_prefix_digest(eligible),availability_kind=:assumed_completed_hour)
    merge(data,(state_id=_hash(data),))
end
function _hazard(m,state,age,q)
    _need(state in (:N,:S) && age isa Integer && !(age isa Bool) && 1<=age<=49,"known state/age required")
    _need(q isa Real && !(q isa Bool) && isfinite(q) && q>=0 && (state==:S||q==0),"invalid current injection")
    slot=state==:N ? 1 : 2
    m.global_risks[slot]>0 || return (;status=:unavailable,reason=:global_state_support_missing,probability=nothing)
    column=state==:N ? 1 : 1+_bin(q,m.edges)
    global_rate=BigInt(m.global_flips[slot])//BigInt(m.global_risks[slot])
    probability=(m.flips[age][column]+m.beta*global_rate)/(m.risks[age][column]+m.beta)
    (;status=:available,reason=:available,probability)
end
"""Return the exact rational shrunk flip probability, or explicit missing global support."""
hazard(m::DurationModel,state::Symbol,age::Integer,q::Real) = (_validate(m);_hazard(m,state,age,q))

struct AffinePath
    id::Tuple{Vararg{Int}}
    probability::Rational{BigInt}
    state::Symbol
    age::Int
    q::Float64
    states::Tuple{Vararg{Symbol}}
    injections::Tuple{Vararg{Float64}}
    coefficients::Tuple{Vararg{NTuple{2,Float64}}}
    endpoint_bases::Tuple{Vararg{Basis}}
end
_matrix(b::Basis) = [b[i][j] for i in 1:2,j in 1:2]
_basis(b) = ntuple(i->ntuple(j->Float64(b[i,j]),2),2)
_unavailable(reason) = (;status=:unavailable,reason,paths=(),claim_ready=false)
function _base_schedule(checkpoint,lambdas,sigmas,pressure,observation_variance,horizon)
    _need(length(lambdas)==length(sigmas)==2,"two modes required")
    l=Tuple(K._number(x,"lambda") for x in lambdas);s=Tuple(K._number(x,"sigma") for x in sigmas)
    _need(all(>=(0),l)&&all(>=(0),s),"nonnegative lambda/sigma required")
    p=K._number(pressure,"fixed pressure");r=K._number(observation_variance,"target observation variance")
    _need(r>=0,"negative observation variance")
    correction=7.26sqrt(max(p,0.))-11.
    kernels=Tuple(K.interval_kernel(l[j],s[j],1.) for j in 1:2)
    mean=collect(checkpoint.mean);covariance=R._matrix(checkpoint.covariance)
    offsets=Float64[];variances=Float64[];means=NTuple{2,Float64}[];covariances=Basis[]
    for _ in 1:horizon
        joint=K.interval_moments(mean,covariance,collect(l),collect(s),[0.,0.],1.)
        push!(offsets,joint.interval_mean+correction);push!(variances,joint.interval_variance+r)
        mean=joint.endpoint_mean;covariance=joint.endpoint_covariance
        push!(means,Tuple(mean));push!(covariances,_basis(covariance))
    end
    _need(all(isfinite,offsets)&&all(isfinite,variances),"nonfinite forecast schedule")
    (;kernels,offsets=Tuple(offsets),variances=Tuple(variances),means=Tuple(means),covariances=Tuple(covariances))
end

"""Enumerate every positive-mass h1–h6 forcing branch and its A affine interval design.
Covariance and zero-forcing base moments are shared by depth. No future wind or
observations are read. Pressure/parameter uncertainty and calibration are absent.
"""
function affine_rollout(model::DurationModel,current,checkpoint::R.Checkpoint,initial_basis::AbstractMatrix;
        lambdas,sigmas,pressure,target_observation_variance,horizon::Integer=6)
    _validate(model)
    _need(!(horizon isa Bool)&&1<=horizon<=6,"horizon must be1:6")
    fields=(:status,:reason,:issue,:state,:age,:q,:input_prefix_sha256,:availability_kind)
    _need(keys(current)==(fields...,:state_id) && _hash(NamedTuple{fields}(Tuple(current[k] for k in fields)))==current.state_id,"current state identity differs")
    current.status==:available || return _unavailable(current.reason)
    _need(current.availability_kind==:assumed_completed_hour && checkpoint.boundary==current.issue &&
          current.issue>=model.boundary,"state/issue/model fitting boundary differs")
    _need(size(initial_basis)==(2,2),"two-mode/two-gain historical basis required")
    basis=_basis(map(x->K._number(x,"historical basis"),initial_basis))
    schedule=_base_schedule(checkpoint,lambdas,sigmas,pressure,target_observation_variance,horizon)
    phi=ntuple(j->schedule.kernels[j].phi,2);c=ntuple(j->schedule.kernels[j].mean_coefficient,2)
    u=ntuple(j->schedule.kernels[j].forcing_endpoint,2);v=ntuple(j->schedule.kernels[j].forcing_mean,2)
    paths=[AffinePath((),BigInt(1)//BigInt(1),current.state,current.age,current.q,(),(),(),())]
    expanded=1;peak=1
    for depth in 1:horizon
        next=AffinePath[]
        for path in paths
            h=_hazard(model,path.state,path.age,path.q)
            h.status==:available || return _unavailable(h.reason)
            transitions=Tuple{Int,Symbol,Int,Float64,Rational{BigInt}}[]
            if h.probability<1
                push!(transitions,(0,path.state,min(path.age+1,49),path.state==:S ? path.q : 0.,1-h.probability))
            end
            if h.probability>0
                if path.state==:S
                    push!(transitions,(1,:N,1,0.,h.probability))
                else
                    model.marks===nothing && return _unavailable(:first_S_mark_support_missing)
                    for mark in 1:4
                        mass=BigInt(model.mark_counts[mark])//BigInt(sum(model.mark_counts))
                        push!(transitions,(mark+1,:S,1,model.marks[mark],h.probability*mass))
                    end
                end
            end
            previous=depth==1 ? basis : last(path.endpoint_bases)
            for (id,state,age,q,mass) in transitions
                coefficients=ntuple(k->sum(c[j]*previous[j][k]+(j==k ? -v[j]*q : 0.) for j in 1:2),2)
                endpoint=ntuple(j->ntuple(k->phi[j]*previous[j][k]+(j==k ? -u[j]*q : 0.),2),2)
                _need(all(isfinite,coefficients)&&all(isfinite,Iterators.flatten(endpoint)),"nonfinite branch affine design")
                push!(next,AffinePath((path.id...,id),path.probability*mass,state,age,q,
                    (path.states...,state),(path.injections...,q),(path.coefficients...,coefficients),
                    (path.endpoint_bases...,endpoint)))
            end
        end
        _need(sum(p.probability for p in next)==1,"forcing path mass differs from one")
        _need(length(next)<=1093,"h6 path bound exceeded")
        expanded+=length(next);peak=max(peak,length(next));paths=next
    end
    _need(expanded<=1639,"h6 expanded-node bound exceeded")
    _validate(model)
    (;status=:available,reason=:available,issue=current.issue,horizon=Int(horizon),model_id=model.model_id,
      current_state_id=current.state_id,paths=Tuple(paths),schedule,peak_paths=peak,expanded_nodes=expanded,
      variance_scope=:fixed_fitted_parameters_fixed_pressure_plus_explicit_target_noise,
      measured_receipts_available=false,claim_ready=false)
end

"""Evaluate a B affine tree at nonnegative gains and retain full mixture components."""
function evaluate_rollout(result,gains)
    source_identity()
    result.status==:available || return result
    _need(length(gains)==2,"two gains required")
    g=Tuple(K._number(x,"gain") for x in gains);_need(all(>=(0),g),"negative gain")
    summaries=K._precise() do
        Tuple(begin
            weights=[BigFloat(p.probability) for p in result.paths]
            offset=result.schedule.offsets[h]
            coefficients=ntuple(k->Float64(sum(weights[i]*BigFloat(result.paths[i].coefficients[h][k]) for i in eachindex(weights))),2)
            mean=offset+sum(coefficients[k]*g[k] for k in 1:2)
            branches=[offset+sum(p.coefficients[h][k]*g[k] for k in 1:2) for p in result.paths]
            _need(isfinite(mean)&&all(isfinite,branches),"nonfinite branch prediction")
            conditional=result.schedule.variances[h]
            between=Float64(sum(weights[i]*(BigFloat(branches[i])-BigFloat(mean))^2 for i in eachindex(weights)))
            total=conditional+between
            _need(isfinite(between)&&isfinite(total)&&between>=0,"invalid mixture variance")
            (;horizon=h,offset,coefficients,mean,conditional_variance=conditional,
              between_forcing_variance=between,total_variance=total)
        end for h in 1:result.horizon)
    end
    components=Tuple(begin
        endpoints=Tuple(ntuple(j->result.schedule.means[h][j]+sum(p.endpoint_bases[h][j][k]*g[k] for k in 1:2),2) for h in 1:result.horizon)
        (;branch_id=p.id,probability=p.probability,states=p.states,injections=p.injections,
          interval_means=Tuple(result.schedule.offsets[h]+sum(p.coefficients[h][k]*g[k] for k in 1:2) for h in 1:result.horizon),
          conditional_variances=result.schedule.variances,endpoint_means=endpoints,
          endpoint_covariances=result.schedule.covariances)
    end for p in result.paths)
    merge(result,(gains=g,summaries,components))
end

function _new(path,body)
    _need(!ispath(path),"refusing to overwrite artifact member")
    open(path,"w") do io;_need(write(io,body)==length(body),"short artifact write");end
end
function _save_model(destination::AbstractString,model::DurationModel,after_commit_check::Function)
    _validate(model);root=abspath(destination);parent=dirname(root)
    _need(isdir(parent)&&!islink(parent)&&!ispath(root),"new ordinary artifact destination required")
    claim=root*".claim";mkdir(claim);marker_body=nothing
    try
        attempt=mktempdir(parent;prefix="duration-attempt-",cleanup=false)
        body=_json(merge(_data(model),(model_id=model.model_id,)))
        _need(length(body)<=MAX_ARTIFACT_BYTES,"model byte budget")
        _new(joinpath(attempt,"model.json"),body)
        manifest=_json((;schema="duration_model_manifest_v1",model_sha256=_sha(body)))
        _new(joinpath(attempt,"MANIFEST.json"),manifest);_validate(model)
        mv(attempt,root;force=false)
        marker_body=_json((;manifest_sha256=_sha(manifest)))
        _new(joinpath(root,"COMMITTED.json"),marker_body)
        after_commit_check()
        _validate(model)
        (;directory=root,manifest_sha256=_sha(manifest),model_id=model.model_id)
    catch error
        marker=joinpath(root,"COMMITTED.json")
        if marker_body!==nothing && isfile(marker) && !islink(marker) && read(marker)==marker_body
            mv(marker,marker*".incomplete";force=false)
        end
        rethrow(error)
    finally
        rm(claim)
    end
end
"""Publish a new immutable data-only model; return the caller-pinnable manifest digest."""
save_model(destination::AbstractString,model::DurationModel) = _save_model(destination,model,()->nothing)
function _bound(path)
    _need(isfile(path)&&!islink(path)&&filesize(path)<=MAX_ARTIFACT_BYTES,"bounded regular artifact member required")
    read(path)
end
"""Load exact data-only model bytes with a caller-pinned manifest and semantic/source guards."""
function load_model(directory::AbstractString;expected_manifest_sha256::AbstractString)
    source_identity();root=abspath(directory)
    _need(_hex(expected_manifest_sha256)&&isdir(root)&&!islink(root),"pinned ordinary model directory required")
    _need(sort(readdir(root))==["COMMITTED.json","MANIFEST.json","model.json"],"artifact membership differs")
    manifest=_bound(joinpath(root,"MANIFEST.json"));marker=JSON3.read(_bound(joinpath(root,"COMMITTED.json")))
    _need(Set(keys(marker))==Set((:manifest_sha256,))&&marker.manifest_sha256==expected_manifest_sha256==_sha(manifest),"artifact manifest differs")
    ledger=JSON3.read(manifest);_need(Set(keys(ledger))==Set((:schema,:model_sha256))&&ledger.schema=="duration_model_manifest_v1","artifact ledger differs")
    body=_bound(joinpath(root,"model.json"));_need(_sha(body)==ledger.model_sha256,"model bytes changed")
    d=JSON3.read(body);fields=(:schema,:availability_kind,:beta,:boundary,:cutoff,:edges,:risks,:flips,:global_risks,:global_flips,
        :marks,:mark_counts,:training_prefix_sha256,:source_sha256,:source_signature,:quantile_rule,:mark_rule,:model_id)
    _need(Set(keys(d))==Set(fields)&&d.schema=="duration_forcing_research_v1"&&d.availability_kind=="assumed_completed_hour"&&
          d.quantile_rule=="nearest_rank_lower_bin_v1"&&d.mark_rule=="four_ceil_rank_blocks_v1","model schema differs")
    integer(x)=(_need(x isa Integer&&!(x isa Bool)&&0<=x<=MAX_HOURS,"invalid stored integer");Int(x))
    cells(x)=(_need(length(x)==49&&all(r->length(r)==5,x),"stored cell shape differs");ntuple(i->ntuple(j->integer(x[i][j]),5),49))
    vector(x,n)=(_need(length(x)==n,"stored vector shape differs");Tuple(integer(v) for v in x))
    floats(x,n)=x===nothing ? nothing : (_need(length(x)==n,"stored float shape differs");Tuple(K._number(v,"stored float") for v in x))
    m=DurationModel(integer(d.beta),DateTime(String(d.boundary)),DateTime(String(d.cutoff)),floats(d.edges,3),cells(d.risks),cells(d.flips),
        vector(d.global_risks,2),vector(d.global_flips,2),floats(d.marks,4),vector(d.mark_counts,4),
        String(d.training_prefix_sha256),String(d.source_sha256),String(d.source_signature),String(d.model_id))
    _validate(m)
    _need(_bound(joinpath(root,"model.json"))==body&&_bound(joinpath(root,"MANIFEST.json"))==manifest,"artifact changed during load")
    m
end
end # module
