module DstIntervalStream

using Dates, LinearAlgebra, SHA
using ..DstIntervalAdapter
const A=DstIntervalAdapter
const R=A.Replay
const K=A.Kernel
const HourlyInputs=A.Features.HourlyInputs
const LOADED_SOURCE_SHA256=bytes2hex(sha256(read(@__FILE__)))
A.verify_source_identity()

export HourlyInputs, StreamCheckpoint, stream_basis

function verify_source_identity()
    dependencies=A.verify_source_identity()
    bytes2hex(sha256(read(@__FILE__)))==LOADED_SOURCE_SHA256 ||
        throw(ArgumentError("loadedsource/disk mismatch: DstIntervalStream.jl"))
    (dependencies...,"DstIntervalStream.jl"=>LOADED_SOURCE_SHA256)
end

struct PendingDriver
    index::Int
    start::DateTime
    stop::DateTime
    q::Float64
    pressure::Float64
end

struct StreamCheckpoint
    last_index::Int
    last_issue::DateTime
    source_sha256::String
    prefix_sha256::String
    delay_hours::Int
    taus::Tuple{Float64,Float64}
    sigma::Float64
    sealed::R.Checkpoint
    basis::NTuple{2,NTuple{2,Float64}}
    pending::Union{Nothing,PendingDriver}
    active::Bool
    consecutive_valid::Int
    reset_tag::DateTime
end

_matrix(rows)=[rows[i][j] for i in 1:2,j in 1:2]
_rows(matrix)=((matrix[1,1],matrix[1,2]),(matrix[2,1],matrix[2,2]))
_pressure(p)=7.26sqrt(max(p,0.))-11.
_prior(t)=R.Checkpoint(t,[0.,0.],[10000. 0.;0. 10000.])
_zero_basis()=((0.,0.),(0.,0.))
_hex(s)=s isa AbstractString && occursin(r"\A[0-9a-f]{64}\z",s)

function _recipe(delay,taus,sigma,chunk_hours)
    delay isa Bool && throw(ArgumentError("Boolean is not a delay"))
    delay in (0,1) || throw(ArgumentError("onlyL0/L1 are defined"))
    length(taus)==2 || throw(ArgumentError("two decaytimes required"))
    tau=Tuple(K._number(x,"decaytime") for x in taus)
    tau in A.TAUS || throw(ArgumentError("decaypair differs from frozen primary grid"))
    noise=K._number(sigma,"processsigma")
    noise in A.SIGMAS || throw(ArgumentError("sigma differs from frozen primary grid"))
    chunk_hours isa Integer && !(chunk_hours isa Bool) && 1<=chunk_hours<=4096 || throw(ArgumentError("chunk bound must be1:4096"))
    Int(delay),tau,noise,Int(chunk_hours)
end

function _driver(raw,index,time)
    speed,bz,pressure=raw[8,index],raw[6,index],raw[12,index]
    all(isfinite,(speed,bz,pressure)) && speed>=0 || return nothing
    injection=max(speed*max(-bz,0.)/1000-.49,0.)
    isfinite(injection) || throw(ArgumentError("injection exceeds supported numerical range"))
    PendingDriver(index,time,A._shift(time,1),injection,pressure)
end

function _prefix_seed(input,delay)
    bytes2hex(sha256("candidate-A-input-prefix-v1|$(input.source_sha256)|$delay|$(first(input.time))"))
end

function _prefix_step(previous,input,raw,index,delay)
    buffer=IOBuffer()
    write(buffer,previous)
    write(buffer,htol(Int64(Dates.value(input.time[index]))))
    for channel in (8,6,12)
        value=raw[channel,index]
        write(buffer,htol(reinterpret(UInt64,isfinite(value) ? value : NaN)))
    end
    observed=index-delay
    write(buffer,UInt8(observed>=1))
    if observed>=1
        write(buffer,htol(Int64(Dates.value(input.time[observed]))))
        for channel in (15,12)
            value=raw[channel,observed]
            write(buffer,htol(reinterpret(UInt64,isfinite(value) ? value : NaN)))
        end
    end
    bytes2hex(sha256(take!(buffer)))
end

function _advance(sealed,basis,pending,raw,delay,asof,lambdas,sigmas;observe)
    id="rawbin:$(pending.index)"
    bin=R.ForcingInterval(id,pending.start,pending.stop,[0.,0.],pending.stop)
    observations=R.MeanObservation[]
    value=raw[15,pending.index]
    if observe && isfinite(value)
        received=A._shift(pending.stop,delay)
        received<=asof || throw(ArgumentError("attempt to assimilate unrevealedDst"))
        push!(observations,R.MeanObservation("dst:$(pending.index)",id,pending.start,pending.stop,
            value-_pressure(pending.pressure),1.,received,received))
    end
    forcing_basis=Dict(id=>[-pending.q 0.;0. -pending.q])
    result=R.affine_gain_replay(sealed,lambdas,sigmas,[bin],observations,pending.stop,
        forcing_basis,_matrix(basis);asof,receipt_boundary=:hypothetical_inclusive)
    result.base,result.coefficients,length(observations)
end

function _targets(state,basis,current,lambdas,sigmas)
    mean=collect(state.mean)
    covariance=R._matrix(state.covariance)
    slopes=_matrix(basis)
    forcing_basis=[-current.q 0.;0. -current.q]
    offsets=Float64[];coefficients=Tuple{Float64,Float64}[];variances=Float64[]
    kernels=[K.interval_kernel(lambdas[j],sigmas[j],1.) for j in 1:2]
    phi=[k.phi for k in kernels];c=[k.mean_coefficient for k in kernels]
    u=[k.forcing_endpoint for k in kernels];v=[k.forcing_mean for k in kernels]
    for horizon in 1:6
        joint=K.interval_moments(mean,covariance,lambdas,sigmas,[0.,0.],1.)
        target_slopes=slopes'*c+forcing_basis'*v
        if horizon in (4,6)
            offset=joint.interval_mean+_pressure(current.pressure)
            all(isfinite,(offset,target_slopes...)) || throw(ArgumentError("nonfiniteaffine target"))
            push!(offsets,offset);push!(coefficients,Tuple(target_slopes));push!(variances,joint.interval_variance)
        end
        slopes=Diagonal(phi)*slopes+Diagonal(u)*forcing_basis
        all(isfinite,slopes) || throw(ArgumentError("nonfinitefuturebasis"))
        mean=joint.endpoint_mean;covariance=joint.endpoint_covariance
    end
    Tuple(offsets),Tuple(coefficients),Tuple(variances)
end

function _validate_checkpoint(cp,input,raw,delay,taus,sigma)
    cp isa StreamCheckpoint || throw(ArgumentError("unsupportedcheckpointtype"))
    cp.source_sha256==input.source_sha256 && cp.delay_hours==delay && cp.taus==taus && cp.sigma==sigma ||
        throw(ArgumentError("checkpoint source/configuration mismatch"))
    0<=cp.last_index<=length(input.time) || throw(ArgumentError("checkpoint index outside input"))
    expected_issue=cp.last_index==0 ? first(input.time) : A._shift(input.time[cp.last_index],1)
    cp.last_issue==expected_issue || throw(ArgumentError("checkpoint issue/index mismatch"))
    _hex(cp.prefix_sha256) || throw(ArgumentError("invalidcheckpointprefixhash"))
    prefix=_prefix_seed(input,delay)
    for index in 1:cp.last_index
        prefix=_prefix_step(prefix,input,raw,index,delay)
    end
    prefix==cp.prefix_sha256 || throw(ArgumentError("checkpoint consumed-prefix mismatch"))
    R.Checkpoint(cp.sealed.boundary,collect(cp.sealed.mean),R._matrix(cp.sealed.covariance))
    all(isfinite,Iterators.flatten(cp.basis)) || throw(ArgumentError("nonfinitecheckpointbasis"))
    if cp.active
        1<=cp.consecutive_valid<=cp.last_index || throw(ArgumentError("invalidcheckpointvaliditycount"))
        cp.reset_tag==input.time[cp.last_index-cp.consecutive_valid+1] || throw(ArgumentError("checkpointresetmismatch"))
        if delay==0
            cp.pending===nothing && cp.sealed.boundary==cp.last_issue || throw(ArgumentError("L0checkpointtailmismatch"))
        else
            cp.pending!==nothing || throw(ArgumentError("L1checkpointlostpendingbin"))
            expected=_driver(raw,cp.last_index,input.time[cp.last_index])
            expected!==nothing && cp.pending==expected && cp.sealed.boundary==cp.pending.start ||
                throw(ArgumentError("L1checkpointpendingdrivermismatch"))
        end
    else
        cp.consecutive_valid==0 && cp.pending===nothing && cp.sealed.boundary==cp.last_issue ||
            throw(ArgumentError("inactivecheckpointtailmismatch"))
    end
    cp
end

"""Stream the archival hypothetical-L0/L1 affine h4/h6 design, without fitting.
`sink` receives bounded independent Vector chunks. Targets are separate labels,
never used by the state/design. All emitted rows are retained, including warmup,
missing-anchor and missing-target rows. A full source identity is supplied by
the verified parser/caller; a consumed-input chain additionally protects resume.
The actual late-revision journal adapter is not implemented by this interface.
Optional `issue_sink` receives a detached past-only issue checkpoint, a fresh
historical gain-basis matrix and an immutable resume checkpoint before future
continuation. L1 retains its sealed state and pending driver separately. The
payload contains no target labels. Callback exceptions abort before that hour's
ordinary row is published; previously published chunks are retained.
"""
function stream_basis(input::HourlyInputs;delay_hours::Integer,taus,sigma::Real,
        chunk_hours::Integer=24,sink=(rows)->nothing,checkpoint=nothing,
        stop_after_index::Integer=length(input.time),issue_sink=nothing)
    implementation_hashes=verify_source_identity()
    issue_sink===nothing || issue_sink isa Function ||
        throw(ArgumentError("issue sink must be nothing or a function"))
    delay,tau,noise,chunk=_recipe(delay_hours,taus,sigma,chunk_hours)
    _hex(input.source_sha256) || throw(ArgumentError("hash-boundinputidentityrequired"))
    raw=A.Features.censor_fills(input.values)
    initial=StreamCheckpoint(0,first(input.time),input.source_sha256,_prefix_seed(input,delay),
        delay,tau,noise,_prior(first(input.time)),_zero_basis(),nothing,false,0,first(input.time))
    cp=checkpoint===nothing ? initial : _validate_checkpoint(checkpoint,input,raw,delay,tau,noise)
    !(stop_after_index isa Bool) && cp.last_index<=stop_after_index<=length(input.time) ||
        throw(ArgumentError("stop index must lie between checkpoint andsourceend"))
    lambdas=[1/tau[1],1/tau[2]];sigmas=[noise,noise]
    sealed,basis,pending=cp.sealed,cp.basis,cp.pending
    active,consecutive,reset=cp.active,cp.consecutive_valid,cp.reset_tag
    prefix=cp.prefix_sha256
    buffer=NamedTuple[];emitted=0;maximum_buffer=0;observations_used=0;negative_pressure_hours=0
    fallback_counts=Dict{String,Int}()
    for index in cp.last_index+1:stop_after_index
        tag=input.time[index];issue=A._shift(tag,1)
        prefix=_prefix_step(prefix,input,raw,index,delay)
        current=_driver(raw,index,tag)
        negative_pressure_hours+=isfinite(raw[12,index]) && raw[12,index]<0
        model_state=nothing;model_basis=nothing
        if current===nothing
            active=false;consecutive=0;pending=nothing;reset=issue
            sealed=_prior(issue);basis=_zero_basis()
        else
            if !active
                sealed=_prior(tag);basis=_zero_basis();pending=nothing
                active=true;consecutive=0;reset=tag
            end
            if delay==1 && pending!==nothing
                sealed,basis,used=_advance(sealed,basis,pending,raw,delay,issue,lambdas,sigmas;observe=true)
                observations_used+=used
            end
            if delay==0
                sealed,basis,used=_advance(sealed,basis,current,raw,delay,issue,lambdas,sigmas;observe=true)
                observations_used+=used
                model_state=sealed;model_basis=basis;pending=nothing
            else
                model_state,model_basis,_=_advance(sealed,basis,current,raw,delay,issue,lambdas,sigmas;observe=false)
                pending=current
            end
            consecutive+=1
        end
        clocks=(A.forecast_clock(issue,4,delay),A.forecast_clock(issue,6,delay))
        anchor_index=index-delay
        anchor=anchor_index>=1 ? raw[15,anchor_index] : NaN
        available=active && consecutive>=168 && isfinite(anchor)
        reason=!isfinite(anchor) ? "anchor_missing" : !active ? "known_driver_missing" : consecutive<168 ? "warmup" : "model"
        if issue_sink!==nothing
            resume=StreamCheckpoint(index,issue,input.source_sha256,prefix,delay,tau,noise,
                sealed,basis,pending,active,consecutive,reset)
            detached=model_state===nothing ? nothing : R.Checkpoint(model_state.boundary,
                collect(model_state.mean),R._matrix(model_state.covariance))
            issue_sink((;index,issue,driver_tag=tag,driver_receipt=issue,delay_hours=delay,
                taus=tau,sigma=noise,availability_kind=input.availability_kind,active,
                consecutive_valid=consecutive,reset_tag=reset,anchor_index,
                anchor_tag=clocks[1].anchor_tag,anchor_complete=clocks[1].anchor_complete,
                anchor_receipt=clocks[1].anchor_receipt,anchor,a_model_available=available,
                a_reason=reason,current_driver=current,issue_checkpoint=detached,
                issue_basis=model_basis===nothing ? nothing : _matrix(model_basis),
                resume_checkpoint=resume,source_sha256=input.source_sha256,
                input_prefix_sha256=prefix,implementation_hashes))
        end
        target=Tuple(index+h<=length(input.time) ? raw[15,index+h] : NaN for h in (4,6))
        scoreable=Tuple(isfinite(anchor)&&isfinite(y) for y in target)
        offsets,coefficients,variances = if available
            _targets(model_state,model_basis,current,lambdas,sigmas)
        else
            ((anchor,anchor),_zero_basis(),(NaN,NaN))
        end
        row=(issue=issue,driver_tag=tag,driver_receipt=issue,
            anchor_tag=clocks[1].anchor_tag,anchor_complete=clocks[1].anchor_complete,
            anchor_receipt=clocks[1].anchor_receipt,anchor=anchor,
            target_tag=Tuple(c.target_tag for c in clocks),target_complete=Tuple(c.target_complete for c in clocks),
            target_receipt=Tuple(c.target_receipt for c in clocks),target=target,scoreable=scoreable,
            model_available=available,reason=reason,offset=offsets,coefficients=coefficients,
            variance=variances,consecutive_valid=consecutive,reset_tag=reset,
            completed_target_leads=(4,6),dst_delay_hours=delay,taus=tau,sigma=noise,
            source_sha256=input.source_sha256,
            latest_q=current===nothing ? NaN : current.q,latest_pressure=raw[12,index])
        push!(buffer,row);emitted+=1;maximum_buffer=max(maximum_buffer,length(buffer))
        fallback_counts[reason]=get(fallback_counts,reason,0)+1
        if length(buffer)==chunk
            sink(copy(buffer));empty!(buffer)
        end
        cp=StreamCheckpoint(index,issue,input.source_sha256,prefix,delay,tau,noise,
            sealed,basis,pending,active,consecutive,reset)
    end
    isempty(buffer) || sink(copy(buffer))
    verify_source_identity()
    (;checkpoint=cp,row_count=emitted,max_buffered_rows=maximum_buffer,fallback_counts,observations_used,negative_pressure_hours,
      source_sha256=input.source_sha256,delay_hours=delay,taus=tau,sigma=noise)
end

end # module
