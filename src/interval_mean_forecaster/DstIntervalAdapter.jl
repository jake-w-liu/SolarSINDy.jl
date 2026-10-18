module DstIntervalAdapter

using Dates, LinearAlgebra, SHA
const LOADED_SOURCE_SHA256=bytes2hex(sha256(read(@__FILE__)))
const PINNED_DEPENDENCIES=(
    "DstIntervalObserver.jl"=>"48e2af386d2b9b1a2b8afb4366e1c48508261e54a81447d438fd8a419581b816",
    "DstIntervalReplay.jl"=>"08f8088e55b3b10bb189d4f89360dd72e30e2c5f0f162f7c687e92a7f4ce1165",
    "sandu_fullchannel_features.jl"=>"2306f63cadfb5d4ae16579cabb114b813f1f41678aaa6dbdd90caad316ff792c")
for (name,digest) in PINNED_DEPENDENCIES
    bytes2hex(sha256(read(joinpath(@__DIR__,name))))==digest || error("pinneddependency bytes differ: $name")
end
include("DstIntervalReplay.jl")
include("sandu_fullchannel_features.jl")
const Replay=DstIntervalReplay
const Kernel=Replay.DstIntervalObserver
const Features=SanduFullChannelFeatures

export forecast_clock, fit_eligible, joint_fit_mask, settings, nnls2, verify_source_identity

function verify_source_identity()
    identities=(PINNED_DEPENDENCIES...,"DstIntervalAdapter.jl"=>LOADED_SOURCE_SHA256)
    for (name,digest) in identities
        bytes2hex(sha256(read(joinpath(@__DIR__,name))))==digest ||
            throw(ArgumentError("loadedsource/disk mismatch: $name"))
    end
    identities
end
verify_source_identity()

const HOUR_MS=Int64(3_600_000)
const TAUS=((2.,12.),(4.,24.),(8.,48.))
const SIGMAS=(1.,3.,10.)
const ETAS=(0.,.25)

function _shift(t::DateTime,h::Integer)
    h isa Bool && throw(ArgumentError("Boolean is not a duration"))
    count=try Int64(h) catch;throw(ArgumentError("duration outsideInt64"));end
    value=try Base.Checked.checked_add(Dates.value(t),Base.Checked.checked_mul(count,HOUR_MS)) catch
        throw(ArgumentError("clock arithmetic overflow"))
    end
    DateTime(Dates.UTInstant{Millisecond}(Millisecond(value)))
end

function forecast_clock(issue::DateTime,horizon::Integer,delay::Integer)
    horizon isa Bool || delay isa Bool ? throw(ArgumentError("Boolean clock parameter")) : nothing
    horizon in (4,6) && delay in (0,1) || throw(ArgumentError("onlyh4/h6 andL0/L1 are defined"))
    floor(issue,Hour)==issue || throw(ArgumentError("issue must lie on an exact hour"))
    anchor=_shift(issue,-1-delay)
    complete=_shift(issue,horizon)
    (;issue,driver_tag=_shift(issue,-1),anchor_tag=anchor,
      anchor_complete=_shift(anchor,1),anchor_receipt=_shift(anchor,1+delay),
      target_tag=_shift(complete,-1),target_complete=complete,target_receipt=_shift(complete,delay))
end

function fit_eligible(issue::DateTime,complete::DateTime,receipt::DateTime,boundary::DateTime)
    issue<complete<=receipt || throw(ArgumentError("invalid forecast-label chronology"))
    cutoff=_shift(boundary,-168)
    issue<boundary && issue<=cutoff && complete<=cutoff && receipt<=cutoff
end

function joint_fit_mask(issues,h4_complete,h4_receipt,h6_complete,h6_receipt,boundary::DateTime)
    n=length(issues)
    all(x->length(x)==n,(h4_complete,h4_receipt,h6_complete,h6_receipt)) || throw(DimensionMismatch("clock lengths differ"))
    BitVector([fit_eligible(issues[i],h4_complete[i],h4_receipt[i],boundary) &&
        fit_eligible(issues[i],h6_complete[i],h6_receipt[i],boundary) for i in 1:n])
end

settings()=[(index=(ti-1)*6+(si-1)*2+ei,taus=taus,sigma=sigma,eta=eta)
    for (ti,taus) in enumerate(TAUS) for (si,sigma) in enumerate(SIGMAS) for (ei,eta) in enumerate(ETAS)]

function _nonzero_result(value,label)
    output=Float64(value)
    isfinite(output) || throw(ArgumentError("$label exceeds finiteFloat64 range"))
    value!=0 && output==0 && throw(ArgumentError("$label underflowsFloat64 range"))
    output
end

function _weighted_problem(Z,b,weights)
    size(Z,2)==2 && size(Z,1)==length(b)==length(weights) || throw(DimensionMismatch("expectedtwo columns andmatchingrows"))
    size(Z,1)>0 || throw(ArgumentError("emptyNNLS problem"))
    x=map(v->Kernel._number(v,"NNLS design"),Z)
    y=[Kernel._number(v,"NNLS residual") for v in b]
    w=[Kernel._number(v,"NNLS weight") for v in weights]
    all(>=(0),w) || throw(ArgumentError("negativeNNLS weight"))
    a=similar(x);rhs=similar(y)
    for i in axes(x,1)
        root=sqrt(w[i])
        for j in 1:2
            value=x[i,j]*root
            isfinite(value) && !(x[i,j]!=0 && root!=0 && value==0) || throw(ArgumentError("weighteddesign range unsupported"))
            a[i,j]=value
        end
        value=y[i]*root
        isfinite(value) && !(y[i]!=0 && root!=0 && value==0) || throw(ArgumentError("weightedresidual range unsupported"))
        rhs[i]=value
    end
    scale=max(maximum(abs,a),maximum(abs,rhs))
    if scale!=0
        for i in eachindex(a)
            value=a[i]/scale
            a[i]!=0 && value==0 && throw(ArgumentError("normalizeddesign underflow"))
            a[i]=value
        end
        for i in eachindex(rhs)
            value=rhs[i]/scale
            rhs[i]!=0 && value==0 && throw(ArgumentError("normalizedresidual underflow"))
            rhs[i]=value
        end
    end
    a,rhs,scale
end

# Wide certificates prevent tiny residual losses/gradients vanishing in a
# candidate comparison. This reference implementation is profiled before any
# optimization. The SVD still solves the explicitly normalized Float64 design.
function _certificate(a,b,g,operator_norm)
    Kernel._precise() do
        gain=BigFloat.(g)
        loss=zero(BigFloat);gradient=zeros(BigFloat,2)
        for i in axes(a,1)
            first,second=BigFloat(a[i,1]),BigFloat(a[i,2])
            residual=first*gain[1]+second*gain[2]-BigFloat(b[i])
            loss+=residual^2
            gradient[1]+=first*residual
            gradient[2]+=second*residual
        end
        bound=128*BigFloat(eps(Float64))*max(size(a,1),2)*BigFloat(operator_norm)*
            (BigFloat(operator_norm)*sqrt(sum(abs2,gain))+BigFloat(norm(b)))
        feasible=all(>=(0),g)
        kkt=feasible && all(g[j]==0 ? gradient[j]>=-bound : abs(gradient[j])<=bound for j in 1:2)
        (;gains=Tuple(g),loss,gradient=Tuple(gradient),bound,feasible,kkt,
           squared_gain_norm=sum(abs2,gain))
    end
end

# Only exact rank-one row spaces are projected: truncated near-rank-one
# matrices retain the original SVD policy. Wide products of Float64 values make
# the collinearity check exact; directed conversion preserves norm monotonicity.
function _refine_exact_rank_one(a,rhs,best,zero_loss,largest)
    all(>(0),best.gains) || return best
    index=findfirst(i->a[i,1]!=0 || a[i,2]!=0,axes(a,1))
    index===nothing && return best
    direction=(a[index,1],a[index,2])
    proposed=Kernel._precise() do
        v1,v2=BigFloat.(direction);g1,g2=BigFloat.(best.gains)
        squared=v1^2+v2^2
        abs(g1*v2-g2*v1)>128*BigFloat(eps(Float64))*sqrt(squared)*max(sqrt(best.squared_gain_norm),one(BigFloat)) || return nothing
        all(BigFloat(a[i,1])*v2==BigFloat(a[i,2])*v1 for i in axes(a,1)) || return nothing
        coefficient=(v1*g1+v2*g2)/squared
        [Float64(coefficient*v1,RoundToZero),Float64(coefficient*v2,RoundToZero)]
    end
    proposed===nothing && return best
    all(isfinite,proposed) && all(>=(0),proposed) || return best
    candidate=_certificate(a,rhs,proposed,largest)
    improved=Kernel._precise() do
        v1,v2=BigFloat.(direction)
        old_defect=abs(BigFloat(best.gains[1])*v2-BigFloat(best.gains[2])*v1)
        new_defect=abs(BigFloat(candidate.gains[1])*v2-BigFloat(candidate.gains[2])*v1)
        candidate.kkt && candidate.squared_gain_norm<best.squared_gain_norm && new_defect<old_defect &&
            candidate.loss<=best.loss+64*BigFloat(eps(Float64))*max(zero_loss,best.loss,candidate.loss)
    end
    improved ? candidate : best
end

function _refine_axis_candidate(a,rhs,coordinate,best,zero_loss,largest)
    column=a[:,coordinate];column_norm=norm(column)
    for _ in 1:2
        needed=Kernel._precise() do
            best.kkt && best.gains[coordinate]>0 && abs(best.gradient[coordinate])>best.bound/max(size(a,1),2)
        end
        needed || break
        residual=rhs-a*collect(best.gains)
        all(isfinite,residual) || break
        correction=dot(column./column_norm,residual)/column_norm
        proposed=collect(best.gains);proposed[coordinate]+=correction
        all(isfinite,proposed) && all(>=(0),proposed) || break
        candidate=_certificate(a,rhs,proposed,largest)
        improved=Kernel._precise() do
            candidate.kkt && abs(candidate.gradient[coordinate])<abs(best.gradient[coordinate]) &&
                candidate.loss<=best.loss+64*BigFloat(eps(Float64))*max(zero_loss,best.loss,candidate.loss)
        end
        improved || break
        best=candidate
    end
    best
end

# Refine only a positive unconstrained candidate whose stationarity loss
# exceeds a dimension-independent accuracy trigger. Final KKT/rank/tie rules
# remain unchanged. Every proposed step is checked on the full original matrix.
function _refine_svd_candidate(a,rhs,factor,usable,best,zero_loss,largest)
    for _ in 1:2
        needed=Kernel._precise() do
            all(>(0),best.gains) && maximum(abs,best.gradient)>best.bound/max(size(a,1),2)
        end
        needed || break
        residual=rhs-a*collect(best.gains)
        all(isfinite,residual) || break
        projection=[dot(view(factor.U,:,j),residual) for j in eachindex(factor.S)]
        correction=factor.V*[(usable[j] ? projection[j]/factor.S[j] : 0.) for j in eachindex(factor.S)]
        proposed=collect(best.gains)+correction
        all(isfinite,proposed) && all(>=(0),proposed) || break
        candidate=_certificate(a,rhs,proposed,largest)
        improved=Kernel._precise() do
            candidate.kkt && maximum(abs,candidate.gradient)<maximum(abs,best.gradient) &&
                candidate.loss<=best.loss+64*BigFloat(eps(Float64))*max(zero_loss,best.loss,candidate.loss)
        end
        improved || break
        best=candidate
    end
    best
end

"""Two-variable nonnegative weighted least squares, with no data access.
Each candidate passes full-matrix KKT before objective/minimum-norm ties.
The raw weightedSSE and normalizedSSE must be finite, representableFloat64.
This is a serial reference solver; no parameter fitting experiment is implied.
"""
function nnls2(Z::AbstractMatrix{<:Real},b::AbstractVector{<:Real};weights=ones(length(b)))
    verify_source_identity()
    a,rhs,scale=_weighted_problem(Z,b,weights)
    factor=svd(a;full=false)
    largest=isempty(factor.S) ? 0. : maximum(factor.S)
    cutoff=max(size(a,1),2)*eps(Float64)*largest
    usable=factor.S.>cutoff
    rank=count(usable)
    unconstrained=factor.V*[(usable[j] ? dot(factor.U[:,j],rhs)/factor.S[j] : 0.) for j in eachindex(factor.S)]
    all(isfinite,unconstrained) || throw(ArgumentError("unconstrainedgain exceeds finite range"))
    axes_gains=Vector{Vector{Float64}}()
    for j in 1:2
        column=a[:,j];column_norm=norm(column)
        gain=column_norm==0 ? 0. : max(0.,dot(column./column_norm,rhs)/column_norm)
        isfinite(gain) || throw(ArgumentError("axisgain exceeds finite range"))
        vector=zeros(2);vector[j]=gain;push!(axes_gains,vector)
    end
    candidates=[_certificate(a,rhs,g,largest) for g in ([0.,0.],axes_gains...,unconstrained)]
    candidates[end]=_refine_svd_candidate(a,rhs,factor,usable,candidates[end],candidates[1].loss,largest)
    rank==1 && (candidates[end]=_refine_exact_rank_one(a,rhs,candidates[end],candidates[1].loss,largest))
    for j in 1:2
        candidates[j+1]=_refine_axis_candidate(a,rhs,j,candidates[j+1],candidates[1].loss,largest)
    end
    valid=findall(x->x.kkt,candidates)
    isempty(valid) && throw(ArgumentError("nofeasibleKKT candidate"))
    baseline=candidates[1].loss
    best=first(valid)
    for index in valid[2:end]
        current,previous=candidates[index],candidates[best]
        tolerance=Kernel._precise() do
            64*BigFloat(eps(Float64))*max(baseline,current.loss,previous.loss)
        end
        if current.loss<previous.loss-tolerance ||
            (abs(current.loss-previous.loss)<=tolerance &&
             (current.squared_gain_norm,current.gains)<(previous.squared_gain_norm,previous.gains))
            best=index
        end
    end
    winner=candidates[best]
    raw=Kernel._precise() do
        winner.loss*BigFloat(scale)^2
    end
    diagnostics=Tuple((gains=c.gains,feasible=c.feasible,kkt=c.kkt,
        normalized_sse=string(c.loss),gradient=Tuple(string.(c.gradient)),kkt_bound=string(c.bound)) for c in candidates)
    result=(;gains=winner.gains,weighted_sse=_nonzero_result(raw,"weightedSSE"),
      normalized_sse=_nonzero_result(winner.loss,"normalizedSSE"),normalization_scale=scale,
      rank,rank_cutoff=cutoff,selected_candidate=best,candidates=diagnostics)
    verify_source_identity()
    result
end

end # module
