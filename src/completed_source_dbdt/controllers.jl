function upper_bound(alpha,center,scale,scores)
    alpha<=0 && return Inf
    alpha>=1 && return -Inf
    rank=ceil(Int,(length(scores)+1)*(1-alpha))
    rank>length(scores) && return Inf
    rank<1 && return -Inf
    expm1(Float64(center)+scores[rank]*Float64(scale))
end

"""
    static_bounds(prepared, centers, scales, calibration; nominal=0.1, asof)

Issue fixed signed-score upper bounds on feature-eligible rows without reading
targets. Unissued rows retain NaN. The outcome event log is not provided here.
"""
function static_bounds(prepared,centers,scales,cal::CompletedCalibration;nominal=0.1,asof::DateTime)
    nominal=nominal_value(nominal)
    stream=checked_stream(prepared,centers,scales,cal,asof)
    upper=fill(NaN,length(prepared))
    for i in stream.indices
        upper[i]=upper_bound(nominal,centers[i],scales[i],cal.sorted_scores)
    end
    (;indices=stream.indices,upper)
end

function mature_label(labels::AbstractDict,row::PreparedIssue,time)
    time>=row.timing.label_assumed_ready || return (:pending,nothing)
    haskey(labels,row.issue) || return (:missing,nothing)
    target=labels[row.issue]
    valid_number(target) && target>=0 || return (:invalid,nothing)
    (:complete,Float64(target))
end

function gap_certificate(issued,resolved,misses,nominal)
    issued==0 && return (;issued,resolved,misses,unknown=0,lower_gap=NaN,upper_gap=NaN)
    low,high=setprecision(BigFloat,256) do
        n=BigFloat(issued);a=BigFloat(nominal)
        (prevfloat(Float64(BigFloat(misses)/n-a)),
         nextfloat(Float64(BigFloat(misses+issued-resolved)/n-a)))
    end
    (;issued,resolved,misses,unknown=issued-resolved,lower_gap=low,upper_gap=high)
end

"""Corrected phase or pooled replay; features, labels and readiness are shared."""
function replay_controllers(prepared,centers,scales,labels::AbstractDict,cal::CompletedCalibration;
                            schedule,gamma,nominal=0.1,asof::DateTime,flush=true)
    nominal=nominal_value(nominal);gamma=gain_value(gamma)
    schedule in (:phase,:fifo) || throw(ArgumentError("invalid schedule"))
    stream=checked_stream(prepared,centers,scales,cal,asof)
    period=CMR.issue_timing(prepared[first(stream.indices)].issue,stream.policy).sufficient_phase_period_minutes
    states=schedule===:phase ? period : 1
    counts=zeros(Int,states);misses=zeros(Int,states);alpha=fill(Float64(nominal),states)
    upper=fill(NaN,length(prepared));alpha_issue=copy(upper)
    status=fill(:not_issued,length(prepared));resolved_at=Union{Nothing,DateTime}[nothing for _ in prepared]
    previous=zeros(Int,states);queue=Int[];events=NamedTuple[];max_batch=0
    function resolve(index,slot,time)
        kind,target=mature_label(labels,prepared[index],time)
        kind===:pending && return false
        status[index]=kind;resolved_at[index]=time
        if kind===:complete
            miss=target>upper[index]
            counts[slot]+=1;misses[slot]+=miss
            alpha[slot]=nominal+gamma*(counts[slot]*nominal-misses[slot])
            push!(events,(;kind=:feedback,index,issue=prepared[index].issue,time,slot,error=miss,target))
        else
            push!(events,(;kind=:censored,index,issue=prepared[index].issue,time,slot,reason=kind))
        end
        true
    end
    for index in stream.indices
        time=prepared[index].issue
        slot=schedule===:phase ? CMR.minute_phase(time,stream.policy) : 1
        batch=0
        if schedule===:phase
            old=previous[slot]
            if old>0
                resolve(old,slot,time) || error("phase reused before label readiness")
                batch=1
            end
        else
            while !isempty(queue) && prepared[first(queue)].timing.label_assumed_ready<=time
                old=popfirst!(queue);resolve(old,1,time);batch+=1
            end
        end
        max_batch=max(max_batch,batch)
        status[index]=:pending;alpha_issue[index]=alpha[slot]
        upper[index]=upper_bound(alpha[slot],centers[index],scales[index],cal.sorted_scores)
        schedule===:phase ? (previous[slot]=index) : push!(queue,index)
    end
    if flush
        pending=schedule===:phase ? filter(!=(0),previous) : copy(queue)
        sort!(pending;by=index->prepared[index].issue)
        for index in pending
            slot=schedule===:phase ? CMR.minute_phase(prepared[index].issue,stream.policy) : 1
            resolve(index,slot,asof)
        end
    end
    n=length(stream.indices);r=sum(counts);m=sum(misses)
    proof=gap_certificate(n,r,m,nominal)
    residual=m-r*nominal+(sum(alpha)-states*nominal)/gamma
    applicable=schedule===:phase && r==n
    bound=applicable ? count(>(0),counts)*max(nominal+gamma*(1-nominal),1-nominal+gamma*nominal)/(n*gamma) : NaN
    selected_upper=upper[stream.indices]
    (;indices=stream.indices,upper,alpha_issue,final_alpha=alpha,status,resolved_at,events,
      resolved_counts=counts,miss_counts=misses,certificate=proof,accounting_residual=residual,
      phase_period_minutes=period,max_feedback_batch=max_batch,pathwise_bound=bound,
      pathwise_bound_applicable=applicable,diagnostic_flush=flush,
      full_count=count(==(Inf),selected_upper),empty_count=count(<(0.0),selected_upper),
      finite_negative_count=count(value->isfinite(value) && value<0,selected_upper),
      asof,mode=stream.mode,policy=stream.policy,gamma,nominal,schedule,
      point_model_sha256=cal.point_model_sha256,calibration_schema=CALIBRATION_SCHEMA,
      availability_kind=AVAILABILITY,measured_receipts_available=false,
      calibration_purpose=cal.purpose,requires_source_bound_refits=true)
end
"""
    replay_phase(prepared, centers, scales, labels, calibration; gamma,
                 nominal=0.1, asof, flush=true)

Replay policy phases under a complete-label dictionary assumption. A missing
ready label is censored, so this path is not a late/graced receipt adapter.
Default sufficient phase period is31 minutes. `flush` changes no issued bound.
"""
replay_phase(prepared,centers,scales,labels,cal;kwargs...)=
    replay_controllers(prepared,centers,scales,labels,cal;schedule=:phase,kwargs...)
"""
    replay_fifo(prepared, centers, scales, labels, calibration; gamma,
                nominal=0.1, asof, flush=true)

Replay pooled FIFO feedback on the same feature schedule and complete-label
dictionary convention. Missing outcomes remain unknown in the certificate.
Use a future shared event adapter for matched late/graced comparisons.
"""
replay_fifo(prepared,centers,scales,labels,cal;kwargs...)=
    replay_controllers(prepared,centers,scales,labels,cal;schedule=:fifo,kwargs...)
