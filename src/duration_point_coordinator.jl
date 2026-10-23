abstract type _Message end
struct _IssueMessage <: _Message
    producer::Int
    payload::NamedTuple
end
struct _ChunkMessage <: _Message
    producer::Int
    rows::Vector{NamedTuple}
end
struct _DoneMessage <: _Message
    producer::Int
    result::NamedTuple
end
struct _ErrorMessage <: _Message
    producer::Int
    error::CapturedException
end
struct _ProducerFailure
    error::CapturedException
    cancelling::Bool
end
const DesignRow=Union{BTrainingDesignRow,BForecastDesignRow}
const JoinedRow=Union{BTrainingRow,BForecastRow}
struct _JointScope
    state::_ScopeState
    view::PointInputView
    laws::NTuple{2,D.DurationModel}
    training::NTuple{2,_BatchTrainingContext}
    forecast::NTuple{2,_ForecastDesignContext}
    advance::Function
    snapshot::Function
end
function _with_joint_design_scope(f::Function,v::PointInputView,laws::NTuple{2,D.DurationModel},training_bindings,forecast_bindings)
    _recheck(v);_need((laws[1].beta,laws[2].beta)==(5,20)&&laws[1].boundary==laws[2].boundary,"B two-law order/boundary differs")
    delay=_int(training_bindings.delay_hours,0,1);boundary=laws[1].boundary
    expected=_masks(v,boundary,forecast_bindings.validation_start,forecast_bindings.validation_end,delay)
    _need(training_bindings.mask==expected.training&&forecast_bindings.mask==expected.forecast,"B mature mask/per-lead union differs")
    state=_new_scope()
    bindings=ntuple(bi->_make_binding(v,laws[bi],expected.training,training_bindings.fold_id,boundary,delay),2)
    train=ntuple(bi->_BatchTrainingContext(bindings[bi],v,laws[bi],copy(expected.training),state,bi),2)
    forecast=ntuple(2) do bi
        id=_id((;schema="b_forecast_design_context_v1",role=:validation_affine_design,fold_id=bindings[bi].fold_id,expected_gain_boundary=boundary,law_boundary=laws[bi].boundary,law_ids=(laws[1].model_id,laws[2].model_id),input_admission_sha256=v.input_admission_sha256,source_sha256=v.source_sha256,source_hashes=_hashdata(v.source_hashes),forecast_union_mask_sha256=_sha(reinterpret(UInt8,collect(expected.forecast))),contract_sha256=CONTRACT))
        _ForecastDesignContext(bindings[bi],v,laws[bi],copy(expected.forecast),state,bi,boundary,id)
    end
    errors=Any[];result=nothing;tracker=nothing
    try
        tracker=C.with_current_state_tracker(v.wind;source_sha256=v.source_sha256) do (advance,snapshot)
            scope=_JointScope(state,v,laws,train,forecast,advance,snapshot)
            try result=f(scope) catch e;push!(errors,e);end
            nothing
        end
    catch e;push!(errors,e)
    finally
        state.active=false;state.pair_id="";empty!(state.schedules)
        try _recheck(v) catch e;push!(errors,e);end
    end
    _raise(errors)
    (;value=result,wind_checkpoint=tracker.checkpoint,training_bindings=bindings,counts=copy(state.counts))
end
function _prepare_shared_forcing(scope::_JointScope,payloads,current_wind)
    s=scope.state;_scope(s);_need(!s.busy&&length(payloads)==9,"B preparation is nonreentrant and requires nine payloads")
    i=payloads[1].index;issue=payloads[1].issue
    training=scope.training[1].mask[i];contexts=training ? scope.training : scope.forecast
    for physical in 1:9
        _need(payloads[physical].index==i&&payloads[physical].issue==issue,"B barrier origins differ")
        _payload(contexts[1],payloads[physical],physical)
    end
    _need(current_wind.issue==issue,"B current wind issue differs")
    s.busy=true
    try
        s.epoch=i;empty!(s.schedules)
        pivot=training ? 0 : something(findfirst(p->p.a_model_available,payloads),0)
        for physical in 1:9
            p=payloads[physical]
            if p.a_model_available
                D._basis(map(x->K._number(x,"historical basis"),p.issue_basis))
                if physical!=pivot
                    push!(s.schedules,D._mean_base_schedule(p.issue_checkpoint,1 ./ p.taus,(p.sigma,p.sigma),p.current_driver.pressure,1.,6))
                    _count!(s,:base_schedule_calls)
                else push!(s.schedules,(;))
                end
            else push!(s.schedules,(;))
            end
        end
        any_available=any(p->p.a_model_available,payloads)
        forces=Vector{NamedTuple}(undef,2);pivots=Union{Nothing,NamedTuple}[nothing,nothing]
        for bi in 1:2
            if !any_available||current_wind.status!=:available
                forces[bi]=D._unavailable(current_wind.reason)
            elseif training
                forces[bi]=D._mean_forcing(scope.laws[bi],current_wind,6);_count!(s,:forcing_DP_calls)
            else
                p=payloads[pivot]
                out=D.mean_affine_rollout(scope.laws[bi],current_wind,p.issue_checkpoint,p.issue_basis;lambdas=1 ./ p.taus,sigmas=(p.sigma,p.sigma),pressure=p.current_driver.pressure,target_observation_variance=1.,horizon=6)
                forces[bi]=out;pivots[bi]=out
                _count!(s,:public_forecast_pivot_calls);_count!(s,:forcing_DP_calls);_count!(s,:base_schedule_calls)
                if out.status==:available
                    _count!(s,:coefficient_evaluations);s.schedules[pivot]=(;offsets=out.offsets)
                end
            end
        end
        statuses=ntuple(bi->forces[bi].status,2);reasons=ntuple(bi->forces[bi].status==:available ? :available : forces[bi].reason,2)
        qmeans=ntuple(bi->forces[bi].status==:available ? forces[bi].qmeans : nothing,2)
        role=training ? :in_sample_two_stage_training : :validation_affine_design
        pairid=_id((;schema="b_shared_forcing_pair_v1",index=i,issue,role,source_sha256=scope.view.source_sha256,input_admission_sha256=scope.view.input_admission_sha256,current_state_id=current_wind.state_id,law_ids=(scope.laws[1].model_id,scope.laws[2].model_id),status=statuses,reason=reasons))
        s.pair_id=pairid
        SharedForcingPair(s,current_task(),i,issue,6,role,scope.view.source_sha256,scope.view.input_admission_sha256,current_wind.state_id,current_wind.input_prefix_sha256,(scope.laws[1].model_id,scope.laws[2].model_id),statuses,reasons,qmeans,pivot,(pivots[1],pivots[2]),pairid)
    finally s.busy=false
    end
end
function _join(design::DesignRow,row,v,delay)
    d=design.values
    _need(d.issue==row.issue&&isequal(d.anchor,row.anchor)&&row.source_sha256==v.source_sha256&&row.dst_delay_hours==delay,"B joined A row differs")
    _need(row.completed_target_leads==(4,6),"B joined lead order differs")
    for k in 1:2
        c=Adapter.forecast_clock(d.issue,(4,6)[k],delay)
        _need((row.target_tag[k],row.target_complete[k],row.target_receipt[k])==(c.target_tag,c.target_complete,c.target_receipt),"B joined target clock differs")
        expected=d.index+(4,6)[k]<=v.source_hours ? _clean(v,15,d.index+(4,6)[k]) : NaN
        _need(isequal(expected,row.target[k]),"B joined target source differs")
    end
    design isa BTrainingDesignRow ? BTrainingRow(design,row.target) : BForecastRow(design,row.target)
end
const _NO_PROBE=(args...)->nothing
function _barrier_payload(scope,p,physical,epoch)
    _need(keys(p)==ISSUE_KEYS&&p.index isa Int&&p.index==epoch&&p.issue==Adapter._shift(scope.view.input.time[epoch],1),"B barrier payload index/keys differ")
    recipe=Adapter.settings()[2*physical-1];v=scope.view;b=scope.training[1].binding
    _need(p.taus==recipe.taus&&p.sigma==recipe.sigma&&p.source_sha256==v.source_sha256&&p.delay_hours==b.delay_hours&&p.implementation_hashes==LOADED_STREAM&&p.availability_kind==:assumed_completed_hour,"B barrier payload source/recipe differs")
    _need(p.resume_checkpoint.last_index==epoch&&p.resume_checkpoint.last_issue==p.issue&&p.resume_checkpoint.prefix_sha256==p.input_prefix_sha256,"B barrier A prefix/resume differs")
    _need(p.driver_tag==v.input.time[epoch]&&p.driver_receipt==p.issue&&p.anchor_index==epoch-b.delay_hours&&_hex(p.input_prefix_sha256),"B barrier history clocks differ")
    clock=Adapter.forecast_clock(p.issue,4,b.delay_hours)
    _need((p.anchor_tag,p.anchor_complete,p.anchor_receipt)==(clock.anchor_tag,clock.anchor_complete,clock.anchor_receipt),"B barrier anchor clocks differ")
    anchor=epoch-b.delay_hours>=1 ? _clean(v,15,epoch-b.delay_hours) : NaN
    _need(isequal(anchor,p.anchor),"B barrier anchor source differs")
    speed=_clean(v,8,epoch);bz=_clean(v,6,epoch);pressure=_clean(v,12,epoch)
    known=all(isfinite,(speed,bz,pressure))&&speed>=0
    if known
        driver=p.current_driver
        _need(driver!==nothing&&driver.index==epoch&&driver.start==v.input.time[epoch]&&driver.stop==p.issue&&isequal(driver.q,v.wind[epoch].q)&&isequal(driver.pressure,pressure),"B barrier current driver differs")
    else
        _need(p.current_driver===nothing,"B barrier missing driver differs")
    end
    available=p.active&&p.consecutive_valid>=168&&isfinite(p.anchor)
    reason=!isfinite(p.anchor) ? :anchor_missing : !p.active ? :known_driver_missing : p.consecutive_valid<168 ? :warmup : :model
    _need(p.active==known&&p.a_model_available==available&&Symbol(p.a_reason)==reason,"B barrier A availability/reset differs")
    if scope.training[1].mask[epoch]||scope.forecast[1].mask[epoch]
        contexts=scope.training[1].mask[epoch] ? scope.training : scope.forecast
        for bi in 1:2;_payload(contexts[bi],p,physical);end
    end
    nothing
end
function _coordinate_issue_streams(v::PointInputView,laws,sink::Function;delay_hours,chunk_hours,training_mask,forecast_union_mask,fold_id="initial",validation_start=laws[1].boundary,validation_end=Adapter._shift(last(v.input.time),1),probe=_NO_PROBE)
    delay=_int(delay_hours,0,1);chunk=_int(chunk_hours,1,MAX_CHUNK)
    _need(Threads.nthreads()==1,"B coordinator requires one Julia thread")
    _need(v.evidence_mode==:synthetic_only||probe===_NO_PROBE,"B actual producer probe substitution forbidden")
    _with_joint_design_scope(v,laws,(;delay_hours=delay,mask=training_mask,fold_id),(;mask=forecast_union_mask,validation_start,validation_end)) do scope
        events=Channel{_Message}(9);acks=[Channel{Int}(1) for _ in 1:9];cancelling=Ref(false)
        tasks=Task[];errors=Any[];cleanup=Any[];results=Vector{NamedTuple}(undef,9)
        pending=[Dict{Int,Tuple{DesignRow,DesignRow}}() for _ in 1:9]
        expected=ones(Int,9);last_chunk=zeros(Int,9);done=falses(9);epoch=1
        payloads=Vector{NamedTuple}(undef,9);seen=falses(9);max_pending=0;max_queued=0
        for physical in 1:9
            recipe=Adapter.settings()[2*physical-1]
            task=@async begin
                try
                    out=Stream.stream_basis(v.input;delay_hours=delay,taus=recipe.taus,sigma=recipe.sigma,chunk_hours=chunk,
                        issue_sink=p->begin
                            replacement=probe(:before_issue,physical,p)
                            message_payload=replacement isa NamedTuple ? replacement : p
                            _need(message_payload isa NamedTuple,"B synthetic probe payload differs")
                            put!(events,_IssueMessage(physical,message_payload));ack=take!(acks[physical])
                            _need(ack==p.index,"B acknowledgment epoch differs")
                            probe(:after_ack,physical,p)
                        end,
                        sink=rows->put!(events,_ChunkMessage(physical,rows)))
                    put!(events,_DoneMessage(physical,out));out
                catch e
                    captured=CapturedException(e,catch_backtrace())
                    if !cancelling[]
                        try put!(events,_ErrorMessage(physical,captured)) catch;end
                    end
                    _ProducerFailure(captured,cancelling[])
                end
            end
            push!(tasks,task)
        end
        try
            while !all(done)
                max_queued=max(max_queued,Base.n_avail(events));message=take!(events);physical=message.producer
                _need(1<=physical<=9&&!done[physical],"B duplicate/unknown completed producer")
                if message isa _IssueMessage
                    p=message.payload
                    _need(!seen[physical]&&p.index==expected[physical]==epoch&&p.issue==Adapter._shift(v.input.time[epoch],1),"B duplicate/wrong-origin barrier message")
                    payloads[physical]=p;seen[physical]=true;expected[physical]+=1
                    if all(seen)
                        probe(:barrier_ready,0,epoch)
                        for recipe_index in 1:9;_barrier_payload(scope,payloads[recipe_index],recipe_index,epoch);end
                        current=scope.advance(payloads[1].issue);_count!(scope.state,:wind_advances)
                        if training_mask[epoch]||forecast_union_mask[epoch]
                            pair=_prepare_shared_forcing(scope,payloads,current)
                            for recipe_index in 1:9
                                p=payloads[recipe_index]
                                designs=ntuple(bi->training_mask[epoch] ? _training_design(scope.training[bi],p,pair) : forecast_design(scope.forecast[bi],p,pair),2)
                                pending[recipe_index][epoch]=designs
                                _count!(scope.state,:affine_recipe_groups,p.a_model_available ? 1 : 0)
                            end
                            probe(:designs_ready,0,epoch)
                        end
                        scope.state.pair_id=""
                        max_pending=max(max_pending,2*sum(length,pending));_need(max_pending<=18*(chunk+1),"B pending design bound exceeded")
                        for recipe_index in 1:9;put!(acks[recipe_index],epoch);end
                        epoch+=1;fill!(seen,false)
                    end
                elseif message isa _ChunkMessage
                    _need(1<=length(message.rows)<=chunk,"B A chunk bound differs")
                    joined=(JoinedRow[],JoinedRow[])
                    for row in message.rows
                        index=last_chunk[physical]+1
                        _need(row.issue==Adapter._shift(v.input.time[index],1),"B A chunks reordered or duplicated")
                        if training_mask[index]||forecast_union_mask[index]
                            _need(haskey(pending[physical],index),"B target section preceded its design barrier")
                            designs=pop!(pending[physical],index)
                            for bi in 1:2;push!(joined[bi],_join(designs[bi],row,v,delay));end
                        end
                        last_chunk[physical]=index
                    end
                    for bi in 1:2;isempty(joined[bi])||sink(physical,bi,joined[bi]);end
                elseif message isa _DoneMessage
                    _need(expected[physical]==v.source_hours+1&&last_chunk[physical]==v.source_hours&&isempty(pending[physical]),"B producer done before complete source/chunks")
                    results[physical]=message.result;done[physical]=true
                elseif message isa _ErrorMessage
                    throw(message.error.ex)
                end
            end
        catch e;push!(errors,e)
        finally
            cancelling[]=true;scope.state.pair_id=""
            close(events);foreach(close,acks)
            for task in tasks
                try
                    value=fetch(task)
                    if value isa _ProducerFailure
                        push!(cleanup,(;cancelling=value.cancelling,error=sprint(showerror,value.error)))
                        if !value.cancelling&&!any(e->e===value.error.ex,errors);push!(errors,value.error.ex);end
                    end
                catch e;push!(errors,e)
                end
            end
            _need(all(istaskdone,tasks),"B producer Task escaped scope")
            try probe(:cleanup_done,0,(;terminated_tasks=count(istaskdone,tasks),cleanup)) catch e;push!(errors,e);end
        end
        _raise(errors)
        _need(epoch==v.source_hours+1&&all(isempty,pending),"B source/pending tail differs")
        (;streams=results,max_pending_design_rows=max_pending,max_queued_messages=max_queued,producer_tasks=9,terminated_tasks=count(istaskdone,tasks),cleanup,counts=copy(scope.state.counts))
    end
end
