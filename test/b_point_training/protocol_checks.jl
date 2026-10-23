function protocol_view(n=64)
    start=DateTime(1900);raw=ones(17,n)
    for i in 1:n;raw[6,i]=(i-1)%10<5 ? 1. : -2.5;raw[8,i]=400.;raw[12,i]=2.;raw[15,i]=-60.;end
    source=T._sha(reinterpret(UInt8,vec(raw)));times=[start+Hour(i) for i in 0:n-1]
    input=T.A.HourlyInputs(times,raw;source_sha256=source)
    wind=[T.D.WindHour(t,t+Hour(1),t+Hour(1),raw[6,i],max(raw[8,i]*max(-raw[6,i],0.)/1000-.49,0.)) for (i,t) in enumerate(times)]
    view=T._point_input_view(input,wind,(;evidence_mode=:synthetic_only,input_admission_sha256=T._sha("generated protocol admission"),contract_sha256=T.CONTRACT))
    (;view,input,wind,raw)
end
function protocol_checks(;measurements=nothing)
    # Primitive telemetry fixture: equal-valued rows with distinct identity must
    # remain distinct; a buffered/offered/current reference is counted once.
    events=Channel{T._Message}(9);offered=fill!(Vector{Union{Nothing,T._Message}}(undef,9),nothing)
    rows=NamedTuple[];message=T._ChunkMessage(1,rows);put!(events,message);offered[1]=message
    q=T._queue_diagnostics(events,offered,message)
    @test q.buffered==1&&q.capacity==9&&q.unique_messages==1&&q.queued_A_chunks==1&&q.with_consumer==1
    equal_distinct=T._ChunkMessage(1,copy(rows));@test equal_distinct.rows==rows&&equal_distinct.rows!==rows
    offered[2]=equal_distinct;q=T._queue_diagnostics(events,offered)
    @test q.unique_messages==2&&q.queued_A_chunks==2
    close(events);fill!(offered,nothing)
    @test_throws InvalidStateException T._offer_event!(events,offered,message)
    @test all(isnothing,offered)
    fixture=protocol_view();v=fixture.view;boundary=first(v.input.time)+Hour(32);stop=first(v.input.time)+Hour(64)
    laws=ntuple(i->T.D.fit_duration(v.wind,boundary;beta=(5,20)[i],source_sha256=v.source_sha256),2)
    masks=T._masks(v,boundary,boundary,stop,0)
    @test !any(masks.training)&&!any(masks.forecast)
    original=copy(v.input.values);fixture.raw[15,1]=17.;fixture.input.values[15,2]=18.
    fixture.wind[1]=T.D.WindHour(fixture.wind[1].tag,fixture.wind[1].completed,fixture.wind[1].received,NaN,NaN)
    @test isequal(v.input.values,original)&&T._recheck(v)
    expired=T._with_batch_training_context(identity,v,laws[1];fold_id="protocol",boundary,delay_hours=0,training_mask=masks.training,contract_sha256=T.CONTRACT)
    @test_throws ArgumentError T._scope(expired.scope)
    cross=T._with_batch_training_context(v,laws[1];fold_id="protocol",boundary,delay_hours=0,training_mask=masks.training,contract_sha256=T.CONTRACT) do context
        fetch(@async try T._scope(context.scope);nothing catch e;e;end)
    end
    @test cross isa ArgumentError
    function run(probe;delay=0,chunk=1)
        mask=T._masks(v,boundary,boundary,stop,delay)
        T._coordinate_issue_streams(v,laws,(physical,beta,rows)->@test(isempty(rows));delay_hours=delay,chunk_hours=chunk,training_mask=mask.training,forecast_union_mask=mask.forecast,validation_start=boundary,validation_end=stop,probe)
    end
    for delay in (0,1),chunk in (1,2,7,64)
        all_seen=Dict{Int,Set{Int}}();ready=Set{Int}();after=Ref(0);cleanup=Ref(0)
        probe=(event,physical,payload)->begin
            if event==:before_issue
                push!(get!(all_seen,payload.index,Set{Int}()),physical)
            elseif event==:barrier_ready
                @test get(all_seen,payload,Set{Int}())==Set(1:9);push!(ready,payload)
            elseif event==:after_ack
                @test payload.index in ready;after[]+=1
            elseif event==:cleanup_done
                @test payload.terminated_tasks==9&&payload.offered_slots_cleared;cleanup[]+=1
            end
            nothing
        end
        result=run(probe;delay,chunk)
        @test length(ready)==64&&after[]==9*64&&cleanup[]==1&&result.value.terminated_tasks==9
        @test result.value.event_channel_capacity==9&&result.value.max_buffered_messages<=9
        @test result.value.max_queued_messages<=18&&result.value.max_queued_unique_messages<=18
        @test result.value.max_A_chunks_with_consumer<=10
        @test result.value.max_pending_design_rows<=18*(chunk+1)
        @test result.counts[:wind_advances]==64&&get(result.counts,:forcing_DP_calls,0)==0
        measurements===nothing||push!(measurements,(;source_rows=64,delay_hours=delay,chunk_hours=chunk,buffered=result.value.max_buffered_messages,capacity=result.value.event_channel_capacity,available_including_offers=result.value.max_queued_messages,unique_messages=result.value.max_queued_unique_messages,unique_A_chunks=result.value.max_queued_A_chunks,chunks_with_consumer=result.value.max_A_chunks_with_consumer,terminated=result.value.terminated_tasks))
    end
    for (case,recipe) in ((:wrong_origin,1),(:callback_throw,1),(:callback_throw,9),(:error_after_ack,1))
        terminated=Ref(0);failed=false
        probe=(event,physical,payload)->begin
            event==:cleanup_done&&(@test(payload.offered_slots_cleared);terminated[]=payload.terminated_tasks)
            if event==:before_issue&&physical==recipe&&payload.index==2
                case==:wrong_origin&&return merge(payload,(;index=3))
                case==:callback_throw&&throw(ArgumentError("injected recipe $recipe callback"))
            elseif case==:error_after_ack&&event==:after_ack&&physical==recipe&&payload.index==1
                yield();throw(ArgumentError("injected callback after acknowledgment"))
            end
            nothing
        end
        try run(probe) catch e;failed=e isa ArgumentError||e isa CompositeException;end
        @test failed&&terminated[]==9&&T._recheck(v)
    end
    # Real design origins exercise all18 continuations before A acknowledgments.
    physical=protocol_view(512);pv=physical.view;pb=first(pv.input.time)+Hour(256);pe=first(pv.input.time)+Hour(512)
    pl=ntuple(i->T.D.fit_duration(pv.wind,pb;beta=(5,20)[i],source_sha256=pv.source_sha256),2)
    for delay in (0,1)
        pm=T._masks(pv,pb,pb,pe,delay);digests=Vector{Vector{String}}()
        for mutate in (false,true)
            ready=Set{Int}();contexts=[SHA.SHA2_256_CTX() for _ in 1:18];terminated=Ref(0)
            probe=(event,recipe,payload)->begin
                if event==:designs_ready;push!(ready,payload)
                elseif event==:after_ack
                    (pm.training[payload.index]||pm.forecast[payload.index])&&@test(payload.index in ready)
                    if mutate&&recipe==1&&payload.issue_checkpoint!==nothing
                        @test !ismutabletype(typeof(payload.issue_checkpoint))
                        @test payload.issue_checkpoint.mean isa Tuple&&payload.issue_checkpoint.covariance isa Tuple
                        @test all(row->row isa Tuple,payload.issue_checkpoint.covariance)
                        @test payload.resume_checkpoint.sealed.mean isa Tuple
                        payload.issue_basis[1,1]+=1000.
                    end
                elseif event==:cleanup_done;terminated[]=payload.terminated_tasks
                end
                nothing
            end
            result=T._coordinate_issue_streams(pv,pl,(recipe,bi,rows)->begin for row in rows;SHA.update!(contexts[2*(recipe-1)+bi],P._basis_bytes(row));end end;delay_hours=delay,chunk_hours=7,training_mask=pm.training,forecast_union_mask=pm.forecast,validation_start=pb,validation_end=pe,probe)
            @test length(ready)==count(pm.training)+count(pm.forecast)&&terminated[]==9&&result.value.terminated_tasks==9
            push!(digests,[bytes2hex(SHA.digest!(x)) for x in contexts])
        end
        @test digests[1]==digests[2]
    end
    true
end
