function scalar_snapshot(value)
    if value isa Union{CompletedMaximumController,CompletedCalibration,CompletedSourceDBDT.MaximumTicket}
        names=fieldnames(typeof(value))
        return NamedTuple{names}(Tuple(scalar_snapshot(getfield(value,name)) for name in names))
    elseif value isa AbstractDict
        return Dict(key=>scalar_snapshot(item) for (key,item) in value)
    elseif value isa AbstractArray
        return map(scalar_snapshot,value)
    end
    deepcopy(value)
end

@testset "Independent scalar Float64 zero-rounding boundaries" begin
    oracle=JSON3.read(read(joinpath(@__DIR__,"frozen","SCALAR_RANGE_PRECODE_ORACLES_V2.json"),String))
    @test String(oracle.delta)=="2^-1074" && oracle.pre_code
    delta=setprecision(BigFloat,256) do
        BigFloat(2)^(-1074)
    end
    @test Float64(delta)==nextfloat(0.0)
    @test Float64(delta/2)==0.0 && Float64(delta/4)==0.0
    @test Float64(3delta/4)==nextfloat(0.0)
    for sign in (-1,1)
        for (multiple,expected) in ((0,true),(1//4,false),(1//2,false),(3//4,true),(1,true),(2,true))
            value=sign*delta*multiple
            @test CompletedSourceDBDT.valid_number(value)==expected
            if expected
                cal=calibration(scores=fill(value,9))
                @test cal.chronological_scores==fill(Float64(value),9)
            else
                @test_throws ArgumentError calibration(scores=fill(value,9))
            end
        end
    end
    @test CompletedSourceDBDT.valid_number(-0.0)
    @test signbit(only(calibration(scores=[-0.0]).chronological_scores))
    @test !CompletedSourceDBDT.valid_number(true)
    @test !CompletedSourceDBDT.valid_number(BigFloat(Inf))
    @test !CompletedSourceDBDT.valid_number(BigFloat(NaN))

    mag=Dict(ISSUE+Minute(k)=>(0.0,0.0) for k in -40:1)
    row=prepare_issue(mag,ISSUE)
    tiny=delta/4
    @test CompletedSourceDBDT.mature_label(Dict(ISSUE=>tiny),row,ISSUE+Minute(31))==(:invalid,nothing)
    @test CompletedSourceDBDT.mature_label(Dict(ISSUE=>-tiny),row,ISSUE+Minute(31))==(:invalid,nothing)
    @test CompletedSourceDBDT.mature_label(Dict(ISSUE=>delta),row,ISSUE+Minute(31))==(:complete,nextfloat(0.0))
    @test CompletedSourceDBDT.mature_label(Dict(ISSUE=>BigFloat(0)),row,ISSUE+Minute(31))==(:complete,0.0)
    for replay in (replay_phase,replay_fifo)
        invalid=replay([row],[0.0],[1.0],Dict(ISSUE=>tiny),calibration();gamma=.1,asof=ISSUE+Minute(31))
        @test invalid.status==[:invalid] && invalid.certificate.unknown==1 && invalid.certificate.resolved==0
        subnormal=replay([row],[0.0],[1.0],Dict(ISSUE=>delta),calibration();gamma=.1,asof=ISSUE+Minute(31))
        @test subnormal.status==[:complete] && subnormal.certificate.misses==1 && subnormal.certificate.resolved==1
        @test only(subnormal.events).target==nextfloat(0.0)
    end
    for bad in (tiny,-tiny)
        @test_throws ArgumentError bind_predictions([row],X->[bad])
        @test_throws ArgumentError static_bounds([row],[bad],[1.0],calibration();asof=ISSUE)
        state=CompletedMaximumController(calibration();start=ISSUE,gamma=.1)
        before=scalar_snapshot(state)
        @test_throws ArgumentError issue!(state,row,bad,1.0)
        @test isequal(scalar_snapshot(state),before)
    end
    accepted=bind_predictions([row],X->[delta])
    @test accepted.centers==[nextfloat(0.0)]

    # Validate the whole batch before any clock, retained source or ticket mutation.
    for order in (false,true), bad in (tiny,-tiny)
        state=CompletedMaximumController(calibration();start=ISSUE,gamma=.1,rolling_capacity=2)
        issue!(state,row,0.0,1.0)
        batch=[(;center=ISSUE+Minute(1),rate=1.0),(;center=ISSUE+Minute(2),rate=bad)]
        order && reverse!(batch)
        before=scalar_snapshot(state)
        @test_throws ArgumentError receive!(state,batch;received=ISSUE+Minute(2)+Second(45))
        @test isequal(scalar_snapshot(state),before)
    end
    state=CompletedMaximumController(calibration();start=ISSUE,gamma=.1)
    issue!(state,row,0.0,1.0)
    events=receive!(state,[(;center=ISSUE+Minute(1),rate=delta)];received=ISSUE+Minute(1)+Second(45))
    @test state.resolved==state.misses==1 && only(events).error
    @test state.pending[1].rates[1]==state.seen[ISSUE+Minute(1)]==nextfloat(0.0)
    events=receive!(state,[(;center=ISSUE+Minute(k),rate=BigFloat(0)) for k in 2:30];received=ISSUE+Minute(31))
    @test state.mature_scores==1 && isempty(state.pending)
    @test only(filter(event->event.kind===:score,events)).target==nextfloat(0.0)

    for bad in (tiny,-tiny)
        vectors=Dict(ISSUE=>(BigFloat(0),BigFloat(0)),ISSUE+Minute(1)=>(bad,BigFloat(0)))
        receipts=materialize_receipts(vectors,[ISSUE+Minute(1)];asof=ISSUE+Minute(2))
        @test isempty(receipts.batches) && only(receipts.rejections).reason===:invalid_ground_rate
    end
    vectors=Dict(ISSUE=>(BigFloat(0),BigFloat(0)),ISSUE+Minute(1)=>(delta,BigFloat(0)))
    receipts=materialize_receipts(vectors,[ISSUE+Minute(1)];asof=ISSUE+Minute(2))
    @test isempty(receipts.rejections) && only(only(receipts.batches).observations).rate==nextfloat(0.0)
end
