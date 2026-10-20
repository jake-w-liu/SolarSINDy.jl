module CMEProspectiveMonitorTests

using Test, Dates, HTTP, SHA
module Before
    include(joinpath(@__DIR__,"cme_prospective","frozen","prospective_probability_monitor_before_v1.jl"))
end
module After
    include(joinpath(@__DIR__,"..","tools","cme_prospective","prospective_probability_monitor.jl"))
end
const OLD=Before.CMEProspectiveMonitor
const NEW=After.CMEProspectiveMonitor
const URL="https://example.invalid"
const FROZEN=joinpath(@__DIR__,"cme_prospective","frozen","prospective_probability_monitor_before_v1.jl")
@test bytes2hex(sha256(read(FROZEN)))=="3fa61b6884a8903cea7603f6b85bc4e29127158bfe91824672436158110f619a"

function timeout_boundary(exception)
    try
        HTTP.Connections.try_with_timeout(_->throw(exception),2.)
    catch captured
        return captured
    end
    error("throwing timeout-boundary control unexpectedly returned")
end

@testset "Installed HTTP timeout boundary retains a retryable transport cause" begin
    io=Base.IOError("read: connection reset by peer (ECONNRESET)",Int32(-4077))
    captured=timeout_boundary(io)
    @test captured isa CapturedException
    @test captured.ex===io
    direct=HTTP.ConnectError(URL,io);wrapped=HTTP.ConnectError(URL,captured)
    @test OLD.recoverable_acquisition(direct)
    @test !OLD.recoverable_acquisition(wrapped)
    @test NEW.recoverable_acquisition(direct)
    @test NEW.recoverable_acquisition(wrapped)
    @test NEW.recoverable_acquisition(HTTP.RequestError(nothing,captured))
    @test occursin("connection reset by peer (ECONNRESET)",sprint(showerror,wrapped))
    for exception in (EOFError(),HTTP.TimeoutError(1))
        @test NEW.recoverable_acquisition(HTTP.ConnectError(URL,timeout_boundary(exception)))
    end
end

@testset "Unknown failures, local I/O and interruptions remain fatal" begin
    io=Base.IOError("local file unavailable",Int32(-2))
    @test !NEW.recoverable_acquisition(io)
    @test !NEW.recoverable_acquisition(CapturedException(io,backtrace()))
    for exception in (ErrorException("changed model hash"),ArgumentError("malformed response"),InterruptException())
        @test !NEW.recoverable_acquisition(exception)
        @test !NEW.recoverable_acquisition(HTTP.ConnectError(URL,exception))
        @test !NEW.recoverable_acquisition(HTTP.ConnectError(URL,timeout_boundary(exception)))
    end
    for status in (408,425,429,500,503,599)
        @test NEW.recoverable_acquisition(ErrorException("DONKI HTTP status $status"))
    end
    for status in (200,400,401,403,404,410,600)
        @test !NEW.recoverable_acquisition(ErrorException("DONKI HTTP status $status"))
    end
    @test !NEW.recoverable_acquisition(ErrorException("parse error: DONKI HTTP status 503"))
    @test !NEW.recoverable_acquisition(HTTP.ConnectError(URL,ErrorException("DONKI HTTP status 503")))
    nested=EOFError()
    for _ in 1:31;nested=CapturedException(nested,backtrace());end
    @test NEW.recoverable_acquisition(HTTP.ConnectError(URL,nested))
    nested=CapturedException(nested,backtrace())
    @test !NEW.recoverable_acquisition(HTTP.ConnectError(URL,nested))
end

@testset "A captured reset records acquisition failure and recovers on the fixed cadence" begin
    mktempdir() do store
        time=Ref(DateTime(2020));verifications=Ref(0);captures=Ref(0);issues=Ref(0)
        statuses=Any[];delays=Float64[];verify_order=Int[]
        actions=(verify=()->begin verifications[]+=1;push!(verify_order,captures[]);nothing end,
            capture=()->begin
                captures[]+=1;time[]+=Second(5)
                captures[]==1 && throw(HTTP.ConnectError(URL,timeout_boundary(EOFError())))
                "capture-2"
            end,
            issue_all=capture->begin @test capture=="capture-2";issues[]+=1;"batch-2" end)
        count=NEW.collect!(store,actions;deadline=DateTime(2020,1,2),interval_seconds=3600.,max_cycles=2,
            clock=()->time[],sleep_fn=seconds->begin push!(delays,seconds);time[]+=Millisecond(round(Int,1000seconds));end,
            status_fn=value->push!(statuses,value))
        @test count==2
        @test verifications[]==2
        @test verify_order==[0,1]
        @test captures[]==2
        @test issues[]==1
        @test delays==[3595.]
        @test length(statuses)==2
        @test statuses[1].status=="acquisition_failed"
        @test statuses[1].capture_key==statuses[1].batch_key==""
        @test statuses[2].status=="ok"
        @test statuses[2].capture_key=="capture-2"
        @test statuses[2].batch_key=="batch-2"
        @test DateTime(statuses[2].started_at_utc)-DateTime(statuses[1].started_at_utc)==Hour(1)
        @test all(!value.outcomes_retrieved && !value.scores_written for value in statuses)
        @test !ispath(joinpath(store,".prospective_collector.claim"))
    end
end

@testset "Integrity and publication failures stop without automatic retry" begin
    for phase in (:verify,:capture,:issue)
        mktempdir() do store
            counts=Dict(:verify=>0,:capture=>0,:issue=>0,:sleep=>0,:status=>0)
            actions=(verify=()->begin counts[:verify]+=1;phase==:verify && error("changed source");nothing end,
                capture=()->begin counts[:capture]+=1;phase==:capture && error("malformed capture");"capture" end,
                issue_all=capture->begin counts[:issue]+=1;phase==:issue && error("publication failed");"batch" end)
            @test_throws ErrorException NEW.collect!(store,actions;deadline=DateTime(2020,1,2),max_cycles=2,
                clock=()->DateTime(2020),sleep_fn=_->(counts[:sleep]+=1),status_fn=_->(counts[:status]+=1))
            @test counts[:verify]==1
            @test counts[:capture]==(phase==:verify ? 0 : 1)
            @test counts[:issue]==(phase==:issue ? 1 : 0)
            @test counts[:sleep]==counts[:status]==0
            @test !ispath(joinpath(store,".prospective_collector.claim"))
        end
    end
    mktempdir() do store
        claim=joinpath(store,".prospective_collector.claim");mkdir(claim)
        actions=(verify=()->error("must not execute"),capture=()->error("must not execute"),issue_all=_->error("must not execute"))
        @test_throws Base.IOError NEW.collect!(store,actions;deadline=DateTime(2020,1,2),clock=()->DateTime(2020))
        @test isdir(claim)
        rm(claim)
    end
end

end # module
