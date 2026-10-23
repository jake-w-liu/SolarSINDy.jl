module BStreamBridgeTests
using Test, Dates, SHA, JSON3, LinearAlgebra
using SolarSINDy
const A=SolarSINDy.IntervalMeanForecaster
const S=A.DstIntervalStream
const D=SolarSINDy.DurationConditionedForcing
const C=SolarSINDy.DurationCurrentStateStream
const SOURCE=bytes2hex(sha256("independent-bridge-fixture-a-v1"))
const T0=DateTime(2000)
load(name)=JSON3.read(read(joinpath(@__DIR__,name),String))
const W=load("WIND_LITERAL_ORACLES_V1.json")
const IDS=load("STATE_ID_LITERAL_ORACLES_V1.json")
const O=load("A_ANALYTIC_LITERAL_ORACLES_V1.json")
const P=load("A_PREFIX_LITERAL_ORACLES_V1.json")
include("isolated_sources.jl")
@testset "Frozen pre-code oracle and A187 source byte pins" begin
    for (name,hash) in (
        "A_ANALYTIC_LITERAL_ORACLES_V1.json"=>"4bcc23a61bdaf5b576c6686765d960eea1157487a5084b8bdb94f7a489df0133",
        "A_PREFIX_LITERAL_ORACLES_V1.json"=>"16c4a07c91a14dc77807648caaa4f8bcbd9c2706bf3bc0f856b8377c5594d4d1",
        "GUARD_LITERAL_ORACLES_V1.json"=>"d09a0c19f34334136edae37f7e259b69265aa55243253fce6161a27fdb4917c1",
        "STATE_ID_LITERAL_ORACLES_V1.json"=>"f19e098dc76c0f2dabdf24b386f19347d4167df3a968469e67804a34c094b511",
        "WIND_LITERAL_ORACLES_V1.json"=>"7e5a798b7b2ccbac74f652a00d9ed6f45dac48bb93682f2ba30608502e3f0686")
        @test bytes2hex(sha256(read(joinpath(@__DIR__,name))))==hash
    end
    for (name,hash) in (
        "DstIntervalObserver.jl"=>"48e2af386d2b9b1a2b8afb4366e1c48508261e54a81447d438fd8a419581b816",
        "DstIntervalReplay.jl"=>"f2962c092bc6bfd6be82a5b8ad95e2c494dddbc4a2addda4f57140e31d0c6fa5",
        "sandu_fullchannel_features.jl"=>"2306f63cadfb5d4ae16579cabb114b813f1f41678aaa6dbdd90caad316ff792c",
        "DstIntervalAdapter.jl"=>"b432a8fc65d3e425a67ff2b225a6c0e5323ae7fdb8822dd228677b7998a5fe5a",
        "DstIntervalStream.jl"=>"7a6319dd3b572f9337c3573a50620cf666ada77c4b4c13bb995564807c1f037d")
        @test bytes2hex(sha256(read(joinpath(@__DIR__,"frozen_a187",name))))==hash
    end
end
f64(x)=x===nothing ? NaN : Float64(x)
wind()=[D.WindHour(T0+Hour(Int(r.hour)),T0+Hour(Int(r.hour)+1),
    DateTime(Dates.UTInstant{Millisecond}(Millisecond(Int64(r.receipt_ms)))),f64(r.bz),f64(r.q)) for r in W.rows]
function check_literal(state,expected,group)
    @test (state.status,state.reason,state.state,state.age,state.q,state.input_prefix_sha256)==
        (Symbol(expected.status),Symbol(expected.reason),Symbol(expected.state),Int(expected.age),expected.q,String(expected.input_prefix_sha256))
    frozen=only(r for r in IDS.records if r.group==group&&r.issue==expected.issue)
    @test state.state_id==frozen.state_id
    @test state==D.current_state(group=="expected" ? wind() :
        [D.WindHour(T0+Hour(i),T0+Hour(i+1),T0+Hour(i+1),-1.,0.) for i in 0:49],state.issue)
end
@testset "Independent exact archival trace, byte prefixes and state IDs" begin
    @test Dates.value(T0)==63082368000000
    data=wind()
    result=C.with_current_state_tracker(data;source_sha256=SOURCE) do (advance,snapshot)
        @test snapshot().last_issue===nothing && snapshot().consumed_count==0
        for expected in W.expected
            state=advance(DateTime(String(expected.issue)))
            check_literal(state,expected,"expected")
            before=snapshot()
            @test advance(state.issue)==state
            @test snapshot()==before
        end
        @test snapshot().consumed_count==10
        "literal_complete"
    end
    @test result.value=="literal_complete" && result.checkpoint.consumed_count==10
    same=[D.WindHour(T0+Hour(i),T0+Hour(i+1),T0+Hour(i+1),-1.,0.) for i in 0:49]
    C.with_current_state_tracker(same;source_sha256=SOURCE) do (advance,_)
        for expected in W.initial_same_state_50_expected
            check_literal(advance(DateTime(String(expected.issue))),expected,"initial_same_state_50_expected")
        end
    end
    @test C.with_current_state_tracker(D.WindHour[];source_sha256=SOURCE) do (advance,_)
        advance(T0)==D.current_state(D.WindHour[],T0)
    end |> x->x.value
end

struct ZeroAxisVector{T}<:AbstractVector{T}
    values::Vector{T}
end
Base.size(v::ZeroAxisVector)=size(v.values)
Base.axes(v::ZeroAxisVector)=(0:length(v.values)-1,)
Base.getindex(v::ZeroAxisVector,i::Int)=v.values[i+1]
function changed_checkpoint(cp;kwargs...)
    names=fieldnames(C.CurrentStateCheckpoint)
    values=NamedTuple{names}(Tuple(getfield(cp,n) for n in names))
    next=merge(values,(;kwargs...))
    C.CurrentStateCheckpoint(Tuple(next)...)
end
function rehashed(cp)
    changed_checkpoint(cp;checkpoint_id=D._hash(C._checkpoint_data(cp)))
end
@testset "Admission, snapshot ownership and atomic failures" begin
    calls=Ref(0);f=_->(calls[]+=1)
    data=wind()
    @test_throws ArgumentError C.with_current_state_tracker(f,vcat(data[1:1],data[1:1]);source_sha256=SOURCE)
    @test_throws ArgumentError C.with_current_state_tracker(f,reverse(data);source_sha256=SOURCE)
    @test_throws ArgumentError C.with_current_state_tracker(f,reshape(data,:,1);source_sha256=SOURCE)
    @test_throws ArgumentError C.with_current_state_tracker(f,ZeroAxisVector(data);source_sha256=SOURCE)
    @test_throws ArgumentError C.with_current_state_tracker(f,Any[data[1],1];source_sha256=SOURCE)
    @test_throws ArgumentError C.with_current_state_tracker(f,data;source_sha256=uppercase(SOURCE))
    late=[D.WindHour(T0,T0+Hour(1),T0+Hour(3),1.,0.),D.WindHour(T0+Hour(1),T0+Hour(2),T0+Hour(2),-1.,0.)]
    @test_throws ArgumentError C.with_current_state_tracker(f,late;source_sha256=SOURCE)
    invalid=[D.WindHour(T0,T0+Hour(1),T0+Hour(1),1.,1.)]
    @test_throws ArgumentError C.with_current_state_tracker(f,invalid;source_sha256=SOURCE)
    @test calls[]==0
    @test_throws ArgumentError C.with_current_state_tracker(f,fill(data[1],500001);source_sha256=SOURCE)
    @test_throws ArgumentError D.WindHour(T0,T0+Hour(1),T0+Hour(1),true,0.)
    @test_throws ArgumentError D.WindHour(T0,T0+Hour(1),T0+Hour(1),-1.,Inf)
    @test_throws ArgumentError D.WindHour(T0,T0+Hour(1),T0+Hour(1),-1.,big"1e-1000")
    escaped=Ref{Any}();retained=Ref{Any}()
    escaped_snapshot=Ref{Any}()
    cp=C.with_current_state_tracker(data;source_sha256=SOURCE) do (advance,snapshot)
        escaped[]=advance
        escaped_snapshot[]=snapshot
        advance(T0+Hour(3));before=snapshot()
        @test_throws ArgumentError advance(T0+Hour(2))
        @test snapshot()==before
        task=@async try advance(T0+Hour(4));false catch e;e isa ArgumentError end
        @test fetch(task)
        @test snapshot()==before
        data[1]=D.WindHour(T0,T0+Hour(1),T0+Hour(1),-1.,10.)
        @test advance(T0+Hour(3)).state_id==only(r for r in IDS.records if r.group=="expected"&&r.issue=="2000-01-01T03:00:00.000").state_id
        retained[]=snapshot()
        nothing
    end
    @test_throws ArgumentError escaped[](T0+Hour(4))
    @test_throws ArgumentError escaped_snapshot[]()
    @test retained[]==cp.checkpoint
    minimum_issue=DateTime(Dates.UTInstant{Millisecond}(Millisecond(typemin(Int64))))
    C.with_current_state_tracker(wind();source_sha256=SOURCE) do (advance,snapshot)
        before=snapshot()
        @test_throws OverflowError advance(minimum_issue)
        @test snapshot()==before
    end
    @test_throws ErrorException C.with_current_state_tracker(_->error("deliberate callback failure"),wind();source_sha256=SOURCE)
    partial=C.with_current_state_tracker(wind();source_sha256=SOURCE) do (advance,_)
        advance(T0+Hour(8))
    end
    resumed=C.with_current_state_tracker(wind();source_sha256=SOURCE,checkpoint=partial.checkpoint) do (advance,_)
        advance(T0+Hour(11))
    end
    @test resumed.value==D.current_state(wind(),T0+Hour(11))
    relocated=deepcopy(partial.checkpoint)
    @test C.with_current_state_tracker(_->nothing,wind();source_sha256=SOURCE,checkpoint=relocated).checkpoint==relocated
    for bad in (changed_checkpoint(relocated;checkpoint_id=repeat("0",64)),
                rehashed(changed_checkpoint(relocated;age=4)),
                rehashed(changed_checkpoint(relocated;consumed_count=0)),
                rehashed(changed_checkpoint(relocated;schema="old_a_checkpoint")))
        @test_throws ArgumentError C.with_current_state_tracker(f,wind();source_sha256=SOURCE,checkpoint=bad)
    end
    @test_throws ArgumentError C.with_current_state_tracker(f,wind();source_sha256=repeat("b",64),checkpoint=relocated)
    future=vcat(wind(),[D.WindHour(T0+Hour(11),T0+Hour(12),T0+Hour(12),1.,0.)])
    @test_throws ArgumentError C.with_current_state_tracker(f,future;source_sha256=SOURCE,checkpoint=relocated)
    other=C.with_current_state_tracker(wind();source_sha256=repeat("b",64)) do (advance,_)
        advance(T0+Hour(8))
    end
    @test other.value==partial.value && other.checkpoint.checkpoint_id!=partial.checkpoint.checkpoint_id
    zeros=[D.WindHour(T0,T0+Hour(1),T0+Hour(1),-1.,0.),D.WindHour(T0+Hour(1),T0+Hour(2),T0+Hour(2),-0.,-0.)]
    C.with_current_state_tracker(zeros;source_sha256=SOURCE) do (advance,_)
        state=advance(T0+Hour(2));@test isequal(state,D.current_state(zeros,T0+Hour(2)))
        @test state.state==:N && signbit(state.q)
    end
    for (bz,q) in ((NaN,2.),(-1.,NaN))
        partial=[D.WindHour(T0,T0+Hour(1),T0+Hour(1),bz,q)]
        result=C.with_current_state_tracker(partial;source_sha256=SOURCE) do (advance,_)
            advance(T0+Hour(1))
        end
        @test result.value==D.current_state(partial,T0+Hour(1))
        @test result.value.reason==:latest_driver_missing
    end
end

@testset "Loaded source mutation rejects at entry and final success boundary" begin
    IsolatedBridgeSources.with_sources() do parent,src
        M=parent.DurationCurrentStateStream;E=parent.DurationConditionedForcing
        rows=[E.WindHour(r.tag,r.completed,r.received,r.bz,r.q) for r in wind()]
        path=joinpath(src,"duration_current_state_stream.jl");body=read(path)
        for altered in (path,joinpath(src,"duration_conditioned_forcing.jl"),joinpath(src,"interval_mean_forecaster","DstIntervalStream.jl"))
            original=read(altered)
            try
                write(altered,vcat(original,UInt8[0x0a]))
                @test_throws ArgumentError Base.invokelatest(M.with_current_state_tracker,_->nothing,rows;source_sha256=SOURCE)
            finally
                write(altered,original)
            end
        end
        try
            alter=callbacks->begin
                advance=first(callbacks)
                advance(T0+Hour(3));write(path,vcat(body,UInt8[0x0a]));nothing
            end
            @test_throws ArgumentError Base.invokelatest(M.with_current_state_tracker,alter,rows;source_sha256=SOURCE)
        finally
            write(path,body)
        end
        @test Base.invokelatest(M.source_identity)==C.source_identity()
    end
end

function a_fixture(n;gap=false)
    values=ones(17,n);values[8,:].=400.
    values[6,:]=[-5+3sin(i/43) for i in 1:n]
    values[12,:]=[2+.5cos(i/17) for i in 1:n]
    values[15,:]=[-60+40sin(i/61) for i in 1:n]
    if gap;values[6,169]=NaN;values[15,168]=NaN;values[12,175]=-1.;end
    A.HourlyInputs([T0+Hour(i-1) for i in 1:n],values;source_sha256=SOURCE)
end
function run_a(input,delay,tau=(2.,12.),sigma=1.;issue_sink=nothing,chunk=24,checkpoint=nothing,stop=length(input.time))
    rows=NamedTuple[]
    result=A.stream_basis(input;delay_hours=delay,taus=tau,sigma,chunk_hours=chunk,
        sink=p->append!(rows,p),issue_sink,checkpoint,stop_after_index=stop)
    rows,result
end
matrix(x)=[Float64(x[i][j]) for i in 1:2,j in 1:2]
function analytic(payload,name)
    x=getproperty(O.cases,Symbol(name));tol=O.mathematical_reference_tolerance
    @test collect(payload.issue_checkpoint.mean)==Float64.(x.mean)
    @test isapprox(S.R._matrix(payload.issue_checkpoint.covariance),matrix(x.covariance);atol=Float64(tol.covariance_atol),rtol=Float64(tol.rtol))
    @test isapprox(payload.issue_basis,matrix(x.basis);atol=Float64(tol.basis_atol),rtol=Float64(tol.rtol))
end
@testset "Detached A issue states match independent OU and prefix literals" begin
    v=zeros(17,2);v[8,:].=1000.;v[6,:].=[-1.49,-2.49];v[15,:].=-11.
    input=A.HourlyInputs([T0,T0+Hour(1)],v;source_sha256=SOURCE)
    for delay in (0,1)
        payloads=Any[];run_a(input,delay;issue_sink=x->push!(payloads,x))
        for i in 1:2
            p=payloads[i];analytic(p,"l$(delay)_index$(i)_issue")
            @test p.issue_checkpoint.boundary==p.issue==T0+Hour(i)
            @test p.input_prefix_sha256==getproperty(P.by_delay,Symbol(string(delay))).steps[i].prefix_sha256
            @test p.current_driver.q==Float64(O.q[i])
            @test !(:target in keys(p)) && !(:scoreable in keys(p))
            @test axes(p.issue_basis)==(1:2,1:2)
            @test p.a_reason==(delay==1&&i==1 ? "anchor_missing" : "warmup")
            @test p.resume_checkpoint.last_index==i
            if delay==1
                @test p.resume_checkpoint.sealed.boundary==p.issue-Hour(1)
                @test p.resume_checkpoint.pending.index==i
                if i==2
                    sealed=O.cases.l1_index2_sealed
                    @test isapprox(S.R._matrix(p.resume_checkpoint.sealed.covariance),matrix(sealed.covariance);atol=5e-11,rtol=5e-14)
                    @test isapprox(S._matrix(p.resume_checkpoint.basis),matrix(sealed.basis);atol=5e-12,rtol=5e-14)
                end
            else
                @test p.resume_checkpoint.pending===nothing
            end
        end
    end
    for delay in (0,1)
        input=a_fixture(180;gap=true);ordinary,baseline=run_a(input,delay)
        seen=Any[]
        mutated,summary=run_a(input,delay;issue_sink=p->begin
            push!(seen,p)
            p.issue_basis===nothing || fill!(p.issue_basis,1e10)
            p.issue_checkpoint===nothing || fill!(collect(p.issue_checkpoint.mean),1e10)
        end)
        @test isequal(ordinary,mutated) && isequal(summary,baseline)
        @test seen[169].issue_checkpoint===nothing && seen[169].issue_basis===nothing
        @test seen[170].consecutive_valid==1 && seen[170].reset_tag==input.time[170]
        @test seen[168].a_reason==(delay==0 ? "anchor_missing" : "model")
        @test seen[169].a_reason==(delay==0 ? "known_driver_missing" : "anchor_missing")
        @test seen[168].issue_checkpoint!==nothing
        cp=seen[90].resume_checkpoint
        tail,resumed=run_a(input,delay;checkpoint=cp)
        @test isequal(tail,ordinary[91:end]) && isequal(resumed.checkpoint,baseline.checkpoint)
        calls=Ref(0);none,nochange=run_a(input,delay;checkpoint=cp,stop=90,issue_sink=_->(calls[]+=1))
        @test isempty(none)&&calls[]==0&&nochange.checkpoint===cp
        published=Ref(0)
        @test_throws ErrorException A.stream_basis(input;delay_hours=delay,taus=(2.,12.),sigma=1.,chunk_hours=1,
            sink=p->(published[]+=length(p)),issue_sink=p->p.index==3 ? error("deliberate issue sink failure") : nothing)
        @test published[]==2
    end
    @test_throws ArgumentError run_a(a_fixture(1),0;issue_sink=1)
end

# Regression comparison comes from immutable A187 source bytes, distinct from
# the independent arithmetic expectations above. Serialize fields by value,
# including Float64 bits, so different module paths cannot mask differences.
module FrozenA187
include("frozen_a187/DstIntervalAdapter.jl")
include("frozen_a187/DstIntervalStream.jl")
end
function token(io,x)
    if x isa Float64;write(io,UInt8('F'),htol(reinterpret(UInt64,x)))
    elseif x isa DateTime;write(io,UInt8('D'),htol(Int64(Dates.value(x))))
    elseif x isa Bool;write(io,UInt8('B'),UInt8(x))
    elseif x isa Integer;write(io,UInt8('I'),htol(Int64(x)))
    elseif x isa AbstractString||x isa Symbol
        s=String(x);write(io,UInt8('S'),htol(UInt64(ncodeunits(s))),codeunits(s))
    elseif x===nothing;write(io,UInt8('0'))
    elseif x isa NamedTuple
        write(io,UInt8('N'),htol(UInt64(length(x))));for (k,v) in pairs(x);token(io,k);token(io,v);end
    elseif x isa Tuple||x isa AbstractArray
        write(io,UInt8('T'),htol(UInt64(length(x))));x isa AbstractArray&&token(io,size(x));for v in x;token(io,v);end
    elseif x isa AbstractDict
        write(io,UInt8('M'));for k in sort(collect(keys(x));by=string);token(io,k);token(io,x[k]);end
    else
        token(io,String(nameof(typeof(x))));for k in fieldnames(typeof(x));token(io,k);token(io,getfield(x,k));end
    end
end
payload(x)=begin io=IOBuffer();token(io,x);take!(io) end
@testset "Tolerance-free old A full-grid row/summary/checkpoint parity" begin
    original=FrozenA187.DstIntervalStream
    input=a_fixture(180;gap=true)
    old=original.HourlyInputs(input.time,input.values;source_sha256=SOURCE)
    for tau in S.A.TAUS,sigma in S.A.SIGMAS,delay in (0,1)
        expected=NamedTuple[]
        baseline=original.stream_basis(old;delay_hours=delay,taus=tau,sigma,chunk_hours=24,sink=p->append!(expected,p))
        actual,summary=run_a(input,delay,tau,sigma)
        @test payload((actual,summary))==payload((expected,baseline))
        hooked,hs=run_a(input,delay,tau,sigma;issue_sink=_->nothing)
        @test payload((hooked,hs))==payload((expected,baseline))
        prefix,partial=run_a(input,delay,tau,sigma;stop=90,issue_sink=_->nothing)
        tail,resumed=run_a(input,delay,tau,sigma;checkpoint=partial.checkpoint,issue_sink=_->nothing)
        @test payload(vcat(prefix,tail))==payload(expected)
        @test payload(resumed.checkpoint)==payload(baseline.checkpoint)
        for chunk in (1,7,4096)
            rows,rs=run_a(input,delay,tau,sigma;chunk,issue_sink=_->nothing)
            @test payload(rows)==payload(expected) && payload(rs.checkpoint)==payload(baseline.checkpoint)
        end
    end
end

println("B_STREAM_BRIDGE_FOCUSED_COMPLETE")
end # module
