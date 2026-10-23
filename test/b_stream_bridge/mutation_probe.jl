module BridgeMutationProbe
using Dates, JSON3, Test, SHA
include("isolated_sources.jl")
const W=JSON3.read(read(joinpath(@__DIR__,"WIND_LITERAL_ORACLES_V1.json"),String))
const IDS=JSON3.read(read(joinpath(@__DIR__,"STATE_ID_LITERAL_ORACLES_V1.json"),String))
const O=JSON3.read(read(joinpath(@__DIR__,"A_ANALYTIC_LITERAL_ORACLES_V1.json"),String))
const SOURCE=bytes2hex(sha256("independent-bridge-fixture-a-v1"))
const T0=DateTime(2000)
function one_replace(body,old,new)
    length(findall(old,body))==1 || error("mutation location is not unique")
    replace(body,old=>new)
end
function b_mismatches(parent,src)
    C=parent.DurationCurrentStateStream;D=parent.DurationConditionedForcing
    rows=[D.WindHour(T0+Hour(Int(r.hour)),T0+Hour(Int(r.hour)+1),
        DateTime(Dates.UTInstant{Millisecond}(Millisecond(Int64(r.receipt_ms)))),
        r.bz===nothing ? NaN : Float64(r.bz),r.q===nothing ? NaN : Float64(r.q)) for r in W.rows]
    bad=Ref(0)
    consume=callbacks->begin
        advance=first(callbacks)
        for expected in W.expected
            actual=advance(DateTime(String(expected.issue)))
            id=only(x.state_id for x in IDS.records if x.group=="expected"&&x.issue==expected.issue)
            bad[]+=actual.state_id!=id
        end
    end
    Base.invokelatest(C.with_current_state_tracker,consume,rows;source_sha256=SOURCE)
    same=[D.WindHour(T0+Hour(i),T0+Hour(i+1),T0+Hour(i+1),-1.,0.) for i in 0:49]
    consume_age=callbacks->begin
        advance=first(callbacks)
        for expected in W.initial_same_state_50_expected
            actual=advance(DateTime(String(expected.issue)))
            id=only(x.state_id for x in IDS.records if x.group=="initial_same_state_50_expected"&&x.issue==expected.issue)
            bad[]+=actual.state_id!=id
        end
    end
    Base.invokelatest(C.with_current_state_tracker,consume_age,same;source_sha256=SOURCE)
    bad[]
end
function a_mismatches(parent,src)
    A=parent.IntervalMeanForecaster
    v=zeros(17,2);v[8,:].=1000.;v[6,:].=[-1.49,-2.49];v[15,:].=-11.
    input=A.HourlyInputs([T0,T0+Hour(1)],v;source_sha256=SOURCE)
    bad=0
    for delay in (0,1)
        seen=Any[]
        Base.invokelatest(A.stream_basis,input;delay_hours=delay,taus=(2.,12.),sigma=1.,issue_sink=p->push!(seen,p))
        for i in 1:2
            p=seen[i];expected=getproperty(O.cases,Symbol("l$(delay)_index$(i)_issue"))
            matrix=[Float64(expected.basis[j][k]) for j in 1:2,k in 1:2]
            bad+=!isapprox(p.issue_basis,matrix;atol=5e-12,rtol=5e-14)
            bad+=p.issue_checkpoint.boundary!=T0+Hour(i)
            cov=[Float64(expected.covariance[j][k]) for j in 1:2,k in 1:2]
            bad+=!isapprox(A.DstIntervalAdapter.Replay._matrix(p.issue_checkpoint.covariance),cov;atol=5e-11,rtol=5e-14)
        end
    end
    bad
end
const CASES=[
    (name="receipt_equality_excluded",kind="tracker",transform=b->one_replace(b,"row.completed<=issue && row.received<=issue || break","row.completed<=issue && row.received<issue || break")),
    (name="age49_at48",kind="tracker",transform=b->one_replace(b,"run>=49 ? 49 : 0","run>=48 ? 49 : 0")),
    (name="South_q0_treated_N",kind="tracker",transform=b->one_replace(b,"next=D._state(row)","next=row.q==0 ? :N : D._state(row)")),
    (name="stale_latest_substitution",kind="tracker",transform=b->one_replace(b,"tracker.rows[count].tag!=latest_tag","false")),
    (name="missing_row_not_hashed",kind="tracker",transform=b->begin
        marker="previous,state,age,run=_trace_step(row,previous,state,age,run)"
        first,last=split(b,marker;limit=2)
        old="SHA.update!(context,codeunits(line));bytes+=ncodeunits(line)"
        one_replace(first*marker*replace(last,old=>"D._known(row) && SHA.update!(context,codeunits(line));bytes+=ncodeunits(line)";count=1),"# Only a complete result commits","# Only a complete result commits")
    end),
    (name="L1_sealed_exposed_as_issue",kind="stream",transform=b->begin
        for part in ("boundary","mean","covariance")
            b=replace(b,"model_state."*part=>"sealed."*part)
        end
        b
    end),
    (name="historical_basis_transposed",kind="stream",transform=b->one_replace(b," : _matrix(model_basis),"," : permutedims(_matrix(model_basis)),")),
    (name="issue_basis_shared_between_hours",kind="stream",transform=b->begin
        b=one_replace(b,"module DstIntervalStream","module DstIntervalStream\nconst _MUTANT_SHARED_BASIS=zeros(2,2)")
        one_replace(b," : _matrix(model_basis),"," : (_MUTANT_SHARED_BASIS .= _matrix(model_basis); _MUTANT_SHARED_BASIS),")
    end)
]
results=Any[]
@testset "Executed numerical/clock/hash/copy mutation controls" begin
    control=IsolatedBridgeSources.with_sources(b_mismatches)
    @test control==0
    control_a=IsolatedBridgeSources.with_sources(a_mismatches)
    @test control_a==0
    for case in CASES
        kwargs=case.kind=="tracker" ? (;tracker_transform=case.transform) : (;stream_transform=case.transform)
        # A parse/import/runtime exception fails this probe; it is never a kill.
        differences=IsolatedBridgeSources.with_sources(case.kind=="tracker" ? b_mismatches : a_mismatches;kwargs...)
        @test differences>0
        push!(results,(;name=case.name,kind=case.kind,loaded=true,executed=true,
            independent_literal_mismatches=differences,numerical_or_clock_hash_copy_kill=true))
    end
end
out=only(ARGS);path=joinpath(out,"mutation_results.json")
ispath(path)&&error("refusing to replace mutation result")
write(path,JSON3.write((;schema="b_stream_bridge_mutations_v1",complete=true,julia_version=string(VERSION),
    unmodified_controls_passed=true,valid_mutants=length(results),invalid_import_parse_kills=0,results)))
println("B_STREAM_BRIDGE_MUTATIONS_COMPLETE valid_mutants=",length(results))
end
