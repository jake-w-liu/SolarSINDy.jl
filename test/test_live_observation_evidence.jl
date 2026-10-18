module LiveObservationEvidenceTests
using Test, Dates, SHA, JSON3, HTTP, CSV, DataFrames

const HELPER=joinpath(@__DIR__,"..","examples","live_observation_evidence.jl")
@testset "Future observation evidence implementation exists" begin
    @test isfile(HELPER)
end
if isfile(HELPER)
include(HELPER)
const E=LiveObservationEvidence
const TAG=DateTime(2026,10,15,9)
const START=DateTime(2026,10,15,9,57)
const COMPLETE=START+Second(2)
const RAW=Vector{UInt8}(codeunits("[{\"time_tag\":\"2026-10-15T09:00:00\",\"dst\":-11}]"))
const RAW_REVISION=Vector{UInt8}(codeunits("[{\"time_tag\":\"2026-10-15T09:00:00\",\"dst\":-12}]"))
const MANIFEST=repeat("a",64)
function fixture_log(path; observation=missing)
    table=DataFrame(issue_time_utc=[DateTime(2026,10,15,8,45)],
        latest_dst_time_utc=[DateTime(2026,10,15,8)],target_time_utc=[TAG],
        model_step_hours=[1],model_version=["base"],
        sub_hourly_model_version=["test-exact-model"],
        superlearner_manifest_sha256=[MANIFEST],
        served_pred_dst_nt=[-10.],served_pred_dst_ci05_nt=[-20.],served_pred_dst_ci95_nt=[0.],
        observation_dst_nt=Union{Missing,Float64}[observation])
    CSV.write(path,table)
    return table
end
function capture(root,bytes=RAW; started=START,completed=COMPLETE,status=200)
    E.capture_response!(root,HTTP.Response(status,["Content-Type"=>"application/json"],bytes);
        request_started_utc=started,receipt_completed_utc=completed,
        monotonic_started_ns=100,monotonic_completed_ns=200)
end
function prepare(root,receipt,table,log; value=-11.,previous=missing,time=COMPLETE+Second(1))
    binding=E.forecast_binding(table[1,:])
    entries=[(;binding,previous_observation=previous,observation=value)]
    E.prepare_verification!(root,receipt,entries;log_path=log,verification_started_utc=time)
end

@testset "Actual receipt clocks, exact bytes and target matching" begin
    mktempdir(realpath(tempdir())) do tmp
        root=joinpath(tmp,"evidence")
        receipt=capture(root)
        @test !occursin('\\',receipt.relative_path)
        @test_throws ArgumentError E.load_receipt(root,(;relative_path="../outside.json",sha256=receipt.sha256))
        @test_throws ArgumentError E.load_receipt(root,(;relative_path="C:/outside.json",sha256=receipt.sha256))
        record=E.load_receipt(root,receipt)
        @test record.source_url=="https://services.swpc.noaa.gov/products/kyoto-dst.json"
        @test DateTime(record.request_started_utc)==START
        @test DateTime(record.receipt_completed_utc)==COMPLETE
        @test record.body_sha256==bytes2hex(sha256(RAW))
        @test record.capture_source_sha256==bytes2hex(sha256(read(HELPER)))
        @test record.julia_version==string(VERSION) && record.julia_threads==Threads.nthreads()
        @test read(joinpath(root,String(record.raw_relative_path)))==RAW
        bound=E.bind_observation(root,receipt,TAG,-11.)
        @test bound.receipt_precedes_nominal_end
        @test DateTime(bound.target_tag_utc)==TAG
        @test DateTime(bound.target_nominal_end_utc)==DateTime(2026,10,15,10)
        @test_throws ArgumentError E.bind_observation(root,receipt,TAG,-12.)
        @test_throws ArgumentError E.bind_observation(root,receipt,TAG+Hour(1),-11.)
        @test_throws ArgumentError E.bind_observation(root,receipt,TAG+Minute(30),-11.)
        late=capture(root;started=START+Hour(1),completed=COMPLETE+Hour(1))
        @test !E.bind_observation(root,late,TAG,-11.).receipt_precedes_nominal_end
        rollback=capture(root;completed=START-Second(1))
        @test !E.load_receipt(root,rollback).clock_eligible
        backwards=E.capture_response!(root,HTTP.Response(200,RAW);
            request_started_utc=START,receipt_completed_utc=COMPLETE,
            monotonic_started_ns=200,monotonic_completed_ns=100)
        @test !E.load_receipt(root,backwards).clock_eligible
        failed=capture(root,UInt8[0xff,0,0x41];status=503)
        f=E.load_receipt(root,failed)
        @test f.http_status==503 && f.body_bytes==3
        @test read(joinpath(root,String(f.raw_relative_path)))==UInt8[0xff,0,0x41]
        @test_throws ArgumentError E.bind_observation(root,failed,TAG,-11.)
        invalid=capture(root,Vector{UInt8}(codeunits("not JSON")))
        @test_throws ArgumentError E.bind_observation(root,invalid,TAG,-11.)
        transport=E.capture_transport_failure!(root,ErrorException("timeout");
            request_started_utc=START,receipt_completed_utc=COMPLETE,
            monotonic_started_ns=100,monotonic_completed_ns=200)
        t=E.load_receipt(root,transport)
        @test t.capture_outcome=="transport_error" && t.body_bytes==0
        @test isempty(t.raw_relative_path) && t.body_sha256===nothing
        ambiguous=capture(root,Vector{UInt8}(codeunits("[{\"time_tag\":\"2026-10-15T09:00:00\",\"dst\":-11},{\"time_tag\":\"2026-10-15T09:00:00\",\"dst\":-12}]")))
        @test_throws ArgumentError E.bind_observation(root,ambiguous,TAG,-11.)
        rawpath=joinpath(root,String(record.raw_relative_path))
        write(rawpath,"tampered")
        @test_throws ArgumentError E.load_receipt(root,receipt)
    end
end

@testset "Intent, actual committed verification and immutable revisions" begin
    mktempdir(realpath(tempdir())) do tmp
        root=joinpath(tmp,"evidence");log=joinpath(tmp,"live.csv")
        table=fixture_log(log);receipt=capture(root)
        binding=E.forecast_binding(table[1,:])
        @test E._binding_key(binding)==E._binding_key(JSON3.read(JSON3.write(binding)))
        @test_throws ArgumentError prepare(root,receipt,table,log;time=START)
        extreme=copy(table);extreme.model_step_hours[1]=(Int(1)<<59)+1
        @test_throws ArgumentError E.forecast_binding(extreme[1,:])
        intent=prepare(root,receipt,table,log)
        @test length(E.verify_observation_evidence(root).pending_intents)==1
        @test_throws ArgumentError E.verify_observation_evidence(root;require_complete=true)
        @test_throws ArgumentError E.commit_verification!(root,intent;log_path=log,
            verification_completed_utc=COMPLETE+Second(2))
        table.observation_dst_nt[1]=-11.;CSV.write(log,table)
        @test_throws ArgumentError E.commit_verification!(root,intent;log_path=log,
            verification_completed_utc=START)
        first=E.commit_verification!(root,intent;log_path=log,
            verification_completed_utc=COMPLETE+Second(3))
        state=E.verify_observation_evidence(root;require_complete=true)
        @test state.sequence==1 && isempty(state.pending_intents)
        @test only(state.events).kind=="first_verification"
        @test only(state.events).observation==-11.
        @test DateTime(only(state.events).verification_completed_utc)==COMPLETE+Second(3)
        @test_throws ArgumentError E.commit_verification!(root,intent;log_path=log,
            verification_completed_utc=COMPLETE+Second(4))
        firstbytes=read(joinpath(root,String(first.relative_path)))
        revision=capture(root,RAW_REVISION;started=START+Hour(1),completed=COMPLETE+Hour(1))
        second_intent=prepare(root,revision,table,log;value=-12.,previous=-11.,time=COMPLETE+Hour(1)+Second(1))
        table.observation_dst_nt[1]=-12.;CSV.write(log,table)
        E.commit_verification!(root,second_intent;log_path=log,
            verification_completed_utc=COMPLETE+Hour(1)+Second(2))
        state=E.verify_observation_evidence(root;require_complete=true)
        @test state.sequence==2 && length(state.events)==2
        @test last(state.events).kind=="revised_observation"
        @test last(state.events).previous_observation==-11.
        @test last(state.events).prior_event_key==first.sha256*":1"
        @test read(joinpath(root,String(first.relative_path)))==firstbytes
        # A later actual reconfirmation of a legacy row never claims an unknown first receipt.
        other=joinpath(tmp,"legacy-evidence");legacy_receipt=capture(other,RAW_REVISION)
        legacy_intent=prepare(other,legacy_receipt,table,log;value=-12.,previous=-12.)
        E.commit_verification!(other,legacy_intent;log_path=log,verification_completed_utc=COMPLETE+Second(2))
        @test only(E.verify_observation_evidence(other).events).kind=="reconfirmed_legacy_observation"
        # Exact model/forecast identity is part of the evidence key.
        changed=copy(table);changed.superlearner_manifest_sha256[1]=repeat("b",64)
        @test E.forecast_binding(changed[1,:])!=E.forecast_binding(table[1,:])
        mismatch=prepare(root,revision,table,log;value=-12.,previous=-12.,time=COMPLETE+Hour(1)+Second(3))
        CSV.write(log,changed)
        @test_throws ArgumentError E.commit_verification!(root,mismatch;log_path=log,
            verification_completed_utc=COMPLETE+Hour(1)+Second(4))
    end
end

@testset "Atomic publication race and journal mutation detection" begin
    mktempdir(realpath(tempdir())) do root
        path=joinpath(root,"immutable.bin")
        @test_throws ArgumentError E.publish_immutable(path,UInt8[1,2,3];
            before_publish=()->write(path,UInt8[9,8,7]))
        @test read(path)==UInt8[9,8,7]
        @test isempty(filter(p->startswith(basename(p),"jl_"),readdir(root;join=true)))
    end
    mktempdir(realpath(tempdir())) do tmp
        root=joinpath(tmp,"evidence");log=joinpath(tmp,"live.csv")
        table=fixture_log(log);receipt=capture(root)
        intent=prepare(root,receipt,table,log)
        table.observation_dst_nt[1]=-11.;CSV.write(log,table)
        record=E.commit_verification!(root,intent;log_path=log,verification_completed_utc=COMPLETE+Second(2))
        path=joinpath(root,String(record.relative_path));bytes=read(path)
        bytes[end-1] ⊻= 0x01;write(path,bytes)
        @test_throws ArgumentError E.verify_observation_evidence(root)
    end
end

@testset "Shared raw bytes retain distinct receipt clocks and are rechecked" begin
    mktempdir(realpath(tempdir())) do tmp
        root=joinpath(tmp,"evidence");log=joinpath(tmp,"live.csv")
        one=fixture_log(log);two=copy(one);two.issue_time_utc[1]-=Minute(1)
        table=vcat(one,two);CSV.write(log,table)
        receipts=Any[]
        for i in 1:2
            receipt=capture(root;started=START+Hour(i-1),completed=COMPLETE+Hour(i-1))
            push!(receipts,receipt)
            intent=E.prepare_verification!(root,receipt,
                [(binding=E.forecast_binding(table[i,:]),previous_observation=missing,observation=-11.)];
                log_path=log,verification_started_utc=COMPLETE+Hour(i-1)+Second(1))
            table.observation_dst_nt[i]=-11.;CSV.write(log,table)
            E.commit_verification!(root,intent;log_path=log,
                verification_completed_utc=COMPLETE+Hour(i-1)+Second(2))
        end
        state=E.verify_observation_evidence(root;require_complete=true)
        @test length(readdir(joinpath(root,"raw")))==1
        @test receipts[1].sha256!=receipts[2].sha256
        @test [event.receipt_completed_utc for event in state.events]==string.([COMPLETE,COMPLETE+Hour(1)])
        @test [event.observation_binding.receipt_precedes_nominal_end for event in state.events]==[true,false]
        # After a successful audit, a new audit must read and check the actual
        # shared body again. No process-global cache may hide this mutation.
        record=E.load_receipt(root,receipts[1]);path=joinpath(root,String(record.raw_relative_path))
        original=read(path);changed=copy(original);changed[end-2]=UInt8('2');write(path,changed)
        @test_throws ArgumentError E.verify_observation_evidence(root)
        write(path,original)
        @test length(E.verify_observation_evidence(root;require_complete=true).events)==2
        duplicate=capture(root,Vector{UInt8}(codeunits(
            "[{\"time_tag\":\"2026-10-15T09:00:00\",\"dst\":-11},{\"time_tag\":\"2026-10-15T09:00:00\",\"dst\":-11}]")))
        bound=E.bind_observation(root,duplicate,TAG,-11.)
        @test bound.duplicate_equal_rows==2 && bound.provider_value==-11.
    end
end

@testset "Concurrent writers and interrupted head recovery preserve actual clocks" begin
    mktempdir(realpath(tempdir())) do tmp
        root=joinpath(tmp,"evidence");log=joinpath(tmp,"live.csv")
        one=fixture_log(log);two=copy(one);two.issue_time_utc[1]+=Minute(1)
        table=vcat(one,two);CSV.write(log,table);receipt=capture(root)
        intents=[E.prepare_verification!(root,receipt,
            [(binding=E.forecast_binding(table[i,:]),previous_observation=missing,observation=-11.)];
            log_path=log,verification_started_utc=COMPLETE+Second(1)) for i in 1:2]
        table.observation_dst_nt .= -11.;CSV.write(log,table)
        tasks=[Threads.@spawn(E.commit_verification!(root,intent;log_path=log,
            verification_completed_utc=COMPLETE+Second(2))) for intent in intents]
        results=fetch.(tasks)
        state=E.verify_observation_evidence(root;require_complete=true)
        @test state.sequence==2 && length(state.events)==2
        @test length(unique(result.sha256 for result in results))==2
        @test length(state.latest_by_forecast)==2
        @test !ispath(joinpath(root,"journal.lock"))
    end
    mktempdir(realpath(tempdir())) do tmp
        root=joinpath(tmp,"evidence");log=joinpath(tmp,"live.csv")
        table=fixture_log(log);receipt=capture(root);intent=prepare(root,receipt,table,log)
        table.observation_dst_nt[1]=-11.;CSV.write(log,table)
        @test_throws ErrorException E.commit_verification!(root,intent;log_path=log,
            verification_completed_utc=COMPLETE+Second(2),
            crash_hook=stage->stage==:after_journal_record ? error("independent crash") : nothing)
        @test_throws ArgumentError E.verify_observation_evidence(root)
        recovered=E.recover_observation_head!(root)
        @test recovered.sequence==1 && isempty(recovered.pending_intents)
        @test only(recovered.events).verification_completed_utc==string(COMPLETE+Second(2))
        @test E.verify_observation_evidence(root;require_complete=true).sequence==1
        @test !ispath(joinpath(root,"journal.lock"))
    end
end

@testset "Invalid clocks preserve bytes but cannot bind observations" begin
    mktempdir(realpath(tempdir())) do root
        for (started,completed,ns_start,ns_end) in (
                (START,START-Second(1),100,200),
                (START,COMPLETE,200,100))
            response=HTTP.Response(503,UInt8[0xff,0x00,0x41])
            ref=E.capture_response!(root,response;request_started_utc=started,
                receipt_completed_utc=completed,monotonic_started_ns=ns_start,
                monotonic_completed_ns=ns_end)
            record=E.load_receipt(root,ref)
            @test !record.clock_eligible
            @test DateTime(record.request_started_utc)==started && DateTime(record.receipt_completed_utc)==completed
            @test read(joinpath(root,String(record.raw_relative_path)))==UInt8[0xff,0x00,0x41]
            @test_throws ArgumentError E.bind_observation(root,ref,TAG,-11.)
        end
    end
end

@testset "Stale and incomparable receipts cannot overwrite a newer label" begin
    mktempdir(realpath(tempdir())) do tmp
        root=joinpath(tmp,"evidence");log=joinpath(tmp,"live.csv")
        table=fixture_log(log);binding=E.forecast_binding(table[1,:])
        old=capture(root;started=START-Minute(10),completed=COMPLETE-Minute(10))
        latest=capture(root,RAW_REVISION)
        intent=prepare(root,latest,table,log;value=-12.)
        table.observation_dst_nt[1]=-12.;CSV.write(log,table)
        E.commit_verification!(root,intent;log_path=log,verification_completed_utc=COMPLETE+Second(3))
        before=read(log)
        @test_throws ArgumentError prepare(root,old,table,log;value=-11.,previous=-12.,time=COMPLETE+Second(4))
        overlap=capture(root;started=START-Second(1),completed=COMPLETE+Second(10))
        @test_throws ArgumentError prepare(root,overlap,table,log;value=-11.,previous=-12.,time=COMPLETE+Second(11))
        ambiguous=capture(root)
        @test_throws ArgumentError prepare(root,ambiguous,table,log;value=-11.,previous=-12.,time=COMPLETE+Second(4))
        @test_throws ArgumentError prepare(root,latest,table,log;value=-12.,previous=-12.,time=COMPLETE+Second(2))
        @test read(log)==before
        state=E.verify_observation_evidence(root;require_complete=true)
        @test state.sequence==1 && only(state.events).observation==-12.
        @test length(readdir(joinpath(root,"intents")))==1
    end
end

@testset "Commit clocks cannot regress across concurrently prepared forecasts" begin
    mktempdir(realpath(tempdir())) do tmp
        root=joinpath(tmp,"evidence");log=joinpath(tmp,"live.csv")
        one=fixture_log(log);two=copy(one);two.issue_time_utc[1]-=Minute(1)
        table=vcat(one,two);CSV.write(log,table);receipt=capture(root)
        intents=[E.prepare_verification!(root,receipt,
            [(binding=E.forecast_binding(table[i,:]),previous_observation=missing,observation=-11.)];
            log_path=log,verification_started_utc=COMPLETE+Second(1)) for i in 1:2]
        table.observation_dst_nt .= -11.;CSV.write(log,table)
        E.commit_verification!(root,intents[1];log_path=log,verification_completed_utc=COMPLETE+Second(3))
        @test_throws ArgumentError E.commit_verification!(root,intents[2];log_path=log,
            verification_completed_utc=COMPLETE+Second(2))
        state=E.verify_observation_evidence(root)
        @test state.sequence==1 && length(state.pending_intents)==1
        @test_throws ArgumentError E.verify_observation_evidence(root;require_complete=true)
    end
end

@testset "Same-session monotonic order and stale-intent commit races" begin
    mktempdir(realpath(tempdir())) do tmp
        root=joinpath(tmp,"evidence");log=joinpath(tmp,"live.csv")
        table=fixture_log(log)
        function captured(raw,start,finish,ns_start,ns_finish)
            E.capture_response!(root,HTTP.Response(200,raw);request_started_utc=start,
                receipt_completed_utc=finish,monotonic_started_ns=ns_start,monotonic_completed_ns=ns_finish)
        end
        older=captured(RAW,START-Second(10),COMPLETE-Second(10),10,20)
        stale_intent=prepare(root,older,table,log;time=COMPLETE-Second(8))
        latest=captured(RAW_REVISION,START,COMPLETE,300,400)
        current=prepare(root,latest,table,log;value=-12.)
        table.observation_dst_nt[1]=-12.;CSV.write(log,table)
        E.commit_verification!(root,current;log_path=log,verification_completed_utc=COMPLETE+Second(3))
        backwards=captured(RAW,START+Second(10),COMPLETE+Second(10),100,200)
        @test_throws ArgumentError prepare(root,backwards,table,log;value=-11.,previous=-12.,time=COMPLETE+Second(11))
        strict=captured(RAW,START,COMPLETE,500,600)
        valid=prepare(root,strict,table,log;value=-11.,previous=-12.,time=COMPLETE+Second(4))
        table.observation_dst_nt[1]=-11.;CSV.write(log,table)
        E.commit_verification!(root,valid;log_path=log,verification_completed_utc=COMPLETE+Second(5))
        @test_throws ArgumentError E.commit_verification!(root,stale_intent;log_path=log,
            verification_completed_utc=COMPLETE+Second(6))
        state=E.verify_observation_evidence(root)
        @test state.sequence==2 && last(state.events).observation==-11.
        @test length(state.pending_intents)==1
        @test_throws ArgumentError E.verify_observation_evidence(root;require_complete=true)
        @test E.load_receipt(root,strict).capture_session_id==E.load_receipt(root,latest).capture_session_id
    end
end

@testset "Failed writes retain a rollback clock without claiming verification" begin
    mktempdir(realpath(tempdir())) do tmp
        root=joinpath(tmp,"evidence");log=joinpath(tmp,"live.csv")
        table=fixture_log(log);receipt=capture(root);intent=prepare(root,receipt,table,log)
        failed_at=START-Second(1)
        E.record_verification_failure!(root,intent,ErrorException("independently injected CSV write failure");
            log_path=log,failed_at_utc=failed_at)
        state=E.verify_observation_evidence(root;require_complete=true)
        @test isempty(state.events) && isempty(state.pending_intents) && length(state.failed_attempts)==1
        failure=only(state.failed_attempts)
        @test DateTime(failure.failed_at_utc)==failed_at
        @test !failure.failure_clock_eligible && failure.failure_clock_status=="utc_regression"
        @test failure.original_verification_completed_utc===nothing
        @test failure.log_sha256_at_failure==bytes2hex(sha256(read(log)))
        @test_throws ArgumentError E.commit_verification!(root,intent;log_path=log,
            verification_completed_utc=COMPLETE+Second(10))
    end
end
end
end
