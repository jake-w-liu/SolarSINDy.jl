module LiveObservationEvidenceIntegrationTests
using Test
const TEST_MONITOR_DIR=mktempdir()
withenv("SOLARSINDY_MONITOR_DIR"=>TEST_MONITOR_DIR) do
    include(joinpath(@__DIR__,"..","examples","live_monitor.jl"))
end
const E=LiveObservationEvidence
const TAG=DateTime(2026,1,1,9)
const RAW="[{\"time_tag\":\"2026-01-01T09:00:00\",\"dst\":-11}]"
const REVISION="[{\"time_tag\":\"2026-01-01T09:00:00\",\"dst\":-12}]"
const FAIL_LOG_REPLACE=Ref(false)
function _live_atomic_replace(source::String,target::String)
    if FAIL_LOG_REPLACE[]
        FAIL_LOG_REPLACE[]=false
        error("independently injected live-log write failure")
    end
    invoke(_live_atomic_replace,Tuple{AbstractString,AbstractString},source,target)
end
function fixture_log(path)
    table=DataFrame(issue_time_utc=[DateTime(2026,1,1,8,45)],
        latest_dst_time_utc=[DateTime(2026,1,1,8)],target_time_utc=[TAG],
        model_step_hours=[1],model_version=["sindy_core"],sub_hourly_model_version=["sindy_core"],
        superlearner_manifest_sha256=[repeat("a",64)],
        pred_dst_nt=[-10.],pred_dst_ci05_nt=[-20.],pred_dst_ci95_nt=[0.],
        served_pred_dst_nt=[-10.],served_pred_dst_ci05_nt=[-20.],served_pred_dst_ci95_nt=[0.],
        observation_dst_nt=Union{Missing,Float64}[missing])
    CSV.write(path,table)
    return table
end
quiet(f)=redirect_stdout(f,devnull)

@testset "Fetch retries retain failed bytes and successful receipt" begin
    mktempdir() do tmp
        root=E.evidence_root(joinpath(tmp,"live.csv"));receipt=Ref{Any}(nothing);calls=Ref(0)
        responses=[HTTP.Response(503,UInt8[0xff,0]),HTTP.Response(200,"not JSON"),HTTP.Response(200,RAW)]
        get_response(url;kwargs...)=begin
            @test url==KYOTO_DST_JSON_URL
            @test kwargs[:status_exception]===false
            @test kwargs[:socket_type_tls]===MbedTLS.SSLContext
            calls[]+=1;responses[calls[]]
        end
        times,values=_fetch_dst(;fetch_fn=get_response,max_retries=3,retry_delay_sec=0,
            evidence_root=root,receipt_ref=receipt)
        @test calls[]==3 && times==[TAG] && values==[-11.]
        @test length(readdir(joinpath(root,"fetches")))==3
        files=readdir(joinpath(root,"raw");join=true)
        @test length(files)==3
        @test any(path->read(path)==UInt8[0xff,0],files)
        @test any(path->read(path)==Vector{UInt8}(codeunits("not JSON")),files)
        @test E.load_receipt(root,receipt[]).http_status==200
        @test E.load_receipt(root,receipt[]).consumer_source_sha256==LIVE_VERIFY_SOURCE_SHA256
    end
end

@testset "Clock rollback responses remain raw evidence and never become inputs" begin
    mktempdir() do tmp
        for status in (200,503)
            root=E.evidence_root(joinpath(tmp,"clock-$(status).csv"));receipt=Ref{Any}(nothing)
            clocks=[TAG+Minute(30),TAG+Minute(29)];nanos=[100,200]
            @test_throws ArgumentError _fetch_dst(;
                fetch_fn=(url;kwargs...)->HTTP.Response(status,RAW),max_retries=1,
                evidence_root=root,receipt_ref=receipt,
                utc_clock=()->popfirst!(clocks),monotonic_clock=()->popfirst!(nanos))
            record=E.load_receipt(root,receipt[])
            @test record.http_status==status && !record.clock_eligible
            @test read(joinpath(root,String(record.raw_relative_path)))==Vector{UInt8}(codeunits(RAW))
            @test length(readdir(joinpath(root,"fetches")))==1
        end
    end
end

@testset "An older fetched snapshot is rejected before the live CSV changes" begin
    mktempdir() do tmp
        log=joinpath(tmp,"live.csv");fixture_log(log);cfg=LiveVerifyConfig(;model=:sindy_core,log_path=log)
        root=E.evidence_root(log)
        function captured(raw,minute,ns)
            E.capture_response!(root,HTTP.Response(200,raw);
                request_started_utc=TAG+Minute(minute)-Second(1),receipt_completed_utc=TAG+Minute(minute),
                monotonic_started_ns=ns,monotonic_completed_ns=ns+1)
        end
        older=captured(RAW,10,100);newer=captured(REVISION,20,200)
        @test quiet(()->refresh_observations!(cfg;dst_times=[TAG],dst_vals=[-12.],dst_receipt=newer))==1
        before=read(log)
        @test_throws ArgumentError quiet(()->refresh_observations!(cfg;
            dst_times=[TAG],dst_vals=[-11.],dst_receipt=older))
        @test read(log)==before
        state=E.verify_observation_evidence(root;require_complete=true)
        @test state.sequence==1 && only(state.events).observation==-12.
        @test length(readdir(joinpath(root,"intents")))==1
    end
end

@testset "Manager reuses one real captured receipt for verification and revisions" begin
    mktempdir() do tmp
        log=joinpath(tmp,"live.csv");before=fixture_log(log)
        cfg=LiveVerifyConfig(;model=:sindy_core,log_path=log)
        calls=Ref(0)
        get_response(url;kwargs...)=(calls[]+=1;HTTP.Response(200,RAW))
        inputs=prepare_issue_inputs(cfg;plasma_fn=()->nothing,mag_fn=()->nothing,http_get=get_response)
        @test calls[]==1 && inputs.dst_receipt!==nothing
        seen=Ref{Any}(nothing)
        refresh(cfg;dst_times=nothing,dst_vals=nothing,dst_receipt=nothing)=begin
            seen[]=dst_receipt
            refresh_observations!(cfg;dst_times,dst_vals,dst_receipt)
        end
        no_op=(args...;kwargs...)->nothing
        outcome=quiet() do
            cycle!(;prepare_fn=cfg->inputs,policy_fn=inputs->:static,
                issue_cycle_fn=(args...;kwargs...)->(succeeded=length(HORIZONS),complete=true),
                dst_fn=()->error("manager must reuse captured Dst"),refresh_fn=refresh,
                verify_fn=(args...;kwargs...)->error("unexpected verification fallback"),
                retention_fn=no_op,snapshot_fn=no_op,claim_audit_fn=no_op,
                report_fn=no_op,daily_review_fn=no_op,log_path=log,report_path=log*".md")
        end
        @test outcome.complete && seen[]==inputs.dst_receipt && calls[]==1
        scored=CSV.read(log,DataFrame)
        @test scored.observation_dst_nt==[-11.]
        @test scored.served_pred_dst_nt==before.served_pred_dst_nt
        root=E.evidence_root(log);state=E.verify_observation_evidence(root;require_complete=true)
        @test state.sequence==1 && length(state.events)==1
        @test only(state.events).receipt.sha256==inputs.dst_receipt.sha256
        @test only(state.events).kind=="first_verification"
        @test quiet(()->refresh_observations!(cfg;dst_times=[TAG],dst_vals=[-11.],dst_receipt=inputs.dst_receipt))==0
        @test E.verify_observation_evidence(root).sequence==1
        @test quiet(()->refresh_observations!(cfg;dst_fetch_fn=(args...;kwargs...)->HTTP.Response(200,REVISION)))==1
        state=E.verify_observation_evidence(root;require_complete=true)
        @test state.sequence==2 && last(state.events).observation==-12.
        @test last(state.events).kind=="revised_observation"
        @test first(state.events).observation==-11.
        @test !ispath(joinpath(root,"journal.lock")) && !ispath(log*".lock")
    end
end

@testset "Direct verification, wait path and mismatch fail closed" begin
    for mode in (:verify,:wait,:mismatch,:write_failure)
        mktempdir() do tmp
            log=joinpath(tmp,"live.csv");table=fixture_log(log);old=read(log)
            cfg=LiveVerifyConfig(;model=:sindy_core,log_path=log)
            response=(args...;kwargs...)->HTTP.Response(200,RAW)
            if mode==:verify
                @test quiet(()->verify_pending!(cfg;dst_fetch_fn=response))==1
                state=E.verify_observation_evidence(E.evidence_root(log);require_complete=true)
                @test only(state.events).observation==-11.
            elseif mode==:wait
                forecast=(;issue_time=DateTime(2026,1,1,8,45),target_time=TAG,
                    model_version="sindy_core",pred_dst=-10.)
                result=quiet(()->wait_for_observation(cfg,forecast;fetch_fn=response))
                @test result.observed_dst==-11.
                @test E.verify_observation_evidence(E.evidence_root(log);require_complete=true).sequence==1
            elseif mode==:mismatch
                receipt=Ref{Any}(nothing)
                _fetch_dst(;fetch_fn=response,evidence_root=E.evidence_root(log),receipt_ref=receipt)
                @test_throws ArgumentError quiet(()->verify_pending!(cfg;
                    dst_times=[TAG],dst_vals=[-99.],dst_receipt=receipt[]))
                @test read(log)==old
                @test E.verify_observation_evidence(E.evidence_root(log)).sequence==0
            else
                FAIL_LOG_REPLACE[]=true
                @test_throws ErrorException quiet(()->verify_pending!(cfg;dst_fetch_fn=response))
                @test read(log)==old
                state=E.verify_observation_evidence(E.evidence_root(log))
                @test state.sequence==0 && isempty(state.events) && isempty(state.pending_intents)
                @test length(state.failed_attempts)==1
                @test only(state.failed_attempts).original_verification_completed_utc===nothing
                @test only(state.failed_attempts).disposition=="failed_attempt_no_verification_claim"
                @test E.verify_observation_evidence(E.evidence_root(log);require_complete=true).sequence==0
                # A new actual attempt can succeed; the failed attempt never becomes a score.
                @test quiet(()->verify_pending!(cfg;dst_fetch_fn=response))==1
                final=E.verify_observation_evidence(E.evidence_root(log);require_complete=true)
                @test length(final.events)==1 && length(final.failed_attempts)==1
            end
            @test !ispath(log*".lock")
        end
    end
end

end
