module TestVersionedShadowTrials
using Test,Dates,CSV,DataFrames,JSON3,SHA,TOML
module Producer
include(joinpath(@__DIR__,"..","examples","live_forecast_verify.jl"))
end
module Auditor
include(joinpath(@__DIR__,"..","validation","operational","superlearner_live_claim_audit.jl"))
end
module Dashboard
include(joinpath(@__DIR__,"..","app","src","server.jl"))
end
module Monitor
include(joinpath(@__DIR__,"..","examples","live_monitor.jl"))
end
const A=Auditor.SuperlearnerLiveClaimAudit
const T=Producer.CalibrationShadowTrials
const fixture=JSON3.read(read(joinpath(@__DIR__,"versioned_shadow_trial_oracle.json")),Dict{String,Any})
struct NoAcquisitionPrepared end
Base.getproperty(::NoAcquisitionPrepared,::Symbol)=error("pre-start rejection must precede prepared input access")

function source_rows(trial,n;start=trial.start_utc,step=1)
    issues=start .+ Hour.(0:n-1)
    targets=issues .+ Hour(step)
    DataFrame(issue_time_utc=string.(issues),latest_dst_time_utc=string.(issues),
        target_time_utc=string.(targets),horizon_hours=fill(Float64(step),n),
        model_step_hours=fill(Float64(step),n),model_version=fill("inertial_sindy",n),
        latest_dst_nt=fill(-10.0,n),pred_dst_nt=fill(-10.0,n),
        pred_dst_ci05_nt=fill(-12.0,n),pred_dst_ci95_nt=fill(-7.0,n),
        observation_dst_nt=-10.0 .+ collect(1.0:n),served_pred_dst_nt=fill(-10.0,n),
        served_pred_dst_ci05_nt=fill(-12.0,n),served_pred_dst_ci95_nt=fill(-7.0,n),
        sub_hourly_model_version=fill(Producer.SUPERLEARNER_SERVED_TAIL_VERSION,n),
        sub_hourly_pred_dst_nt=fill(-10.0,n),sub_hourly_pred_dst_ci05_nt=fill(-12.0,n),
        sub_hourly_pred_dst_ci95_nt=fill(-7.0,n),superlearner_status=fill("ok",n),
        superlearner_manifest_sha256=fill(trial.served_manifest_sha256,n),
        superlearner_pred_dst_nt=fill(-10.0,n),superlearner_ci05_nt=fill(-12.0,n),superlearner_ci95_nt=fill(-7.0,n),
        superlearner_cal_shadow_model_version=fill(trial.identity,n),
        superlearner_cal_shadow_config_sha256=fill(trial.config_sha256,n),
        superlearner_cal_shadow_status=fill("warmup:0/30",n),
        superlearner_cal_shadow_ci05_nt=Union{Missing,Float64}[missing for _ in 1:n],
        superlearner_cal_shadow_ci95_nt=Union{Missing,Float64}[missing for _ in 1:n],
        superlearner_cal_shadow_history_n=zeros(Int,n),
        superlearner_cal_shadow_location_shift_nt=Union{Missing,Float64}[missing for _ in 1:n])
end

@testset "Pinned versioned shadow spec preserves legacy defaults" begin
    @test Producer.SUPERLEARNER_CALIBRATION_SHADOW_VERSION==fixture["legacy_identity"]
    @test A.SHADOW_IDENTITY==fixture["legacy_identity"]
    @test Dashboard.SUPERLEARNER_CALIBRATION_SHADOW_MODEL_VERSION==fixture["legacy_identity"]
    @test T.LEGACY_A3.config_sha256==fixture["legacy_config_sha256"]
    @test T.LEGACY_A3.served_manifest_sha256==fixture["legacy_manifest_sha256"]
    @test T.configured_trial(;environment=Dict{String,String}())===T.LEGACY_A3
    @test_throws Exception T.configured_trial(;environment=Dict("SOLARSINDY_SHADOW_TRIAL_CONFIG"=>"missing"))
    mktempdir() do dir
        config=joinpath(dir,"synthetic.toml");write(config,fixture["config_toml"])
        trial=T.load_trial(config,fixture["config_sha256"])
        @test trial.trial_id=="synthetic-2027"
        @test trial.start_utc==DateTime(2027,1,1)
        @test !trial.legacy
        @test_throws Exception T.configured_trial(;environment=Dict("SOLARSINDY_SHADOW_TRIAL_CONFIG"=>config,
            "SOLARSINDY_SHADOW_TRIAL_SHA256"=>fixture["config_sha256"]))
        @test T.configured_trial(;environment=Dict("SOLARSINDY_SHADOW_TRIAL_CONFIG"=>config,
            "SOLARSINDY_SHADOW_TRIAL_SHA256"=>fixture["config_sha256"],
            "SOLARSINDY_SHADOW_TRIAL_SEAL_SHA256"=>"1"^64)).seal_sha256=="1"^64
        @test_throws Exception T.load_trial(config,"0"^64)
        for (old,new) in (("schema = 1","schema = true"),
            ("median24-width1h1.50-widthOther1.20-warm30","median1-width1h1.50-widthOther1.20-warm1"),
            ("cold-exact-trial-issued-since-start","reuse-old-history"),
            ("2026-12-31T00:00:00Z","2027-01-02T00:00:00Z"),
            ("trial_id = \"synthetic-2027\"","trial_id = \"../escape\""))
            text=replace(fixture["config_toml"],old=>new)
            write(config,text)
            @test_throws Exception T.load_trial(config,bytes2hex(sha256(text)))
        end
        text=fixture["config_toml"]*"unknown_key = 1\n";write(config,text)
        @test_throws Exception T.load_trial(config,bytes2hex(sha256(text)))
        text=fixture["config_toml"]*"schema = 1\n";write(config,text)
        @test_throws Exception T.load_trial(config,bytes2hex(sha256(text)))
    end
end

@testset "Cold history is independent of old or foreign residuals" begin
    mktempdir() do dir
        config=joinpath(dir,"synthetic.toml");write(config,fixture["config_toml"])
        trial=T.load_trial(config,fixture["config_sha256"])
        log=joinpath(dir,"live_forecast_log.csv")
        old=source_rows(T.LEGACY_A3,40;start=trial.start_utc-Hour(100))
        old.superlearner_manifest_sha256 .= trial.served_manifest_sha256
        old.observation_dst_nt .= 10_000.0
        prestart=source_rows(trial,40;start=trial.start_utc-Hour(80))
        prestart.observation_dst_nt .= -10_000.0
        wrong_digest=source_rows(trial,40);wrong_digest.superlearner_cal_shadow_config_sha256 .= "0"^64
        wrong_manifest=source_rows(trial,40);wrong_manifest.superlearner_manifest_sha256 .= "0"^64
        foreign=source_rows(T.LEGACY_A3,40;start=trial.start_utc)
        foreign.superlearner_manifest_sha256 .= trial.served_manifest_sha256
        foreign.superlearner_cal_shadow_config_sha256 .= trial.config_sha256
        issue=trial.start_utc+Hour(60);cutoff=issue-Hour(1)
        for row in fixture["cases"]
            own=source_rows(trial,row["residual_count"])
            future=source_rows(trial,1;start=issue-Minute(30),step=1)
            future.observation_dst_nt .= 20_000.0
            contaminated=vcat(old,prestart,wrong_digest,wrong_manifest,foreign,future,own;cols=:union)
            CSV.write(log,contaminated)
            result=Producer._superlearner_calibration_shadow_from_log(log,-10.0,-12.0,-7.0,1,cutoff,issue;shadow_trial=trial)
            @test result.status==row["status"]
            @test result.history_n==row["residual_count"]
            if row["status"]=="ok"
                @test result.location==row["location"]
                @test result.lo==row["lo"]
                @test result.hi==row["hi"]
            else
                @test ismissing(result.location)&&ismissing(result.lo)&&ismissing(result.hi)
            end
        end
        own=source_rows(trial,35;step=2);CSV.write(log,own)
        result=Producer._superlearner_calibration_shadow_from_log(log,-10.0,-12.0,-7.0,2,cutoff,issue;shadow_trial=trial)
        @test result.location==fixture["step2"]["location"]
        @test isapprox(result.lo,fixture["step2"]["lo"];rtol=0,atol=1e-12)
        @test isapprox(result.hi,fixture["step2"]["hi"];rtol=0,atol=1e-12)
        revised=copy(own);revised.observation_dst_nt .= 10_000.0;CSV.write(log,revised)
        @test Producer._superlearner_calibration_shadow_from_log(log,-10.0,-12.0,-7.0,2,cutoff,issue;
            shadow_trial=trial,dst_times=DateTime.(own.target_time_utc),dst_vals=own.observation_dst_nt)==result
        before=Producer._superlearner_calibration_shadow_from_log(log,-10.0,-12.0,-7.0,2,cutoff,trial.start_utc-Millisecond(1);shadow_trial=trial)
        @test before.status=="unavailable:trial_not_started"
        duplicated=vcat(own,own[end:end,:]);CSV.write(log,duplicated)
        @test Producer._superlearner_calibration_shadow_from_log(log,-10.0,-12.0,-7.0,2,cutoff,issue;shadow_trial=trial)==result
        duplicated.observation_dst_nt[end]+=1;CSV.write(log,duplicated)
        @test Producer._superlearner_calibration_shadow_from_log(log,-10.0,-12.0,-7.0,2,cutoff,issue;shadow_trial=trial).status=="unavailable:history_error"
    end
end

@testset "Exclusive sealed legacy assessment and separate trial audit/API" begin
    mktempdir() do dir
        config=joinpath(dir,"synthetic.toml");write(config,fixture["config_toml"])
        trial=T.load_trial(config,fixture["config_sha256"])
        log=joinpath(dir,"live_forecast_log.csv")
        old=source_rows(T.LEGACY_A3,1;start=trial.start_utc-Hour(100))
        old.superlearner_manifest_sha256 .= trial.served_manifest_sha256
        CSV.write(log,old)
        original=read(log)
        @test length(A.prospective_status(old;bootstrap_reps=100).integrity.violations)==fixture["old_snapshot_findings_preserved"]
        @test_throws Exception T.verify_legacy_seal(log,trial)
        @test_throws Exception A.seal_legacy_trial!(log,trial;reference=trial.start_utc+Millisecond(1),bootstrap_reps=100)
        seal=A.seal_legacy_trial!(log,trial;reference=trial.start_utc-Hour(12),bootstrap_reps=100)
        @test_throws Exception T.verify_legacy_seal(log,trial)
        trial=T.bind_legacy_seal(trial,seal.receipt_sha256)
        @test trial.config_sha256==fixture["config_sha256"]
        @test read(log)==original
        sealed=[seal.source,seal.assessment,seal.report,seal.receipt]
        hashes=T.digest.(sealed)
        @test_throws Exception A.seal_legacy_trial!(log,trial;reference=trial.start_utc-Hour(11),bootstrap_reps=100)
        legacy_output=joinpath(dir,"superlearner_live_claim_status.json")
        write(legacy_output,"unchanged legacy output")
        own=source_rows(trial,35)
        CSV.write(log,vcat(old,own;cols=:union))
        status=A.run_trial_claim_audit(log,trial;bootstrap_reps=100)
        @test status.integrity.gate_pass
        @test status.integrity.canonical_rows==35
        @test status.marginal.n_rows==0
        @test !status.marginal.claim_ready && !status.storm.claim_ready
        @test read(legacy_output,String)=="unchanged legacy output"
        @test T.digest.(sealed)==hashes
        prepared_early=Ref(false)
        @test_throws ArgumentError Monitor.cycle!(;shadow_trial=trial,log_path=log,
            trial_reference=trial.start_utc-Millisecond(1),
            prepare_fn=cfg->(prepared_early[]=true;error("must not acquire")))
        @test !prepared_early[]
        @test_throws ArgumentError Producer.issue_forecast(Producer.LiveVerifyConfig(;log_path=log);
            shadow_trial=trial,trial_reference=trial.start_utc-Millisecond(1),inputs=NoAcquisitionPrepared())
        legacy_calls=Int[]
        strict_issue=function(cfg;inputs,write_trajectory,verbose,interval_policy)
            push!(legacy_calls,cfg.horizon_hours);nothing
        end
        legacy_result=redirect_stdout(devnull) do
            Monitor._issue_horizon_cycle!((issue_time=DateTime(2026,1,1),);
                issue_fn=strict_issue,log_path=log,calibration_path=Monitor.OPERATIONAL_CALIB,
                complete_fn=(args...)->true,interval_policy=:static)
        end
        @test legacy_result.succeeded==4 && legacy_calls==[1,2,3,6]
        called=String[]
        record=name->(args...;kwargs...)->(push!(called,name);nothing)
        issuance=redirect_stdout(devnull) do
            Monitor.cycle!(;shadow_trial=trial,trial_reference=trial.start_utc+Hour(60),log_path=log,report_path=log*".md",
                prepare_fn=cfg->(dst=([trial.start_utc],[0.0]),),policy_fn=inputs->:static,
                issue_cycle_fn=(inputs;shadow_trial,kwargs...)->begin
                    @test shadow_trial.identity==trial.identity
                    push!(called,"issue");(succeeded=4,complete=true)
                end,
                refresh_fn=record("refresh"),verify_fn=record("verify"),retention_fn=record("retention"),
                claim_audit_fn=(args...;kwargs...)->error("legacy output must remain sealed"),
                trial_claim_audit_fn=(path,spec)->begin
                    @test spec.config_sha256==trial.config_sha256
                    push!(called,"new_audit");nothing
                end,
                snapshot_fn=record("snapshot"),report_fn=record("report"),daily_review_fn=record("review"))
        end
        @test issuance.complete
        @test called==["issue","refresh","retention","new_audit","snapshot","report","review"]
        paths=T.trial_paths(log,trial)
        @test paths.status!=legacy_output
        # Historical display authenticates its frozen snapshot, not today's
        # mutable pre-start scoring rows. Legitimate revisions do not reseal it.
        revised_old=copy(old);revised_old.observation_dst_nt .= -999.0
        CSV.write(log,vcat(revised_old,own;cols=:union))
        @test A.run_trial_claim_audit(log,trial;bootstrap_reps=100).integrity.gate_pass
        CSV.write(log,own)
        @test A.run_trial_claim_audit(log,trial;bootstrap_reps=100).integrity.gate_pass
        @test T.digest.(sealed)==hashes
        CSV.write(log,vcat(old,own;cols=:union))
        A.run_trial_claim_audit(log,trial;bootstrap_reps=100)
        bad=source_rows(trial,1;start=trial.start_utc-Millisecond(1))
        @test !A.prospective_status(bad;trial,bootstrap_reps=100).integrity.gate_pass
        bad=source_rows(T.LEGACY_A3,1;start=trial.start_utc)
        @test !A.prospective_status(bad;trial,bootstrap_reps=100).integrity.gate_pass
        bad=source_rows(trial,1);bad.issue_time_utc[1]="not-a-time"
        @test !A.prospective_status(bad;trial,bootstrap_reps=100).integrity.gate_pass
        # Freshness is literal test evidence, not an activation of the synthetic protocol.
        report=JSON3.read(read(paths.status),Dict{String,Any})
        reference=DateTime(2027,1,3)
        report["generated_utc"]=string(reference)*"Z"
        write(paths.status,JSON3.write(report))
        payload=Dashboard.versioned_shadow_status_payload(log;trial,reference)
        @test payload.calibration_shadow_trial.available
        @test length(payload.calibration_shadow_trial.legacy.assessment.integrity.violations)==1
        @test payload.calibration_shadow_trial.legacy.historical
        @test !Dashboard.versioned_shadow_status_payload(log;trial,reference=reference+Minute(181)).calibration_shadow_trial.available
        for field in ("shadow_identity","shadow_config_sha256","served_manifest_sha256")
            saved=report[field];report[field]="incorrect";write(paths.status,JSON3.write(report))
            @test !Dashboard.versioned_shadow_status_payload(log;trial,reference).calibration_shadow_trial.available
            report[field]=saved
        end
        report["marginal"]["claim_ready"]=true;write(paths.status,JSON3.write(report))
        @test !Dashboard.versioned_shadow_status_payload(log;trial,reference).calibration_shadow_trial.available
        report["marginal"]["claim_ready"]=false
        backups=read.(sealed)
        receipt=TOML.parsefile(seal.receipt)
        clean_old=copy(old);clean_old.superlearner_manifest_sha256 .= T.LEGACY_A3.served_manifest_sha256
        clean=A.prospective_status(clean_old;bootstrap_reps=100)
        @test isempty(clean.integrity.violations)
        CSV.write(seal.source,clean_old);write(seal.assessment,JSON3.write(clean));write(seal.report,A._report(clean))
        for (key,path) in (("source",seal.source),("assessment",seal.assessment),("report",seal.report))
            receipt[key*"_sha256"]=T.digest(path)
        end
        receipt["legacy_integrity_findings"]=0
        open(seal.receipt,"w") do io;TOML.print(io,receipt;sorted=true);end
        report["legacy_seal_sha256"]=T.digest(seal.receipt);write(paths.status,JSON3.write(report))
        @test trial.config_sha256==fixture["config_sha256"]
        @test_throws Exception T.verify_legacy_seal(log,trial)
        @test !Dashboard.versioned_shadow_status_payload(log;trial,reference).calibration_shadow_trial.available
        for (path,bytes) in zip(sealed,backups);write(path,bytes);end
        @test T.digest.(sealed)==hashes
        report["legacy_seal_sha256"]="0"^64;write(paths.status,JSON3.write(report))
        @test !Dashboard.versioned_shadow_status_payload(log;trial,reference).calibration_shadow_trial.available
        source_bytes=read(seal.source);write(seal.source,"changed source")
        @test_throws Exception T.verify_legacy_seal(log,trial)
        write(seal.source,source_bytes)
        @test T.digest.(sealed)==hashes
        prepared=Ref(false)
        write(seal.assessment,"corrupt")
        @test_throws Exception T.verify_legacy_seal(log,trial)
        @test !Dashboard.versioned_shadow_status_payload(log;trial,reference).calibration_shadow_trial.available
        @test_throws Exception Monitor.cycle!(;shadow_trial=trial,log_path=log,
            trial_reference=trial.start_utc+Hour(60),
            prepare_fn=cfg->(prepared[]=true;error("must not acquire any source")))
        @test !prepared[]
    end
end

@testset "Seal completion clock follows generated assessment" begin
    mktempdir() do dir
        text=replace(fixture["config_toml"],"2026-12-31T00:00:00Z"=>"2000-01-01T00:00:00Z",
            "2027-01-01T00:00:00Z"=>"2099-01-01T00:00:00Z")
        config=joinpath(dir,"synthetic-clock.toml");write(config,text)
        trial=T.load_trial(config,bytes2hex(sha256(text)))
        log=joinpath(dir,"live_forecast_log.csv")
        old=source_rows(T.LEGACY_A3,1;start=DateTime(2020))
        old.superlearner_manifest_sha256 .= trial.served_manifest_sha256
        CSV.write(log,old)
        seal=A.seal_legacy_trial!(log,trial;reference=DateTime(2000,1,2),bootstrap_reps=100)
        assessment=JSON3.read(read(seal.assessment))
        @test T.utc_time(assessment.generated_utc)<=T.utc_time(seal.sealed_utc)
        bound=T.bind_legacy_seal(trial,seal.receipt_sha256)
        @test T.verify_legacy_seal(log,bound).receipt_sha256==seal.receipt_sha256
    end
end
end
