@testset "Independent issue, source, feature and target adapter" begin
    mag=Dict(ISSUE+Minute(k)=>(Float64(k*k),0.0) for k in -40:31)
    original=copy(mag)
    wind=Dict(ISSUE-Minute(1)=>(400.0,-5.0,1.0,8.0),
              ISSUE-Minute(4)=>(450.0,3.0,-2.0,6.0),
              ISSUE-Minute(2)=>(450.0,3.0,-2.0,6.0))
    for expected in ORACLE["feature_cases"]
        policy=CMR.ReplayPolicy(;ground_extra_ms=expected["ground_extra_ms"],
            omni_extra_ms=expected["omni_extra_ms"],label_extra_ms=expected["label_extra_ms"])
        prefix=PrefixMag(mag,DateTime(expected["ground_anchor"]))
        ground=prepare_issue(prefix,ISSUE;policy,omni=ForbiddenDict())
        @test ground.accepted && ground.reason===:complete
        @test ground.source_row.ground_anchor==DateTime(expected["ground_anchor"])
        @test ground.mode===:ground11 && length(ground.features)==11
        @test all(isapprox.(ground.features,expected["ground11"];rtol=0,atol=1e-13))
        @test ground.features[8:11]==[0.0,1.0,0.0,1.0]
        @test ground.timing.label_assumed_ready==DateTime(expected["label_ready"])
        @test ground.timing.sufficient_phase_period_minutes==expected["phase_period"]
        mixed=prepare_issue(prefix,ISSUE;policy,mode=:mixed8,omni=wind)
        @test mixed.accepted && length(mixed.features)==8
        @test mixed.source_row.omni_anchor==DateTime(expected["omni_anchor"])
        @test all(isapprox.(mixed.features,expected["mixed8"];rtol=0,atol=1e-13))
        @test mixed.ground11==ground.ground11
        ready=DateTime(expected["label_ready"])
        @test label_audit(ground,ForbiddenDict(),ready-Millisecond(1)).reason===:label_not_ready
        target=label_audit(ground,mag,ready)
        @test target.accepted && target.value.target_max_dbdt==59.0
        @test !split_eligible(ground,ready)
        @test split_eligible(ground,ready+Millisecond(1))
        @test !ground.source_row.measured_receipts_available
    end
    @test mag==original
    axis_ground=Dict(k=>ZeroBasedVector(v) for (k,v) in mag)
    axis_wind=Dict(k=>ZeroBasedVector(v) for (k,v) in wind)
    @test prepare_issue(axis_ground,ISSUE).features==prepare_issue(mag,ISSUE).features
    @test prepare_issue(axis_ground,ISSUE;mode=:mixed8,omni=axis_wind).features==prepare_issue(mag,ISSUE;mode=:mixed8,omni=wind).features
    @test prepare_issue(mag,ISSUE).accepted
    @test prepare_issue(mag,ISSUE;mode=:mixed8,omni=Dict()).reason===:missing_omni_history
    @test_throws ArgumentError prepare_issue(mag,ISSUE;mode=:mixed8)
    @test_throws ArgumentError prepare_issue(mag,ISSUE;mode=:invalid)
    @test_throws ArgumentError prepare_schedule(mag,[ISSUE,ISSUE])
    @test_throws ArgumentError prepare_schedule(mag,[ISSUE+Millisecond(1)])
    incomplete=copy(mag);delete!(incomplete,ISSUE+Minute(30))
    schedule=prepare_schedule(incomplete,[ISSUE])
    @test only(schedule).accepted
    called=Ref(false)
    bound=bind_predictions(schedule,X->(called[]=true;@test size(X)==(1,11);zeros(size(X,1))))
    @test called[] && bound.indices==[1]
    audit=case_audit(schedule,incomplete;asof=ISSUE+Minute(31))
    @test length(audit)==1 && only(audit).feature_accepted
    @test only(audit).label_reason===:missing_ground_target && only(audit).target===nothing
    pending=only(case_audit(schedule,ForbiddenDict();asof=ISSUE+Minute(30)))
    @test pending.feature_accepted && pending.label_reason===:label_not_ready
    rejected=copy(mag);delete!(rejected,ISSUE-Minute(31))
    rows=prepare_schedule(rejected,[ISSUE,ISSUE+Minute(1)])
    @test !rows[1].accepted && rows[2].accepted
    predictions=bind_predictions(rows,X->zeros(size(X,1)))
    @test predictions.indices==[2] && isnan(predictions.centers[1])
    audit=case_audit(rows,ForbiddenDict();asof=ISSUE+Minute(1))
    @test audit[1].label_reason===:not_issued
    @test audit[2].label_reason===:label_not_ready
end

@testset "Float64 controller parameter representability" begin
    mag=Dict(ISSUE+Minute(k)=>(0.0,0.0) for k in -40:0)
    row=prepare_issue(mag,ISSUE)
    tiny=big"1e-10000";near_one=BigFloat(1)-BigFloat(2)^(-100)
    @test_throws ArgumentError CompletedMaximumController(calibration();start=ISSUE,gamma=tiny)
    @test_throws ArgumentError CompletedMaximumController(calibration();start=ISSUE,gamma=0.1,nominal=near_one)
    @test_throws ArgumentError issue!(CompletedMaximumController(calibration();start=ISSUE,gamma=0.1),row,0.0,tiny)
    @test_throws ArgumentError replay_phase([row],[0.0],[1.0],ForbiddenDict(),calibration();gamma=tiny,asof=ISSUE)
    @test_throws ArgumentError replay_fifo([row],[0.0],[tiny],ForbiddenDict(),calibration();gamma=0.1,asof=ISSUE)
    @test_throws ArgumentError static_bounds([row],[0.0],[1.0],calibration();nominal=near_one,asof=ISSUE)
    @test CompletedSourceDBDT.upper_bound(0.0,0.0,1.0,zeros(9))==Inf
    @test CompletedSourceDBDT.upper_bound(1.0,0.0,1.0,zeros(9))==-Inf
    @test CompletedSourceDBDT.upper_bound(0.1,0.0,1.0,[0.0])==Inf
    @test CompletedSourceDBDT.upper_bound(0.1,-10.0,1.0,zeros(9))<0.0
end

@testset "Distinct calibration identity and strict readiness" begin
    @test_throws ArgumentError CompletedCalibration(zeros(9),fill(ISSUE-Millisecond(1),9);
        model_schema="dbdt_ground_only_v1",point_model_sha256=repeat("1",64),purpose=:synthetic)
    @test_throws ArgumentError CompletedCalibration(zeros(9),fill(ISSUE-Millisecond(1),9);
        model_schema="dbdt_completed_source_point_v1",calibration_schema="legacy",
        point_model_sha256=repeat("1",64),purpose=:synthetic)
    @test_throws ArgumentError CompletedCalibration([NaN],[ISSUE-Millisecond(1)];
        model_schema="dbdt_completed_source_point_v1",point_model_sha256=repeat("1",64),purpose=:synthetic)
    @test_throws ArgumentError CompletedCalibration([0.0],[ISSUE-Millisecond(1)];
        model_schema="dbdt_completed_source_point_v1",point_model_sha256="unverified",purpose=:synthetic)
    @test_throws ArgumentError CompletedCalibration([0.0,1.0],[ISSUE,ISSUE-Millisecond(1)];
        model_schema="dbdt_completed_source_point_v1",point_model_sha256=repeat("1",64),purpose=:synthetic)
    mag=Dict(ISSUE+Minute(k)=>(0.0,0.0) for k in -40:1)
    rows=prepare_schedule(mag,[ISSUE])
    immature=calibration(;ready=ISSUE)
    @test_throws ArgumentError replay_phase(rows,[0.0],[1.0],ForbiddenDict(),immature;gamma=0.1,asof=ISSUE)
    @test_throws ArgumentError CompletedMaximumController(immature;start=ISSUE,gamma=0.1)
    @test_throws ArgumentError replay_fifo(rows,[0.0],[1.0],ForbiddenDict(),calibration(;mode=:mixed8);gamma=0.1,asof=ISSUE)
    @test_throws ArgumentError replay_fifo(rows,[0.0],[1.0],ForbiddenDict(),calibration();gamma=0.1,asof=ISSUE-Millisecond(1))
end
