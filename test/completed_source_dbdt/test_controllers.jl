@testset "Exact independent31-phase and FIFO traces" begin
    mag=Dict(ISSUE+Minute(k)=>(0.0,0.0) for k in -40:80)
    rows=prepare_schedule(mag,ISSUE:Minute(1):ISSUE+Minute(43))
    labels=Dict(ISSUE+Minute(k)=>(k==0 ? 1.0 : 0.0) for k in 0:43)
    centers=zeros(44);scales=ones(44);cal=calibration()
    phase=replay_phase(rows,centers,scales,labels,cal;gamma=0.1,asof=ISSUE+Minute(43),flush=false)
    fifo=replay_fifo(rows,centers,scales,labels,cal;gamma=0.1/31,asof=ISSUE+Minute(43),flush=false)
    for (actual,name) in ((phase,"phase_trace"),(fifo,"fifo_trace"))
        for (index,expected) in enumerate(ORACLE[name])
            @test isapprox(actual.alpha_issue[index],expected["alpha"];rtol=0,atol=1e-14)
            @test actual.upper[index]==(expected["upper"]=="Inf" ? Inf : 0.0)
        end
        @test actual.certificate.issued==44 && actual.certificate.resolved==13
        @test actual.certificate.misses==1 && actual.certificate.unknown==31
        @test actual.phase_period_minutes==31
        @test !actual.pathwise_bound_applicable
        @test abs(actual.accounting_residual)<1e-12
    end
    phase_flushed=replay_phase(rows,centers,scales,labels,cal;gamma=0.1,asof=ISSUE+Minute(74))
    fifo_flushed=replay_fifo(rows,centers,scales,labels,cal;gamma=0.1/31,asof=ISSUE+Minute(74))
    @test phase_flushed.upper==phase.upper && phase_flushed.alpha_issue==phase.alpha_issue
    @test fifo_flushed.upper==fifo.upper && fifo_flushed.alpha_issue==fifo.alpha_issue
    # The hand trace labels phases relative to its first issue; the implementation
    # labels the same phases in absolute UTC order. Their state multiset agrees.
    @test all(isapprox.(sort(phase_flushed.final_alpha),sort(ORACLE["phase_final_after_flush"]);rtol=0,atol=1e-14))
    parts=split(ORACLE["fifo_final_after_flush_fraction"],"/")
    expected_final=Float64(parse(BigInt,parts[1])//parse(BigInt,parts[2]))
    @test isapprox(only(fifo_flushed.final_alpha),expected_final;rtol=0,atol=1e-14)
    @test phase_flushed.certificate.resolved==fifo_flushed.certificate.resolved==44
    @test phase_flushed.certificate.unknown==fifo_flushed.certificate.unknown==0
    @test phase_flushed.pathwise_bound_applicable && !fifo_flushed.pathwise_bound_applicable
    @test abs(1/44-0.1)<=phase_flushed.pathwise_bound
    @test abs(phase_flushed.accounting_residual)<1e-12 && abs(fifo_flushed.accounting_residual)<1e-12
    prefix=rows[1:31]
    early=replay_phase(prefix,centers[1:31],scales[1:31],ForbiddenDict(),cal;
        gamma=0.1,asof=ISSUE+Minute(30)+Second(44)+Millisecond(999))
    @test all(==(0.0),early.upper) && early.certificate.resolved==0
    exact=replay_phase(prefix,centers[1:31],scales[1:31],Dict(ISSUE=>1.0),cal;
        gamma=0.1,asof=ISSUE+Minute(30)+Second(45))
    @test exact.certificate.resolved==1 && exact.certificate.misses==1
    missing_labels=replay_fifo(rows,centers,scales,Dict{DateTime,Float64}(),cal;
        gamma=0.1/31,asof=ISSUE+Minute(74))
    @test missing_labels.certificate.issued==44 && missing_labels.certificate.unknown==44
    @test missing_labels.certificate.resolved==0 && all(==(:missing),missing_labels.status)
    @test !missing_labels.pathwise_bound_applicable
    static=static_bounds(rows,centers,scales,cal;asof=ISSUE+Minute(43))
    @test static.indices==phase.indices==fifo.indices && all(==(0.0),static.upper)
end

@testset "Irregular gaps, chronological FIFO batches and invalid labels" begin
    offsets=[row["offset"] for row in ORACLE["gap_fifo_trace"]]
    mag=Dict(ISSUE+Minute(k)=>(0.0,0.0) for k in -40:80)
    rows=prepare_schedule(mag,ISSUE .+ Minute.(offsets))
    labels=Dict(ISSUE+Minute(offset)=>(offset==0 ? 1.0 : 0.0) for offset in reverse(offsets))
    run=replay_fifo(rows,zeros(5),ones(5),labels,calibration();gamma=0.1/31,asof=ISSUE+Minute(41),flush=false)
    for (i,expected) in enumerate(ORACLE["gap_fifo_trace"])
        @test isapprox(run.alpha_issue[i],expected["alpha"];rtol=0,atol=1e-14)
        @test run.upper[i]==(expected["upper"]=="Inf" ? Inf : 0.0)
    end
    @test run.max_feedback_batch==3
    @test [event.issue for event in run.events]==ISSUE .+ Minute.([0,1,2])
    @test all(event.time==ISSUE+Minute(40) for event in run.events)
    @test !run.pathwise_bound_applicable
    bad=replay_fifo(rows,zeros(5),ones(5),Dict(ISSUE=>NaN),calibration();gamma=0.1/31,asof=ISSUE+Minute(41),flush=false)
    @test bad.status[1]===:invalid && bad.certificate.resolved==0
    @test bad.certificate.issued==5 && bad.certificate.unknown==5
end
