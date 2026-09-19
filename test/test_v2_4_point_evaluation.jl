module V24PointEvaluationTests
using Test, DataFrames, Dates, Statistics
include(joinpath(@__DIR__,"..","validation","operational","v2_4_point_evaluation.jl"))
const E=V24PointEvaluation
const P=E.P

function fixture()
    t=DateTime(2026,1,5)
    times=[t,t+Day(1),t+Day(1)]
    df=DataFrame(issue_time_utc=times,latest_dst_time_utc=times.-Hour(1),
        target_time_utc=times.+Hour(1),model_step_hours=fill(2,3),
        latest_dst_nt=[-31.,0.,-31.],observation_dst_nt=[3.,-1.,NaN],
        static_lo_nt=fill(-2.,3),static_hi_nt=fill(3.,3),model_epoch=fill("fixture",3),
        dst_delta_1h_nt=[-1.,-5.,1.],coupling_active_mvm=[0.,1.,0.],
        row_key=["a","b","c"],a3_status=["absent","ok","ok"],
        a3_lo_nt=fill(-2.,3),a3_hi_nt=fill(3.,3))
    for name in P.POINT_MODELS
        df[!,name]=zeros(3)
    end
    for name in P.POINT_CANDIDATES
        df[!,Symbol(name,"_point")]=zeros(3)
        df[!,Symbol(name,"_lower")]=fill(-2.,3)
        df[!,Symbol(name,"_upper")]=fill(3.,3)
        df[!,Symbol(name,"_corrected")]=[true,false,true]
        for flag in ("cap_applied","projection_applied")
            df[!,Symbol(name,"_",flag)]=falses(3)
        end
        df[!,Symbol(name,"_actual_shift")]=zeros(3)
    end
    return E.annotate(df,Dict{DateTime,Int}())
end

function passing_inputs()
    ms=DataFrame([(candidate=name,cell=cell,model_step_hours=step,n=600,
        corrected_n=name in P.POINT_CANDIDATES ? 600 : 0,event_n=6,
        rmse_nt=name in P.POINT_CANDIDATES ? 1. : name=="oracle_realized" ? .01 : 2.,
        bias_nt=1.,coverage=.9,row_key_sha256="fixture") for name in [String.(P.POINT_MODELS)...,"v2_4f","oracle_realized",P.POINT_CANDIDATES...]
        for cell in ("all","storm","active_deepening","recovery","deep_dst") for step in (0,P.STEPS...)])
    gain=DataFrame([(candidate=name,cell=cell,model_step_hours=step,reference=ref,
        gain_nt=1.,lower_nt=.2) for name in P.POINT_CANDIDATES for cell in ("all","storm")
        for step in (0,P.STEPS...) for ref in ("v2_4e","static_v2_2")])
    ints=DataFrame([(candidate=name,complete_utc_days=7,coverage=.9,
        minimum_seven_day_coverage=.85,width_ratio=1.25,score_difference_nt=-1.,
        coverage_ci_lower=.88,coverage_ci_upper=.92,score_ci_upper_nt=-.2) for name in P.POINT_CANDIDATES])
    return ms,gain,ints
end

@testset "Point evaluation shared rows and scoring strata" begin
    panel=fixture()
    @test panel.issue_regime==["active_deepening","active_deepening","recovery"]
    @test panel.issue_week==fill(Date(2026,1,5),3)
    @test panel.driver_gap==fill("unknown",3)
    masks=Dict(E.cells(panel))
    @test masks["driver_unknown"]==trues(3)
    @test masks["lag_le_1h"]==trues(3)
    @test masks["storm"]==falses(3)
    m=E.metrics(panel,"Mean24")
    @test (m.raw_n,m.n,m.corrected_n)==(3,2,1)
    @test m.rmse_nt≈sqrt(5.) atol=1e-14 rtol=0
    @test m.mae_nt==2.
    @test m.bias_nt==1.
    @test m.error_min_nt==-1. && m.error_max_nt==3.
    @test m.coverage==1. && m.width_ratio==1. && m.interval_score_nt==5.
    @test m.row_key_sha256==E.metrics(panel,"v2_4e").row_key_sha256
    @test m.row_key_sha256=="7e18f737311b2dc3b2f269dd78396b0351f14fb66efa879f768cb23181883c78"
    bad=copy(panel);bad.Mean24_point[1]=NaN
    @test_throws ErrorException E.metrics(bad,"Mean24")
    bad=copy(panel);bad.Mean24_lower[1]=4.
    @test_throws ErrorException E.metrics(bad,"Mean24")
    empty=E.metrics(panel[1:0,:],"Mean24")
    @test empty.n==0 && isnan(empty.rmse_nt) && isnan(empty.coverage)
    @test empty.row_key_sha256=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    summary=E.summarize(panel;dataset="fixture",scope="fixture")
    @test nrow(summary)==20*7*18
    row=E.unique_row(summary;candidate="Mean24",cell="all",model_step_hours=2)
    @test row.n==2 && row.corrected_n==1
    daily=E.summarize(panel;dataset="fixture",scope="fixture",daily=true)
    @test nrow(daily)==2*18
    ci=E.interval_summary(panel;dataset="live",scope="fixture",start=DateTime(2026,1,5),stop=DateTime(2026,1,7))
    @test E.unique_row(ci.summary;candidate="A3_issued").n==1
    @test E.unique_row(ci.summary;candidate="Mean24").n==2
    @test E.unique_row(ci.summary;candidate="Mean24").row_key_sha256=="7e18f737311b2dc3b2f269dd78396b0351f14fb66efa879f768cb23181883c78"
    @test E.unique_row(ci.summary;candidate="A3_issued").row_key_sha256=="3e23e8160039594a33894f6564e1b1348bbd7a0088d42c4acb73eeaed59c009d"
    @test isnan(E.unique_row(ci.summary;candidate="Mean24").minimum_seven_day_coverage)
    @test_throws ArgumentError E.interval_summary(panel;dataset="live",scope="fixture",start=DateTime(2026,1,7),stop=DateTime(2026,1,5))
    gains=E.uncertainty(panel;dataset="live",scope="fixture")
    @test nrow(gains.summary)==2*7*6*2
    @test all(==(0.),gains.draws.gain_nt)
    @test E.unique_row(gains.summary;candidate="Mean24",reference="v2_4e",cell="all",model_step_hours=2).block_n==2
    @testset "complete seven-day windows" begin
        full=reduce(vcat,[panel[1:1,:] for _ in 1:8])
        full.issue_time_utc=DateTime(2026,1,1).+Day.(0:7)
        full.issue_day=Date.(full.issue_time_utc)
        full.row_key=string.(1:8)
        full.observation_dst_nt=[10.;fill(0.,7)]
        ints=E.interval_summary(full;dataset="live",scope="fixture",start=DateTime(2026,1,1),stop=DateTime(2026,1,9))
        @test E.unique_row(ints.summary;candidate="Mean24").complete_utc_days==8
        @test E.unique_row(ints.summary;candidate="Mean24").minimum_seven_day_coverage≈6/7 atol=1e-15
        short=E.interval_summary(full;dataset="live",scope="fixture",start=DateTime(2026,1,1),stop=DateTime(2026,1,7,23))
        @test E.unique_row(short.summary;candidate="Mean24").complete_utc_days==6
        @test isnan(E.unique_row(short.summary;candidate="Mean24").minimum_seven_day_coverage)
    end
end

@testset "Point advancement cannot pass missing support or weaker comparisons" begin
    ms,gain,ints=passing_inputs()
    gates=E.advancement(ms,gain,ints,Dict(0=>ms,1=>ms),Dict(0=>gain,1=>gain))
    @test all(gates.pass)
    ranks=DataFrame(candidate=collect(P.POINT_CANDIDATES),worst_step_gain_nt=ones(6),live_rmse_nt=ones(6))
    @test E.choose_candidate(gates,ranks)=="Mean24"
    partial=copy(gates);partial.pass[1]=false
    @test E.choose_candidate(partial,ranks)=="Mean48"
    r=copy(ranks);r.worst_step_gain_nt[6]=2.
    @test E.choose_candidate(gates,r)=="Median96"
    r=copy(ranks);r.live_rmse_nt[2]=.5
    @test E.choose_candidate(gates,r)=="Mean48"
    none=copy(gates);none.pass.=false
    @test E.choose_candidate(none,ranks)===nothing
    @test E.choose_candidate(gates[1:0,:],ranks)===nothing
    @test_throws ErrorException E.choose_candidate(gates[2:end,:],ranks)
    @test_throws ErrorException E.choose_candidate(vcat(gates,gates[1:1,:]),ranks)
    bad=copy(gates);bad.candidate[1]="noncandidate"
    @test_throws ErrorException E.choose_candidate(bad,ranks)
    for (field,value) in ((:n,39),(:corrected_n,39),(:coverage,.849))
        bad=copy(ms)
        mask=(bad.candidate .=="Mean24").&(bad.cell .=="all").&(bad.model_step_hours .==7)
        if field==:n
            mask=(bad.cell .=="all").&(bad.model_step_hours .==7)
            bad[mask,:corrected_n].=min.(bad[mask,:corrected_n],value)
        end
        bad[mask,field].=value
        got=E.advancement(bad,gain,ints,Dict(0=>ms,1=>ms),Dict(0=>gain,1=>gain))
        @test any(.!got.pass[got.candidate .=="Mean24"])
    end
    for (field,value) in ((:gain_nt,0.),(:lower_nt,0.),(:lower_nt,NaN))
        bad=copy(gain)
        mask=(bad.candidate .=="Mean24").&(bad.cell .=="all").&(bad.model_step_hours .==1)
        bad[mask,field].=value
        got=E.advancement(ms,bad,ints,Dict(0=>ms,1=>ms),Dict(0=>gain,1=>gain))
        @test any(.!got.pass[got.candidate .=="Mean24"])
    end
    for cell in ("all","storm","active_deepening","recovery","deep_dst")
        bad=copy(ms)
        mask=(bad.candidate .=="direct_gbm").&(bad.cell .==cell).&(bad.model_step_hours .==1)
        bad[mask,:rmse_nt].=0.
        got=E.advancement(ms,gain,ints,Dict(0=>ms,1=>bad),Dict(0=>gain,1=>gain))
        @test any(.!got.pass[(got.dataset .=="historical").&(got.delay_hours .==1).&(got.cell .==cell)])
    end
    for (field,value) in ((:n,29),(:event_n,4),(:bias_nt,10.01),(:coverage,.799))
        bad=copy(ms)
        mask=(bad.candidate .=="Mean24").&(bad.cell .=="storm").&(bad.model_step_hours .==2)
        if field==:n
            mask=(bad.cell .=="storm").&(bad.model_step_hours .==2)
            bad[mask,:corrected_n].=min.(bad[mask,:corrected_n],value)
        end
        bad[mask,field].=value
        got=E.advancement(ms,gain,ints,Dict(0=>bad,1=>ms),Dict(0=>gain,1=>gain))
        @test any(.!got.pass[got.candidate .=="Mean24"])
    end
    for (field,value) in ((:complete_utc_days,6),(:coverage,.879),(:coverage,.921),
        (:minimum_seven_day_coverage,.799),(:width_ratio,1.251),(:score_difference_nt,.001),
        (:coverage_ci_lower,.849),(:coverage_ci_lower,.901),(:coverage_ci_upper,.899),(:score_ci_upper_nt,.001))
        bad=copy(ints);bad[1,field]=value
        got=E.advancement(ms,gain,bad,Dict(0=>ms,1=>ms),Dict(0=>gain,1=>gain))
        @test any(.!got.pass[got.candidate .=="Mean24"])
    end
    @test_throws ErrorException E.advancement(ms[2:end,:],gain,ints,Dict(0=>ms,1=>ms),Dict(0=>gain,1=>gain))
    bad=copy(ms);bad.row_key_sha256[1]="different rows"
    @test_throws ErrorException E.advancement(bad,gain,ints,Dict(0=>ms,1=>ms),Dict(0=>gain,1=>gain))
    bad=copy(ms);bad.n[1]=1
    @test_throws ErrorException E.advancement(bad,gain,ints,Dict(0=>ms,1=>ms),Dict(0=>gain,1=>gain))
end
end
