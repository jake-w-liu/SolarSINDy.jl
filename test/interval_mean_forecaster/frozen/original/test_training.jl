using Dates, Test, SHA
include("DstIntervalAdapter.jl")
include("DstIntervalStream.jl")
include("DstIntervalTraining.jl")
using .DstIntervalAdapter, .DstIntervalStream, .DstIntervalTraining
t=DateTime(2018)
source=bytes2hex(sha256("synthetic-training-fixture"))
@testset "maturejointaffinegainfit" begin
    n=1000
    times=[t+Hour(i-1) for i in 1:n]
    values=ones(17,n);values[8,:].=400.
    values[6,:]=[-5.0+3sin(i/43) for i in 1:n]
    values[12,:].=2.
    values[15,:]=[-60.0+40sin(i/61) for i in 1:n]
    input=HourlyInputs(times,values;source_sha256=source)
    setting=settings()[7] # tau4/24,sigma1,eta0
    for delay in (0,1)
        rows=NamedTuple[]
        stream_basis(input;delay_hours=delay,taus=setting.taus,sigma=setting.sigma,sink=x->append!(rows,x))
        boundary=t+Hour(700)
        model=fit_gains(rows,boundary,setting;delay_hours=delay,source_sha256=source)
        @test all(isfinite,model.gains) && all(>=(0),model.gains)
        @test model.max_label_receipt==boundary-Hour(168)
        @test model.max_target_complete==boundary-Hour(168+delay)
        @test model.max_fit_issue==boundary-Hour(168+6+delay)
        @test model.informative_count==359-delay
        predicted=predict_rows(model,rows)
        @test predicted[100].prediction==(rows[100].anchor,rows[100].anchor)
        @test all(isfinite,predicted[500].prediction)
        changed=[row.issue>boundary-Hour(168) ? merge(row,(target=Tuple(isfinite(y) ? -100. : y for y in row.target),)) : row for row in rows]
        later=fit_gains(changed,boundary,setting;delay_hours=delay,source_sha256=source)
        @test later.gains==model.gains
        @test later.fit_data_sha256==model.fit_data_sha256
        @test later.fit_mask_sha256==model.fit_mask_sha256
        @test_throws ArgumentError fit_gains(rows,t+Hour(300),setting;delay_hours=delay,source_sha256=source)
        mixed=copy(rows);mixed[1]=merge(mixed[1],(sigma=10.,))
        @test_throws ArgumentError fit_gains(mixed,boundary,setting;delay_hours=delay,source_sha256=source)
    end
end

function selection_fixture()
    start=DateTime(2020);ending=start+Hour(500)
    template=NamedTuple[]
    for i in 1:10
        issue=start+Hour(24i);clocks=(forecast_clock(issue,4,0),forecast_clock(issue,6,0))
        target=i==1 ? (-60.,-60.) : (0.,0.)
        push!(template,(issue=issue,driver_tag=clocks[1].driver_tag,driver_receipt=issue,
            anchor_tag=clocks[1].anchor_tag,anchor_complete=issue,anchor_receipt=issue,anchor=-20.,
            target_tag=Tuple(c.target_tag for c in clocks),target_complete=Tuple(c.target_complete for c in clocks),
            target_receipt=Tuple(c.target_receipt for c in clocks),target=target,scoreable=(true,true),
            model_available=true,offset=(0.,0.),coefficients=((0.,0.),(0.,0.)),source_sha256=source,dst_delay_hours=0))
    end
    baseline4=[5.,2.,2.,2.,1.,1.,1.,0.,0.,0.]
    baseline6=2baseline4
    function panel(setting,e4,e6)
        [merge(row,(taus=setting.taus,sigma=setting.sigma,eta=setting.eta,setting_index=setting.index,
            prediction=(row.target[1]+e4[i],row.target[2]+e6[i]))) for (i,row) in enumerate(template)]
    end
    reference=(rows=panel(settings()[1],baseline4,baseline6),training_boundary=start,
        max_fit_target_complete=(start-Hour(168),start-Hour(168)),
        max_fit_label_receipt=(start-Hour(168),start-Hour(168)),source_sha256=source,delay_hours=0,
        model_ids=(bytes2hex(sha256("gbm-h4-initial-synthetic")),bytes2hex(sha256("gbm-h6-initial-synthetic"))),
        evidence_kind=:synthetic_initial_reference)
    panels=Dict{Int,Any}()
    for s in settings()
        panels[s.index]=panel(s,2baseline4,2baseline6)
    end
    panels[1]=panel(settings()[1],[5.,1.,2.,2.,1.,1.,1.,0.,0.,0.],2 .* [5.,1.,2.,2.,1.,1.,1.,0.,0.,0.])
    panels[2]=panel(settings()[2],[2.,2.,2.,2.,1.,1.,1.,0.,0.,0.],fill(5.,10))
    panels[3]=panel(settings()[3],[4.,2.,2.,2.,2.,2.,2.,0.,0.,0.],2 .* [4.,2.,2.,2.,2.,2.,2.,0.,0.,0.])
    panels[4]=panel(settings()[4],[4.,1.5,1.5,1.,1.,0.,0.,0.,0.,0.],[8.,4.5,4.5,3.,3.,0.,0.,0.,0.,0.])
    start,ending,panels,reference
end
@testset "stronginitialreferencegatedselection" begin
    start,ending,panels,reference=selection_fixture()
    result=select_setting(panels,reference;validation_start=start,validation_end=ending,delay_hours=0)
    @test result.status==:selected
    @test result.selected_index==3
    @test result.panels[3].qualifies_pooled
    @test !result.panels[2].qualifies_pooled
    @test result.panels[3].normalized_storm_rmse==result.panels[4].normalized_storm_rmse==.8
    @test length(result.panels)==18
    @test select_setting(panels,nothing;validation_start=start,validation_end=ending,delay_hours=0).status==:blocked_reference_absent
    none=Dict(k=>[merge(r,(prediction=Tuple(r.target[j]+100 for j in 1:2),)) for r in v] for (k,v) in panels)
    negative=select_setting(none,reference;validation_start=start,validation_end=ending,delay_hours=0)
    @test negative.status==:none && negative.selected_index===nothing && length(negative.panels)==18
    final=merge(reference,(training_boundary=DateTime(2023),))
    @test_throws ArgumentError select_setting(panels,final;validation_start=start,validation_end=ending,delay_hours=0)
    missing=copy(panels);missing[1]=missing[1][2:end]
    @test_throws ArgumentError select_setting(missing,reference;validation_start=start,validation_end=ending,delay_hours=0)
    conflict=copy(panels);conflict[1]=copy(conflict[1]);conflict[1][1]=merge(conflict[1][1],(target=(-61.,-60.),))
    @test_throws ArgumentError select_setting(conflict,reference;validation_start=start,validation_end=ending,delay_hours=0)
end
println("TRAINING_SHA256=",bytes2hex(sha256(read(joinpath(@__DIR__,"DstIntervalTraining.jl")))))
