module InitialReferenceImportTests

using Test, Dates, JSON3, SHA, SolarSINDy
include(joinpath(@__DIR__,"..","tools","initial_reference_import","InitialReferenceImport.jl"))
const I=InitialReferenceImport
const F=SolarSINDy.IntervalMeanForecaster
const A=F.DstIntervalAdapter
const oracle_path=joinpath(@__DIR__,"initial_reference","frozen","a_reference_import_literal_oracles_v1.json")
@test bytes2hex(sha256(read(oracle_path)))=="3c7acac28467b6a18f0c15c2b72b4d6f7d0ba0c2f7004afcffa9656673462d6a"
const O=JSON3.read(read(oracle_path),Dict{String,Any})
const SOURCE=repeat("a",64)

@testset "Pre-code separate-lead union and opposite-slot clocks" begin
    dst=[x===nothing ? NaN : Float64(x) for x in O["raw_dst"]]
    start=DateTime(O["raw_start"]);first=DateTime(O["validation_start"]);last=DateTime(O["validation_end"])
    cohorts=I.raw_cohorts(dst,start,first,first,last)
    @test cohorts==([4,6],[4])
    @test sort!(unique(vcat(cohorts...)))==[4,6]
    @test intersect(cohorts...)==[4]
    for expected in O["union_rows"]
        i=expected["origin_row"];issue=DateTime(expected["issue"])
        @test dst[i]==expected["anchor"]
        for (slot,h) in enumerate((4,6))
            clock=A.forecast_clock(issue,h,0)
            @test clock.driver_tag==DateTime(expected["driver_tag"])
            @test clock.anchor_tag==DateTime(expected["anchor_tag"])
            @test clock.anchor_complete==DateTime(expected["anchor_complete"])
            @test clock.anchor_receipt==DateTime(expected["anchor_receipt"])
            @test clock.target_tag==DateTime(expected["target_tags"][slot])
            @test clock.target_complete==DateTime(expected["target_complete"][slot])
            @test clock.target_receipt==DateTime(expected["target_receipt"][slot])
            @test dst[i+h]==expected["targets"][slot]
            @test A.fit_eligible(issue,clock.target_complete,clock.target_receipt,last)==expected["selection_eligible"][slot]
        end
    end
    missing=copy(dst);missing[12]=NaN
    @test I.raw_cohorts(missing,start,first,first,last)==([4,6],[4])
    @test !isfinite(missing[6+6])
    @test A.fit_eligible(first,DateTime(2020,1,1,6),DateTime(2020,1,1,6),last)
    @test !A.fit_eligible(first,DateTime(2020,1,1,6),DateTime(2020,1,1,6)+Millisecond(1),last)
end

function baseline_rows()
    [begin
        issue=DateTime(item["issue"]);clocks=Tuple(A.forecast_clock(issue,h,0) for h in (4,6))
        (issue=issue,driver_tag=clocks[1].driver_tag,driver_receipt=issue,anchor_tag=clocks[1].anchor_tag,
         anchor_complete=clocks[1].anchor_complete,anchor_receipt=clocks[1].anchor_receipt,
         anchor=Float64(item["anchor"]),target_tag=Tuple(c.target_tag for c in clocks),
         target_complete=Tuple(c.target_complete for c in clocks),target_receipt=Tuple(c.target_receipt for c in clocks),
         target=Tuple(Float64.(item["targets"])),scoreable=(true,true),source_sha256=SOURCE,dst_delay_hours=0,
         prediction=position==1 ? (-50.,-60.) : (-60.,NaN))
    end for (position,item) in enumerate(O["union_rows"])]
end
function panels_for(rows)
    Dict(index=>[begin
        setting=A.settings()[index]
        prediction=if index==1
            position==1 ? (-55.,-65.) : (-55.,NaN)
        elseif index in (2,4)
            position==1 ? (-51.,-61.) : (-61.,NaN)
        else
            position==1 ? (-45.,-55.) : (-55.,NaN)
        end
        merge(row,(prediction=prediction,setting_index=index,eta=setting.eta,taus=setting.taus,sigma=setting.sigma,
            model_available=true,offset=(0.,0.),coefficients=((0.,0.),(0.,0.)),model_id=repeat("d",64)))
    end for (position,row) in enumerate(rows)] for index in 1:18)
end

@testset "Frozen selection rejects the intersection-only winner" begin
    first=DateTime(O["validation_start"]);last=DateTime(O["validation_end"])
    rows=baseline_rows();panels=panels_for(rows)
    reference=(evidence_kind=:synthetic_initial_reference,training_boundary=first,delay_hours=0,
        max_fit_target_complete=(first-Hour(168),first-Hour(168)),max_fit_label_receipt=(first-Hour(168),first-Hour(168)),
        source_sha256=SOURCE,model_ids=(repeat("b",64),repeat("c",64)),rows=rows)
    result=F.select_setting(panels,reference;validation_start=first,validation_end=last,delay_hours=0)
    @test result.status==:selected
    @test result.selected_index==2
    @test !result.panels[1].qualifies_pooled
    @test result.panels[2].qualifies_pooled
    @test result.panels[2].normalized_storm_rmse==0.8
    @test result.panels[4].normalized_storm_rmse==0.8
    @test result.reference[1].pooled.n==2
    @test result.reference[2].pooled.n==1
    @test result.reference[1].pooled.rmse==5.
    @test result.reference[2].pooled.rmse==5.
    forged=merge(reference,(evidence_kind=:artifact_initial_reference,))
    @test_throws ArgumentError F.select_setting(panels,forged;validation_start=first,validation_end=last,delay_hours=0)
    absent=F.select_setting(Dict(),nothing;validation_start=first,validation_end=last,delay_hours=0)
    @test absent.status==:blocked_reference_absent
    @test absent.selected_index===nothing
    omitted=copy(panels);omitted[1]=omitted[1][1:1]
    @test_throws ArgumentError F.select_setting(omitted,reference;validation_start=first,validation_end=last,delay_hours=0)
    earlier=merge(reference,(training_boundary=first-Hour(1),
        max_fit_target_complete=(first-Hour(169),first-Hour(169)),
        max_fit_label_receipt=(first-Hour(169),first-Hour(169))))
    @test_throws ArgumentError F.select_setting(panels,earlier;validation_start=first,validation_end=last,delay_hours=0)
    @test F.DstIntervalTraining._select_initial_reference(panels,earlier;
        validation_start=first,validation_end=last,delay_hours=0).selected_index==2
    late=merge(earlier,(max_fit_target_complete=reference.max_fit_target_complete,
        max_fit_label_receipt=reference.max_fit_label_receipt))
    @test_throws ArgumentError F.DstIntervalTraining._select_initial_reference(panels,late;
        validation_start=first,validation_end=last,delay_hours=0)
end

@testset "Earlier initial model retains its original maturity boundary" begin
    path=joinpath(@__DIR__,"initial_reference","frozen","a_reference_subwindow_clock_oracle_v1.json")
    @test bytes2hex(sha256(read(path)))=="67fb7626df20fe129f6084d2cbc439ed58c7aa75e800b586b67b0da9f8322ae5"
    clock=JSON3.read(read(path),Dict{String,Any})
    boundary=DateTime(clock["training_boundary"]);first=DateTime(clock["selection_start"])
    legal=DateTime(clock["max_legal_fit_label"]);illegal=DateTime(clock["first_illegal_fit_label"])
    @test first==boundary+Hour(1)
    @test A.fit_eligible(boundary-Hour(180),legal,legal,boundary)
    @test !A.fit_eligible(boundary-Hour(180),illegal,illegal,boundary)
    @test A.fit_eligible(boundary-Hour(180),illegal,illegal,first)
end

@testset "Selection CSV preflight bounds physical records" begin
    mktempdir() do root
        path=joinpath(root,"rows.csv")
        write(path,join(I.HEADERS,',')*"\n"*join(fill("0",26),',')*"\n")
        @test I.preflight_selection_csv(path,1)===nothing
        for bytes in ("x"^4097*"\n",join(I.HEADERS,',')*"\n"*"\"quoted\"\n",join(I.HEADERS,',')*"\n"*"\0\n")
            write(path,bytes)
            @test_throws ArgumentError I.preflight_selection_csv(path,1)
        end
        write(path,join(I.HEADERS,',')*"\n"*join(fill("0",26),','))
        @test_throws ArgumentError I.preflight_selection_csv(path,1)
    end
end

end # module
