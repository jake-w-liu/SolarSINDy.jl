using Test, Dates, JSON3, SHA, SolarSINDy
include(joinpath(@__DIR__,"..","..","tools","initial_reference_import","InitialReferenceImport.jl"))
const I=InitialReferenceImport
const F=SolarSINDy.IntervalMeanForecaster
const A=F.DstIntervalAdapter
length(ARGS)==2 || error("CASES_JSON RESULTS_JSON")
cases=JSON3.read(read(ARGS[1]),Vector{Dict{String,Any}})
results=Any[]
for case in cases
    accepted=false;reason="";checks=0
    try
        token=I.load_reference(case["directory"],case["sha256"];audit_completion_sha256=case["audit_sha256"])
        rows=I.reference_rows(token)
        if case["name"] in ("positive","relocated-output")
            expected=case["expected"]
            @test length(rows)==226;checks+=1
            for (slot,h) in enumerate((4,6))
                actual=[r for r in rows if r.status[slot]!="outside_selection_cohort"]
                @test length(actual)==expected["counts"][slot];checks+=1
                @test count(r->r.target[slot]<=-50,actual)==expected["storm_counts"][slot];checks+=1
                @test [string(r.issue) for r in actual]==expected["issues"][slot];checks+=1
                @test getindex.(getproperty.(actual,:prediction),slot)==Float64.(expected["predictions"][slot]);checks+=1
            end
            original=copy(rows);rows[1]=merge(rows[1],(anchor=123456.,))
            @test isequal(I.reference_rows(token),original);checks+=1
            rows=original
            panels=Dict(index=>[begin
                setting=A.settings()[index]
                merge(row,(prediction=row.target,setting_index=index,eta=setting.eta,taus=setting.taus,
                    sigma=setting.sigma,model_available=true,offset=(0.,0.),coefficients=((0.,0.),(0.,0.)),
                    model_id=repeat("d",64)))
            end for row in rows] for index in 1:18)
            meta=JSON3.read(read(joinpath(case["directory"],"completion.json")),Dict{String,Any})
            keywords=(validation_start=DateTime(meta["validation_start"]),validation_end=DateTime(meta["validation_end"]),delay_hours=0)
            selected=F.select_setting(panels,token;keywords...)
            @test selected.status==:selected;checks+=1
            @test selected.selected_index==1;checks+=1
            @test [r.pooled.n for r in selected.reference]==[224,222];checks+=1
            @test [r.storm.n for r in selected.reference]==[93,91];checks+=1
            @test selected.evidence_mode=="synthetic_only" && selected.scientific_verdict===false;checks+=1
            @test selected.independent_audit_completion_sha256==case["audit_sha256"];checks+=1
            omitted=copy(panels);omitted[1]=panels[1][1:end-1]
            @test_throws ArgumentError F.select_setting(omitted,token;keywords...);checks+=1
            wrong=copy(panels);wrong[1]=copy(panels[1]);wrong[1][1]=merge(wrong[1][1],(anchor=wrong[1][1].anchor+1,))
            @test_throws ArgumentError F.select_setting(wrong,token;keywords...);checks+=1
        elseif case["name"]=="post-load-mutation"
            open(joinpath(case["directory"],"selection_rows.csv"),"a") do stream;write(stream,"junk\n");end
            I.reference_rows(token)
        end
        accepted=true
    catch error
        error isa ArgumentError || rethrow()
        reason=sprint(showerror,error)
    end
    push!(results,(name=case["name"],accepted=accepted,reason=reason,checks=checks))
end
write(ARGS[2],JSON3.write(results))
@testset "Pinned serialized initial reference import" begin
    for result in results
        @test result.accepted==(result.name in ("positive","relocated-output"))
    end
    @test count(result->result.accepted,results)==2
end
