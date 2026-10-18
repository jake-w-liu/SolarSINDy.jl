using Test, JSON3
include(joinpath(@__DIR__,"..","..","tools","initial_reference_audit","audit_prepared_reference.jl"))
length(ARGS)==2 || error("CASES_JSON RESULTS_JSON")
cases=JSON3.read(read(ARGS[1]),Vector{Dict{String,Any}})
results=Any[]
for case in cases
    accepted=false;reason=""
    try
        audit(case["plan"],case["sha256"]);accepted=true
    catch error
        error isa ArgumentError || rethrow()
        reason=sprint(showerror,error)
    end
    push!(results,(name=case["name"],accepted=accepted,reason=reason))
end
write(ARGS[2],JSON3.write(results))
@testset "Rehashed independent reader corruptions" begin
    for result in results
        @test !result.accepted
    end
    @test isempty(intersect(Set(pkg.name for pkg in keys(Base.loaded_modules)),
                           Set(("SolarSINDy","XGBoost","EvoTrees","BSON"))))
end
