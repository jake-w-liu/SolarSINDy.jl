module V21OneHourInertiaTests

using Test
using DataFrames

const H1_SELECTION_SCRIPT = normpath(joinpath(
    @__DIR__, "..", "validation", "operational",
    "v2_1_one_hour_inertia_selection.jl",
))
include(H1_SELECTION_SCRIPT)

@testset "V2.1 one-hour inertia selection" begin
    broad = DataFrame(
        lead=fill(1, 8),
        obs=fill(1.5, 8),
        persistence=fill(0.0, 8),
        v2_0=fill(1.7, 8),
        v2_1_pre_one_hour_inertia=fill(2.0, 8),
        storm_split=vcat(fill("train", 4), fill("val", 4)),
    )
    candidates, selected = select_one_hour_inertia(broad)
    @test nrow(candidates) == length(H1_WEIGHT_GRID)
    @test selected.weight == 0.75
    @test selected.train_rmse_v2_1_nt == 0.0
    @test selected.validation_rmse_v2_1_nt == 0.0
    @test selected.passes_development_gate

    rows = _h1_rows(broad)
    @test all(_h1_candidate(rows, 0.75) .== 1.5)
    @test _h1_metric(rows, 0.75).improvement_vs_best_nt > 0.0

    broken = select(broad, Not(:v2_0))
    @test_throws ErrorException _h1_rows(broken)
end

end # module V21OneHourInertiaTests
