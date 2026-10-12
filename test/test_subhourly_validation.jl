module SubhourlyValidationTests

using Test
using SolarSINDy
using DataFrames
using Dates

include(joinpath(
    @__DIR__, "..", "validation", "operational", "validate_ballistic_subhourly.jl",
))

@testset "Subhourly target-pressure conversion" begin
    lib, ξ, _ = _shadow_library()
    ξ .= 0.0
    cal = default_operational_v2_calibration()
    issue = DateTime(2024, 5, 10, 18)
    issue_pdyn = dynamic_pressure(5.0, 400.0)
    issue_drv = (V=400.0, Bz=-5.0, By=1.0, n=5.0, Pdyn=issue_pdyn)
    anchor_star = -100.0

    # Measured-timeshift helper: every minute in each Earth-arrival hour was
    # already measured at L1. The second rollout step has the target pressure.
    earth_times = [issue + Hour(k) + Minute(m) for k in 0:1 for m in 0:59]
    hro = DataFrame(
        time=earth_times,
        ltime=[issue - Hour(2) + Minute(i - 1) for i in eachindex(earth_times)],
        V=vcat(fill(500.0, 60), fill(600.0, 60)),
        Bz=fill(-5.0, 120), By=fill(1.0, 120),
        n=vcat(fill(8.0, 60), fill(10.0, 60)),
        Pdyn=vcat(fill(dynamic_pressure(8.0, 500.0), 60),
                  fill(dynamic_pressure(10.0, 600.0), 60)),
    )
    raw, corrected = _subA_forecast(
        lib, ξ, anchor_star, issue_drv, hro, issue, -100.0, cal, 2,
    )
    expected = dst_star_to_dst(anchor_star, dynamic_pressure(10.0, 600.0))
    @test raw ≈ expected atol=1e-12
    @test corrected ≈ expected atol=1e-12
    @test raw != dst_star_to_dst(anchor_star, issue_drv.Pdyn)

    # Ballistic helper: populate exactly the L1 measurement-time window mapped
    # into the first target hour by the serving lag approximation.
    lag = Millisecond(round(Int, (L1_DIST_KM / issue_drv.V / 3600.0) * 3_600_000))
    ballistic_ltime = [issue - lag + Minute(m) for m in 0:59]
    ballistic = DataFrame(
        time=fill(issue, 60), ltime=ballistic_ltime,
        V=fill(700.0, 60), Bz=fill(-5.0, 60), By=fill(1.0, 60),
        n=fill(12.0, 60), Pdyn=fill(dynamic_pressure(12.0, 700.0), 60),
    )
    predicted = _subA_ballistic_forecast(
        lib, ξ, anchor_star, issue_drv, ballistic, issue, -100.0, cal, 1,
    )
    @test predicted ≈ dst_star_to_dst(anchor_star, dynamic_pressure(12.0, 700.0)) atol=1e-12
    @test predicted != dst_star_to_dst(anchor_star, issue_drv.Pdyn)

    # Equal pressure remains continuous with the frozen-driver conversion.
    frozen = DataFrame(
        time=DateTime[], ltime=DateTime[], V=Float64[], Bz=Float64[],
        By=Float64[], n=Float64[], Pdyn=Float64[],
    )
    raw_frozen, _ = _subA_forecast(
        lib, ξ, anchor_star, issue_drv, frozen, issue, -100.0, cal, 1;
        force_frozen=true,
    )
    @test raw_frozen ≈ dst_star_to_dst(anchor_star, issue_drv.Pdyn) atol=1e-12
end

# V10-01 regression: the lineage replays must label the anchor Dst rate with the TRUE 1 h delta
# Dst(it) - Dst(it-1h) from the Dst series (as ekf_storm/v2/lookahead/improved replays do), never the
# delta to the previous *processed* issue (which reads a multi-hour gap as nT/h and makes the first
# issue of every storm NaN). The storm loops need the OMNI HRO archive, which is not shipped, so this
# pins the loop source; a real-OMNI behavioural check of the envelope loop is kept in deep_debug/FIX_K.md.
@testset "Lineage replays use the true 1 h Dst rate (V10-01)" begin
    opdir = joinpath(@__DIR__, "..", "validation", "operational")
    for (file, fname) in (("v2_envelope_replay.jl", "replay_env_storm"),
                          ("v2_subhourly_A_replay.jl", "replay_subA_storm"),
                          ("validate_ballistic_subhourly.jl", "replay_ballistic_storm"))
        src = read(joinpath(opdir, file), String)
        start = findfirst("function $fname(", src)
        @test start !== nothing
        stop = findnext("\nend\n", src, last(start))
        body = src[first(start):last(stop)]
        @test occursin("prev1 = get(dst_map, it - Hour(1), NaN)", body)
        @test occursin("rate = isfinite(prev1) ? latest - prev1 : NaN", body)
        @test !occursin("prev_anchor", body)
    end
end

end
