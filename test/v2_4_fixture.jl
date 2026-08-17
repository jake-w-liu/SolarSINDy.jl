# v2_4_fixture.jl — synthetic Task A fold tables in the V2.4 file contract.
#
# `validation/operational/v2_4_learn.jl` reads `oof_year_<Y>.csv` and nothing
# else, so its whole input surface can be manufactured. This file manufactures
# it: a driver and ring-current series with injected storms, the 29 issue-time
# feature columns derived from that series, the nine expert forecasts with
# different skill and independent noise, the comparator columns, and the
# realized-driver oracle.
#
# Two properties are built in deliberately, because they are what the learning
# stage is supposed to exploit and what a broken stage would fail to exploit:
#
#   * the experts carry independent noise around differently shrunk versions of
#     the true increment, so a non-negative combination beats every single one;
#   * every expert except persistence and climatology shares one systematic bias
#     that is a function of the issue-time features, so no convex combination can
#     remove it and only the residual layer can.
#
# It also injects the two conditions the storm gates need — rows with
# `Dst <= -100` nT and a one-hour fall steeper than `-15` nT/h — and a small
# share of Task A fallback rows (incomplete features, expert columns served) and
# of rows whose realized-driver oracle is absent.
#
# Definitions only; nothing runs on include.

using CSV
using DataFrames
using Dates
using Random
using SolarSINDy

"Model steps of the V2.4 study, repeated here so the fixture does not import the runner."
const V24_FIXTURE_STEPS = (1, 2, 3, 4, 6, 7)

"Hours of pre-history each fold window needs so that the 24 h Dst lag is defined."
const V24_FIXTURE_PREHISTORY_H = 26

"""
    _v24_fixture_series(rng, hours; storms) -> NamedTuple

Hourly driver and ring-current series. `Bz` is an AR(1) process with injected
southward storm intervals, `V` and `n` follow the storm with a speed bump and a
density pulse, and `Dst` integrates the rectified coupling against a decay
toward a quiet level, which is the qualitative behaviour every comparator in the
study is built around.
"""
function _v24_fixture_series(rng::AbstractRNG, hours::Int; storms::Int=5)
    bz = zeros(Float64, hours)
    by = zeros(Float64, hours)
    speed = fill(400.0, hours)
    density = fill(5.0, hours)
    for t in 2:hours
        bz[t] = 0.75 * bz[t - 1] + 2.2 * randn(rng)
        by[t] = 0.75 * by[t - 1] + 2.2 * randn(rng)
        speed[t] = 400.0 + 0.90 * (speed[t - 1] - 400.0) + 12.0 * randn(rng)
        density[t] = max(0.4, 5.0 + 0.85 * (density[t - 1] - 5.0) + 0.9 * randn(rng))
    end
    onsets = Int[]
    for s in 1:storms
        push!(onsets, clamp(round(Int, hours * (s - 0.5) / storms +
                                  0.15 * hours * (rand(rng) - 0.5)),
                            V24_FIXTURE_PREHISTORY_H + 4, hours - 12))
    end
    for onset in onsets
        main_phase = 5 + rand(rng, 0:7)
        depth = 14.0 + 16.0 * rand(rng)
        for k in 0:(main_phase - 1)
            t = onset + k
            t <= hours || break
            bz[t] = -depth - 3.0 * rand(rng)
            speed[t] = 520.0 + 180.0 * rand(rng)
            density[t] = 6.0 + 9.0 * rand(rng)
        end
    end
    pdyn = @. 2.0e-6 * density * speed^2
    dst = zeros(Float64, hours)
    dst[1] = -8.0
    quiet = -8.0
    tau = 9.0
    for t in 2:hours
        vbs = speed[t - 1] * max(0.0, -bz[t - 1]) / 1000.0
        dst[t] = dst[t - 1] - 6.2 * vbs - (dst[t - 1] - quiet) / tau + 1.1 * randn(rng)
        dst[t] = clamp(dst[t], -420.0, 40.0)
    end
    return (bz=bz, by=by, speed=speed, density=density, pdyn=pdyn, dst=dst)
end

"Rectified coupling proxy `VBs = V·max(0, -Bz)/1000` in mV/m."
_v24_fixture_vbs(speed::Float64, bz::Float64) = speed * max(0.0, -bz) / 1000.0

"""
    _v24_fixture_features(series, t) -> Vector{Float64}

The 29 issue-time feature columns for the issue whose index in `series` is `t`,
in `SolarSINDy.v23_direct_feature_names()` order. Driver feature lag `j` is
record `t - 1 - j` and Dst lag `j` is record `t - j`, which is the causal
convention of `operational_v23_features.jl`.
"""
function _v24_fixture_features(series, t::Int)
    bz = series.bz
    by = series.by
    speed = series.speed
    density = series.density
    dst = series.dst
    lag(j) = t - j
    bz_window = [bz[lag(j)] for j in 1:6]
    bz_mean6 = sum(bz_window) / 6
    bz_sd6 = sqrt(sum(abs2, bz_window .- bz_mean6) / 6)
    bperp = [hypot(by[lag(j)], bz[lag(j)]) for j in 1:6]
    logn = [log(density[lag(j)]) for j in 1:6]
    vbs = [_v24_fixture_vbs(speed[lag(j)], bz[lag(j)]) for j in 1:3]
    south_run = 0
    for j in 1:12
        bz[lag(j)] < 0.0 || break
        south_run += 1
    end
    adc = Float64[
        bz[lag(1)], bz[lag(2)], bz[lag(3)], bz_mean6, bz_sd6,
        by[lag(1)], bperp[1], sum(bperp) / 6,
        speed[lag(1)], speed[lag(1)] - speed[lag(7)],
        logn[1], sum(logn) / 6, series.pdyn[lag(1)],
        vbs[1], sum(vbs) / 3, Float64(south_run),
        dst[t], dst[t] - dst[lag(1)],
    ]
    extra = Float64[
        dst[lag(1)], dst[lag(2)], dst[lag(3)], dst[lag(4)], dst[lag(5)], dst[lag(6)],
        dst[lag(12)], dst[lag(24)],
        _v24_fixture_vbs(speed[lag(2)], bz[lag(2)]),
        _v24_fixture_vbs(speed[lag(3)], bz[lag(3)]),
        _v24_fixture_vbs(speed[lag(4)], bz[lag(4)]),
    ]
    return vcat(adc, extra)
end

"""
    v24_synthesize_fixture(dir; kwargs...) -> NamedTuple

Write `oof_year_<Y>.csv` for every year in `years` into `dir` under the Task A
file contract, plus a `manifest_year_<Y>.csv` companion so the learning stage's
provenance path is exercised. Returns the row counts and the storm-cell
populations, so a caller can assert that the fixture actually reaches the cells
the gates need.

`hours_per_year` contiguous hourly issues start at `start_month`/1 of each year;
contiguity matters because the residual layer needs six consecutive matured
innovations. `shadow_column` selects which of the two contracted spellings of
the V2.3 shadow column the fixture uses. With `separate_lat = true` the
lead-aware composition is written as its own weaker column beside the shadow, the
way a fold whose error layers were refitted persists them; with the default the
single composition column supplies both comparators.
"""
function v24_synthesize_fixture(dir::AbstractString; years=2013:2018,
                                hours_per_year::Int=2160, start_month::Int=1,
                                seed::Integer=20_260_817,
                                fallback_fraction::Real=0.01,
                                oracle_missing_fraction::Real=0.004,
                                shadow_column::Symbol=:v2_3_shadow,
                                separate_lat::Bool=false,
                                feature_prefix::AbstractString="f_")
    shadow_column in (:v2_3_shadow, :v2_3_lat) ||
        throw(ArgumentError("shadow column must be v2_3_shadow or v2_3_lat"))
    (separate_lat && shadow_column === :v2_3_lat) && throw(ArgumentError(
        "a separate lead-aware column needs the shadow column to be v2_3_shadow",
    ))
    mkpath(dir)
    feature_names = String.(SolarSINDy.v23_direct_feature_names())
    max_step = maximum(V24_FIXTURE_STEPS)
    counts = Dict{Int,Int}()
    cells = Dict{Symbol,Int}()
    for (index, year) in enumerate(years)
        rng = MersenneTwister(Int(seed) + 1_000 * index)
        hours = V24_FIXTURE_PREHISTORY_H + hours_per_year + max_step
        series = _v24_fixture_series(rng, hours)
        origin = DateTime(year, start_month, 1)
        rows = NamedTuple[]
        for k in 0:(hours_per_year - 1)
            t = V24_FIXTURE_PREHISTORY_H + k
            issue = origin + Hour(k)
            features = _v24_fixture_features(series, t)
            latest = series.dst[t]
            rate = series.dst[t] - series.dst[t - 1]
            coupling = _v24_fixture_vbs(series.speed[t - 1], series.bz[t - 1])
            is_fallback = rand(rng) < fallback_fraction
            # One systematic bias shared by every physics-informed expert: a
            # convex combination cannot remove it, the residual layer can.
            shared_bias = 0.35 * features[4] + 0.06 * features[16]
            for step in V24_FIXTURE_STEPS
                observation = series.dst[t + step]
                increment = observation - latest
                bias = shared_bias * min(step, 4) / 4
                served = latest + 0.95 * increment + bias + 3.0 * randn(rng)
                if is_fallback
                    values = Dict(
                        :served_v2_1 => served, :frozen_v2_1 => served,
                        :t1r_analog => served, :t1_analog_raw => served,
                        :direct_gbm => served, :v2_3_shadow => served,
                        :persistence => latest,
                        :burton => latest + 0.88 * increment + 5.0 * randn(rng),
                        :burton_full => latest + 0.90 * increment + 4.6 * randn(rng),
                        :obrien => latest + 0.89 * increment + 4.8 * randn(rng),
                        :climatology => latest * exp(-step / 12.0),
                    )
                else
                    values = Dict(
                        :served_v2_1 => served,
                        :frozen_v2_1 => latest + 0.93 * increment + bias +
                                        3.4 * randn(rng),
                        :t1r_analog => latest + 0.96 * increment + bias +
                                       2.8 * randn(rng),
                        :t1_analog_raw => latest + 0.94 * increment + bias +
                                          3.6 * randn(rng),
                        :direct_gbm => latest + 0.97 * increment + bias +
                                       2.6 * randn(rng),
                        :v2_3_shadow => latest + 0.965 * increment + bias +
                                        2.7 * randn(rng),
                        :persistence => latest,
                        :burton => latest + 0.88 * increment + 5.0 * randn(rng),
                        :burton_full => latest + 0.90 * increment + 4.6 * randn(rng),
                        :obrien => latest + 0.89 * increment + 4.8 * randn(rng),
                        :climatology => latest * exp(-step / 12.0),
                    )
                end
                static_v22 = 0.5 * values[:served_v2_1] + 0.5 * values[:persistence]
                oracle = rand(rng) < oracle_missing_fraction ? NaN :
                    latest + 0.95 * increment + 1.0 * randn(rng)
                row = Dict{Symbol,Any}(
                    :issue_time_utc => issue,
                    :model_step_hours => step,
                    :observation_dst_nt => observation,
                    :latest_dst_nt => latest,
                    :dst_delta_1h_nt => rate,
                    :coupling_active_mvm => coupling,
                    :fallback => is_fallback,
                    :served_v2_1 => values[:served_v2_1],
                    :frozen_v2_1 => values[:frozen_v2_1],
                    :persistence => values[:persistence],
                    :burton => values[:burton],
                    :burton_full => values[:burton_full],
                    :obrien => values[:obrien],
                    :static_v2_2 => static_v22,
                    :climatology => values[:climatology],
                    :t1_analog_raw => values[:t1_analog_raw],
                    :t1r_analog => values[:t1r_analog],
                    :direct_gbm => values[:direct_gbm],
                    shadow_column => values[:v2_3_shadow],
                    :oracle_realized => oracle,
                )
                # The composition before the error layers: the same center with a
                # little more of the shared bias left in it.
                separate_lat && (row[:v2_3_lat] = values[:v2_3_shadow] +
                                                  (is_fallback ? 0.0 : 0.4 * bias))
                for (j, name) in enumerate(feature_names)
                    row[Symbol(feature_prefix * name)] =
                        is_fallback ? NaN : features[j]
                end
                push!(rows, NamedTuple(row))
                for label in SolarSINDy.v23_regime_cells(latest, rate, coupling)
                    cells[Symbol(label)] = get(cells, Symbol(label), 0) + 1
                end
            end
        end
        frame = DataFrame(rows)
        sort!(frame, [:issue_time_utc, :model_step_hours])
        CSV.write(joinpath(dir, "oof_year_$(year).csv"), frame)
        CSV.write(joinpath(dir, "manifest_year_$(year).csv"), DataFrame((
            entry_type=["fixture", "fixture"], name=["rows", "hours"],
            count=[Float64(nrow(frame)), Float64(hours_per_year)],
            value=["synthetic", "synthetic"],
        )))
        counts[Int(year)] = nrow(frame)
    end
    return (rows=counts, cells=cells, dir=dir, years=collect(Int, years))
end
