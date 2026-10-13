#!/usr/bin/env julia

# analog_common.jl — machinery shared by the Operational analog ensemble development runner
# (`analog_development.jl`, plan section 4 selection) and the single-shot
# confirmatory runner (`analog_confirmatory.jl`, plan section 6 gates).
#
# The two runners differ only in which rows are fitted on and which rows are
# scored, so everything else lives here: the preregistered configuration grid,
# the anchor view of the base table, the multi-step ensemble rollout, the cell
# metrics and storm-safety guards, the lead-aware tail composition (LAT), the
# E-layers, and the artifact/manifest input–output helpers. Keeping one
# implementation is not a convenience: a development selection and a
# confirmatory score computed by two different code paths would not be
# comparable, and the preregistration requires that the configuration selected
# on DEV is the configuration scored on TEST.
#
# Time convention (inherited from the base table, hourly OMNI record tagged τ
# covering [τ, τ+1)): issue t, latest Dst = Dst(t), issue driver = record t−1,
# rate = Dst(t) − Dst(t−1), `future(k)` = record t+k−1, model steps
# {1,2,3,4,6,7}, target t + h.
#
# Run from the package root. This file defines only; it has no side effects
# beyond loading the replay chain and the base-table helpers.

using CSV
using DataFrames
using Dates
using JSON3
using LinearAlgebra
using Logging
using Printf
using Random
using SHA
using SolarSINDy
using Statistics

# `analog_kernel.jl` pulls in `operational_replay.jl` (and through it the live-forecast
# verify helpers): `_operational_forecast`, `_operational_admitted_driver`, `_transit_hours`,
# `_apply_inertial_sindy_safeguards`, `_advance_baselines`, `_dst_from_dst_star`,
# `OPERATIONAL_OUTPUT_DIR`. `analog_base_table.jl` adds the artifact paths, the
# partition labels and the frame → driver lookup. Both files include
# `operational_replay.jl`; the second include re-evaluates identical constants, which is
# a no-op, and is cheap (about one second) compared with re-deriving the
# constants here and letting them drift.
isdefined(@__MODULE__, :_selftest_analog) || include(joinpath(@__DIR__, "analog_kernel.jl"))
# The second pass through `operational_replay.jl` re-attaches identical docstrings, which
# Julia reports; the definitions themselves are unchanged, so the notices are
# silenced rather than left to bury a real warning in a long run log.
Logging.with_logger(Logging.NullLogger()) do
    isdefined(@__MODULE__, :ANALOG_BASE_DIR) || include(joinpath(@__DIR__, "analog_base_table.jl"))
end

# ---------------------------------------------------------------------------
# Preregistered grid and fixed protocol constants (plan sections 4 and 6)
# ---------------------------------------------------------------------------

"Model steps carried by the base table, in written order."
const ANALOG_MODEL_STEPS = (1, 2, 3, 4, 6, 7)
const ANALOG_STEP_COUNT = length(ANALOG_MODEL_STEPS)
const ANALOG_MAX_STEP = maximum(ANALOG_MODEL_STEPS)

"Column slot of each model step; zero for a step the study does not score."
const ANALOG_STEP_SLOT = let slots = zeros(Int, ANALOG_MAX_STEP)
    for (slot, h) in enumerate(ANALOG_MODEL_STEPS)
        slots[h] = slot
    end
    slots
end

"Analog distance weight sets (plan section 4, W1 and W2)."
const ANALOG_WEIGHT_SETS = (:uniform, :magnetic)
"Analog ensemble sizes; the K = 200 retrieval is nested, so all four are one search."
const ANALOG_T1_K_GRID = (25, 50, 100, 200)
"Ensemble size of the direct-value ablation T1a."
const ANALOG_T1A_K = 100
"Boosted driver-continuation hyper-grid (depth, rounds)."
const GBM_DRIVER_GRID = ((4, 200), (4, 400), (6, 200), (6, 400))
"Direct-GBM comparator hyper-grid (depth, rounds)."
const DIRECT_GBM_GRID = ((4, 200), (4, 400), (6, 200), (6, 400))
"Safeguard option S; `true` applies the published inertial-SINDy point operator."
const ANALOG_SAFEGUARD_OPTIONS = (true, false)
"Lead-aware tail composition weights (plan section 4, C1)."
const ANALOG_LAT_WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
"Model steps whose mean pooled RMSE drives the selection rule."
const ANALOG_SELECTION_STEPS = (2, 3, 6)
"Ridge penalties of the E1 innovation error state."
const ANALOG_E1_LAMBDA_GRID = (10.0, 100.0, 1000.0)
"Number of matured one-step innovations used by the E-layers."
const ANALOG_E_INNOVATION_LAGS = 6
"E2 boosted residual hyperparameters (plan section 4, E2)."
const ANALOG_E2_DEPTH = 3
const ANALOG_E2_ROUNDS = 128
"Preregistered seed for every stochastic component."
const ANALOG_SEED = 22022026
"Common learning rate and leaf weight of every boosted model in this study."
const GBM_ETA = 0.05
const GBM_MIN_WEIGHT = 64

"""
Regression target of the direct-GBM comparator. The comparator fits the change
of Dst over the lead, `Dst(t+h) - Dst(t)`, and the reported center is the fitted
value plus the issue-time Dst. Fitting the level instead lets the histogram
binning of the boosted learner collapse the whole deep tail of `dst0` into one
bin, which caps the attainable prediction near that bin's mean and loses the
storm rows the comparator exists to contest.
"""
const DIRECT_GBM_TARGET = "increment"

"Issue-time quantity the direct-GBM increment is measured from and added back to."
const DIRECT_GBM_TARGET_ANCHOR = "latest_dst_nt"

"Storm-safety cell minimum size for the plan section 6-A2 loss guard."
const ANALOG_GUARD_MIN_CELL_ROWS = 40
"Maximum RMSE a cell may lose to served inertial-SINDy (nT)."
const ANALOG_GUARD_MAX_CELL_LOSS_NT = 0.50
"Signed-error band of the intense deepening cell at six hours (nT)."
const ANALOG_GUARD_INTENSE_BIAS_NT = 10.0
"Model step at which the intense/extreme deepening guard is evaluated."
const ANALOG_GUARD_INTENSE_STEP = 6
"Latest-Dst and rate thresholds of the intense/extreme deepening cell."
const ANALOG_INTENSE_DST_NT = -100.0
const ANALOG_INTENSE_RATE_NT_PER_H = -15.0
"Disturbed-state thresholds of the two latest-Dst cells."
const ANALOG_CELL_DST_50_NT = -50.0
const ANALOG_CELL_DST_100_NT = -100.0

"""
    ANALOG_CELL_NAMES

Cells reported for every configuration and step. The labels come from
`SolarSINDy.ANALOG_CELL_LABELS` when the analog ensemble statistics module is loaded, so the
development tables, the confirmatory gates and the bootstrap all partition
issues identically; the literal tuple is the fallback used only when that module
is absent.
"""
const ANALOG_CELL_NAMES = isdefined(SolarSINDy, :ANALOG_CELL_LABELS) ?
    getfield(SolarSINDy, :ANALOG_CELL_LABELS) :
    (:all, :latest_le_m50, :latest_le_m100, :active_deepening, :recovery, :quiet,
     :intense_deepening)
"Cells that carry the plan section 6-A2 loss guard."
const ANALOG_GUARDED_CELLS = (:latest_le_m50, :latest_le_m100, :active_deepening, :recovery)

"Physical clamps applied to a continued driver, matching `analog_member_driver`."
const ANALOG_DRIVER_MIN_V_KMS = ANALOG_MEMBER_MIN_V_KMS
const ANALOG_DRIVER_MIN_N_CM3 = ANALOG_MEMBER_MIN_N_CM3
const ANALOG_DRIVER_MAX_N_CM3 = ANALOG_MEMBER_MAX_N_CM3

"Output roots. The confirmatory smoke run is deliberately a separate tree."
const ANALOG_DEV_DIR = joinpath(OPERATIONAL_OUTPUT_DIR, "analog_dev")
const ANALOG_DEV_SMOKE_DIR = joinpath(OPERATIONAL_OUTPUT_DIR, "analog_dev_smoke")
const ANALOG_TEST_DIR = joinpath(OPERATIONAL_OUTPUT_DIR, "analog_test")
const ANALOG_TEST_SMOKE_DIR = joinpath(OPERATIONAL_OUTPUT_DIR, "analog_test_smoke")

"Inner cross-validation blocks of the development runner (plan section 3)."
const ANALOG_DEV_BLOCKS = collect(ANALOG_CV_BLOCK_LABELS)

# ---------------------------------------------------------------------------
# Run plan
# ---------------------------------------------------------------------------

"""
    AnalogRunPlan

Everything a runner needs to know about *scale*: where artifacts go, which
inner blocks are visited, how many query anchors per fold are scored, and which
grid points are visited. Only a smoke run reduces the grid; the development and
confirmatory runs always carry the full preregistered grid, and
[`analog_assert_full_grid`](@ref) fails closed if they do not.
"""
struct AnalogRunPlan
    label::String
    outdir::String
    smoke::Bool
    blocks::Vector{Int}
    max_anchors::Int              # 0 = every eligible query anchor
    weight_sets::Vector{Symbol}
    t1_ks::Vector{Int}
    gdc_grid::Vector{Tuple{Int,Int}}
    direct_grid::Vector{Tuple{Int,Int}}
end

"Full preregistered development plan."
function analog_dev_plan(; smoke::Bool=false)
    # Two consecutive blocks, because a single block has no embargoed predecessor
    # and the error layers would never be exercised.
    smoke && return AnalogRunPlan(
        "dev_smoke", ANALOG_DEV_SMOKE_DIR, true, [2015, 2016], 2000,
        collect(ANALOG_WEIGHT_SETS), [25, 50], [(4, 200)], [(4, 200)],
    )
    return AnalogRunPlan(
        "dev", ANALOG_DEV_DIR, false, copy(ANALOG_DEV_BLOCKS), 0,
        collect(ANALOG_WEIGHT_SETS), collect(ANALOG_T1_K_GRID),
        collect(GBM_DRIVER_GRID), collect(DIRECT_GBM_GRID),
    )
end

"""
    analog_assert_full_grid(plan)

Fail closed when a non-smoke run would visit a reduced grid. The selection rule
is only preregistered over the full grid, so a silently trimmed development run
would produce a selection that the plan does not authorise.
"""
function analog_assert_full_grid(plan::AnalogRunPlan)
    plan.smoke && return plan
    Tuple(plan.weight_sets) == ANALOG_WEIGHT_SETS || error(
        "run plan $(plan.label) trims the preregistered weight sets",
    )
    Tuple(plan.t1_ks) == ANALOG_T1_K_GRID || error(
        "run plan $(plan.label) trims the preregistered analog ensemble sizes",
    )
    Tuple(plan.gdc_grid) == GBM_DRIVER_GRID || error(
        "run plan $(plan.label) trims the preregistered boosted-driver grid",
    )
    Tuple(plan.direct_grid) == DIRECT_GBM_GRID || error(
        "run plan $(plan.label) trims the preregistered direct-GBM grid",
    )
    plan.max_anchors == 0 || error(
        "run plan $(plan.label) caps the scored anchors",
    )
    return plan
end

_analog_sha256_file(path::AbstractString) = open(path, "r") do io
    bytes2hex(sha256(io))
end

_analog_sha256_text(text::AbstractString) = bytes2hex(sha256(String(text)))

"""
    analog_code_signature() -> String

SHA-256 over the sources that define a analog ensemble center: the three analog ensemble source
modules, the forecast kernel, and this file. A change to any of them
invalidates every persisted per-configuration artifact, which is what makes the
resume path safe.
"""
function analog_code_signature()
    package_root = normpath(joinpath(@__DIR__, "..", ".."))
    paths = [
        joinpath(package_root, "src", "analog_features.jl"),
        joinpath(package_root, "src", "analog.jl"),
        joinpath(package_root, "src", "analog_gbm.jl"),
        joinpath(package_root, "src", "analog_stats.jl"),
        joinpath(@__DIR__, "analog_kernel.jl"),
        joinpath(@__DIR__, "analog_common.jl"),
    ]
    parts = String[]
    for path in paths
        push!(parts, isfile(path) ? "$(basename(path))=$(_analog_sha256_file(path))" :
                                    "$(basename(path))=absent")
    end
    return _analog_sha256_text(join(parts, "\n"))
end

"Package root without a trailing separator, so `basename`/`dirname` behave on it."
_analog_package_root() = rstrip(normpath(joinpath(@__DIR__, "..", "..")), '/')

"""
    ANALOG_RESEARCH_PLAN_PATH

Preregistered research plan of the analog ensemble study. The plan lives in the study
repository beside this package rather than inside it, so the location is derived
from the package directory name and can be redirected with the
`ANALOG_RESEARCH_PLAN` environment variable for a relocated checkout.
"""
const ANALOG_RESEARCH_PLAN_PATH = let root = _analog_package_root(),
                                   study = replace(basename(root), "_analog_package" => "")
    _analog_env_path("ANALOG_RESEARCH_PLAN",
                  joinpath(dirname(root), study, "ANALOG_RESEARCH_PLAN.md"))
end

"""
    analog_provenance_paths() -> Vector{String}

Every file whose content can change what a analog ensemble run computes or what it was
authorised to compute: the analog ensemble runner and base-table sources, the replay chain
they include, the package sources, the deployed inertial-SINDy calibration, and the
preregistered plan. This is deliberately wider than
[`analog_code_signature`](@ref): it exists to make a confirmatory run auditable,
not to key the resume path, so adding a file here never invalidates a persisted
artifact.
"""
function analog_provenance_paths()
    root = _analog_package_root()
    paths = String[]
    for file in sort(readdir(@__DIR__))
        (startswith(file, "analog_") && endswith(file, ".jl")) || continue
        push!(paths, joinpath(@__DIR__, file))
    end
    for file in ("operational_replay.jl", "operational_lookahead_replay.jl", "ekf_storm_replay.jl", "paths.jl")
        push!(paths, joinpath(@__DIR__, file))
    end
    push!(paths, joinpath(root, "examples", "live_forecast_verify.jl"))
    source_dir = joinpath(root, "src")
    if isdir(source_dir)
        for file in sort(readdir(source_dir))
            endswith(file, ".jl") || continue
            push!(paths, joinpath(source_dir, file))
        end
    end
    push!(paths, joinpath(root, "deploy", "inertial_sindy_calibration.csv"))
    push!(paths, ANALOG_RESEARCH_PLAN_PATH)
    return paths
end

"Stable label of a provenance entry: package-relative where possible, otherwise the file name."
function _analog_provenance_label(path::AbstractString)
    root = _analog_package_root()
    full = abspath(path)
    return startswith(full, root * "/") ? full[length(root) + 2:end] : basename(full)
end

"""
    analog_provenance_signature() -> String

SHA-256 over the full include chain, the package sources, the deployed inertial-SINDy
calibration and the preregistered plan, as listed by
[`analog_provenance_paths`](@ref).

This signature is recorded in run manifests and is never consulted by the resume
path: [`analog_config_signature`](@ref) alone decides whether a persisted
configuration can be reused, so widening the provenance record cannot silently
discard an hour of computed artifacts.
"""
function analog_provenance_signature()
    parts = String[]
    for path in analog_provenance_paths()
        digest = isfile(path) ? _analog_sha256_file(path) : "absent"
        push!(parts, "$(_analog_provenance_label(path))=$(digest)")
    end
    return _analog_sha256_text(join(parts, "\n"))
end

# ---------------------------------------------------------------------------
# Multi-step ensemble rollout (the kernel variant the runners use)
# ---------------------------------------------------------------------------

"""
    analog_member_step_raw!(out, lib, ξ0, anchor_dst_star, issue_drv, future, members;
                         baselines = nothing) -> out

Fill `out[m, s]` with the *unclamped* raw analog ensemble center of ensemble member `m` at
model step `ANALOG_MODEL_STEPS[s]`, using one rollout per member to
`ANALOG_MAX_STEP`. The trajectory is step-independent — the L1 admission gate
`kΔ = ⌊transit(V_issue)⌋`, the admitted record and the member continuation all
depend on the rollout step alone, never on the horizon being scored — so the
six scored steps are read off a single integration instead of six. The
per-member value at step `k` is the kernel's `dst* + 7.26√Pdyn_k − 11` with the
step-`k` driver, exactly as `_analog_forecast` forms it for `h = k`; that equality
is the executable oracle in `test/test_analog_runners.jl`.

`baselines`, when given as a named tuple of `burton`, `burton_full` and
`obrien` matrices of the same shape, additionally advances the three causal
ring-current baselines along the *same* driver sequence. That is the
preregistered SINDy-removal ablation: the analog driver is fed to Burton and
O'Brien–McPherron instead of the SINDy core, and sharing this loop is what
guarantees the ablation differs from the candidate only in the integrator.
"""
function analog_member_step_raw!(out::AbstractMatrix{Float64}, lib, ξ0, anchor_dst_star::Float64,
                              issue_drv, future, members; baselines = nothing)
    size(out, 1) == length(members) || throw(DimensionMismatch(
        "member-step matrix has $(size(out, 1)) rows but $(length(members)) members were given",
    ))
    size(out, 2) == ANALOG_STEP_COUNT || throw(DimensionMismatch(
        "member-step matrix must have $(ANALOG_STEP_COUNT) step columns",
    ))
    isempty(members) && throw(ArgumentError("a analog ensemble rollout needs at least one member"))
    kΔ = floor(Int, _transit_hours(issue_drv.V))
    for (m, member) in enumerate(members)
        last_known = issue_drv
        filter = init_assimilation(lib, ξ0, Int[], anchor_dst_star)
        burton = anchor_dst_star
        burton_full = anchor_dst_star
        obrien = anchor_dst_star
        for k in 1:ANALOG_MAX_STEP
            drv_k = if k <= kΔ
                admitted = _operational_admitted_driver(future, k, kΔ, last_known)
                last_known = admitted
                admitted
            else
                member(k, last_known, kΔ)
            end
            assimilation_predict!(filter, drv_k)
            filter.mean[1] = clamp(filter.mean[1], -2000.0, 50.0)
            if baselines !== nothing
                burton = _advance_baselines(burton, drv_k).burton
                burton_full = _advance_baselines(burton_full, drv_k).burton_full
                obrien = _advance_baselines(obrien, drv_k).obrien
            end
            slot = ANALOG_STEP_SLOT[k]
            slot == 0 && continue
            out[m, slot] = current_dst(filter) + 7.26 * sqrt(max(drv_k.Pdyn, 0.0)) - 11.0
            if baselines !== nothing
                baselines.burton[m, slot] = _dst_from_dst_star(burton, drv_k.Pdyn)
                baselines.burton_full[m, slot] = _dst_from_dst_star(burton_full, drv_k.Pdyn)
                baselines.obrien[m, slot] = _dst_from_dst_star(obrien, drv_k.Pdyn)
            end
        end
    end
    return out
end

"""
    analog_prefix_means!(dest, out, ks) -> dest

`dest[j, s] = mean(out[1:ks[j], s])`. The nested-K identity of the analog
search — `analog_knn` returns neighbours in increasing distance, so the first
`ks[j]` rows of `out` are exactly the `ks[j]`-nearest ensemble — turns four
ensemble sizes into four prefix means of one rollout set. Each mean is taken
over a materialised vector so the summation order matches `_analog_forecast`'s
`mean(member_pred)` bit for bit.
"""
function analog_prefix_means!(dest::AbstractMatrix{Float64}, out::AbstractMatrix{Float64},
                           ks::AbstractVector{<:Integer})
    size(dest) == (length(ks), size(out, 2)) || throw(DimensionMismatch(
        "prefix-mean destination must be length(ks) × step count",
    ))
    issorted(ks) || throw(ArgumentError("prefix ensemble sizes must be sorted"))
    isempty(ks) && throw(ArgumentError("at least one prefix ensemble size is required"))
    first(ks) >= 1 || throw(ArgumentError("prefix ensemble sizes must be positive"))
    last(ks) <= size(out, 1) || throw(ArgumentError(
        "largest prefix ensemble size $(last(ks)) exceeds the $(size(out, 1)) available members",
    ))
    for s in axes(out, 2), (j, k) in enumerate(ks)
        dest[j, s] = mean(out[1:k, s])
    end
    return dest
end

"""
    analog_center(raw, corr, latest, h, rate, safeguards) -> Float64

Turn an ensemble-mean raw center into the reported center: add the deployed
inertial-SINDy ridge correction, then either project to the physical range (S = off) or
apply the published inertial-SINDy point operator (S = on). `raw` must be the *unclamped*
ensemble mean, because `_analog_forecast` clamps the raw only for reporting and
feeds the unclamped sum to the safeguard stage.
"""
@inline function analog_center(raw::Float64, corr::Float64, latest::Float64, h::Int,
                            rate::Float64, safeguards::Bool)
    total = raw + corr
    return safeguards ?
        _apply_inertial_sindy_safeguards(total, latest, h, isfinite(rate) ? rate : 0.0) :
        clamp(total, -2000.0, 50.0)
end

# ---------------------------------------------------------------------------
# Anchor view of the base table
# ---------------------------------------------------------------------------

const _ANALOG_DRIVER_NT = NamedTuple{(:V, :Bz, :By, :n, :Pdyn),NTuple{5,Float64}}

"""
    AnalogAnchors

Issue-time view of the base table. One entry per issue (the analog search, the
rollout and the cells are all issue-level quantities) with the six model steps
carried as columns of the per-step matrices. `present[i, s]` records whether the
base table actually holds the `(issue i, step s)` row: a target whose Dst was
never observed has no row, and scoring one would invent an observation.
"""
struct AnalogAnchors
    issue::Vector{DateTime}
    index::Dict{DateTime,Int}
    partition::Vector{String}
    cv_block::Vector{Int}
    drv::Vector{_ANALOG_DRIVER_NT}
    latest::Vector{Float64}
    rate::Vector{Float64}
    coupling::Vector{Float64}
    anchor_star::Vector{Float64}
    present::BitMatrix
    obs::Matrix{Float64}
    served::Matrix{Float64}
    frozen::Matrix{Float64}
    raw_served::Matrix{Float64}
    persistence::Matrix{Float64}
    burton::Matrix{Float64}
    burton_full::Matrix{Float64}
    obrien::Matrix{Float64}
    static_stack::Matrix{Float64}
    corr::Matrix{Float64}
end

Base.length(a::AnalogAnchors) = length(a.issue)

"""
    analog_read_base_table(; path, partitions) -> DataFrame

Read the base table, keeping only the requested partitions. Every column is
needed: the explicit comparator columns feed the metrics and the calibration
feature columns feed the inertial-SINDy ridge correction.
"""
function analog_read_base_table(; path::AbstractString=ANALOG_BASE_TABLE,
                             partitions=("DEV",))
    isfile(path) || error("analog ensemble base table is missing: $path (run analog_base_table.jl)")
    table = CSV.read(path, DataFrame;
                     types=Dict("issue_time_utc" => DateTime,
                                "target_time_utc" => DateTime))
    wanted = Set(String.(partitions))
    keep = [p in wanted for p in table.partition]
    table = table[keep, :]
    nrow(table) > 0 || error("base table holds no rows for partitions $(collect(partitions))")
    issorted(collect(zip(table.issue_time_utc, table.model_step_hours))) ||
        sort!(table, [:issue_time_utc, :model_step_hours])
    return table
end

"""
    analog_row_correction(row, cal) -> Float64

Deployed inertial-SINDy ridge correction of a base-table row. The correction depends only
on the row's calibration features, which are fixed by the frozen inertial-SINDy replay,
so it is identical for every analog ensemble candidate at that `(issue, step)` and can be
computed once. `_analog_forecast` evaluates exactly this quantity when it is handed
the base-table row as `calibration_features`.
"""
function analog_row_correction(row, cal)
    feats = NamedTuple{Tuple(cal.feature_names)}(
        Tuple(Float64(getproperty(row, c)) for c in cal.feature_names),
    )
    return SolarSINDy.operational_inertial_sindy_correction(cal, feats)
end

"""
    analog_build_anchors(table, cal) -> AnalogAnchors

Collapse the `(issue, step)` base table into the issue-level anchor view. The
issue-level columns are asserted constant across an issue's steps, because the
rollout uses one issue driver and one anchor for all six steps and a silent
disagreement would make the six centers incomparable.
"""
function analog_build_anchors(table::DataFrame, cal)
    n_rows = nrow(table)
    issues = DateTime[]
    sizehint!(issues, div(n_rows, ANALOG_STEP_COUNT) + 1)
    index = Dict{DateTime,Int}()
    partition = String[]
    cv_block = Int[]
    drv = _ANALOG_DRIVER_NT[]
    latest = Float64[]
    rate = Float64[]
    coupling = Float64[]
    anchor_star = Float64[]
    for i in 1:n_rows
        issue = table.issue_time_utc[i]
        haskey(index, issue) && continue
        push!(issues, issue)
        index[issue] = length(issues)
        push!(partition, String(table.partition[i]))
        push!(cv_block, Int(table.cv_block[i]))
        pdyn = Float64(table.Pdyn_npa[i])
        push!(drv, (V=Float64(table.V_kms[i]), Bz=Float64(table.Bz_nt[i]),
                    By=Float64(table.By_nt[i]), n=Float64(table.n_cm3[i]), Pdyn=pdyn))
        push!(latest, Float64(table.latest_dst_nt[i]))
        push!(rate, Float64(table.dst_delta_1h_nt[i]))
        push!(coupling, Float64(table.coupling_active_mvm[i]))
        push!(anchor_star, pressure_correct_dst([Float64(table.latest_dst_nt[i])], [pdyn])[1])
    end
    n = length(issues)
    present = falses(n, ANALOG_STEP_COUNT)
    new_matrix() = fill(NaN, n, ANALOG_STEP_COUNT)
    obs, served, frozen, raw_served = new_matrix(), new_matrix(), new_matrix(), new_matrix()
    persistence, burton = new_matrix(), new_matrix()
    burton_full, obrien, static_stack, corr = new_matrix(), new_matrix(), new_matrix(), new_matrix()
    for i in 1:n_rows
        issue = table.issue_time_utc[i]
        a = index[issue]
        step = Int(table.model_step_hours[i])
        slot = ANALOG_STEP_SLOT[step]
        slot == 0 && error("base table row at $issue carries unscored model step $step")
        present[a, slot] && error("base table repeats issue $issue at step $step")
        # The issue-level state must be one state per issue; a disagreement here
        # would silently give the six steps of one issue different rollouts.
        (Float64(table.V_kms[i]) == drv[a].V && Float64(table.Bz_nt[i]) == drv[a].Bz &&
         Float64(table.By_nt[i]) == drv[a].By && Float64(table.n_cm3[i]) == drv[a].n &&
         Float64(table.Pdyn_npa[i]) == drv[a].Pdyn &&
         Float64(table.latest_dst_nt[i]) == latest[a] &&
         Float64(table.dst_delta_1h_nt[i]) == rate[a] &&
         Float64(table.coupling_active_mvm[i]) == coupling[a]) || error(
            "base table issue $issue carries inconsistent issue-time state across steps",
        )
        present[a, slot] = true
        obs[a, slot] = Float64(table.observation_dst_nt[i])
        served[a, slot] = Float64(table.served_inertial_sindy_dst_nt[i])
        frozen[a, slot] = Float64(table.frozen_inertial_sindy_dst_nt[i])
        raw_served[a, slot] = Float64(table.raw_sindy_dst_nt[i])
        persistence[a, slot] = Float64(table.persistence_dst_nt[i])
        burton[a, slot] = Float64(table.burton_dst_nt[i])
        burton_full[a, slot] = Float64(table.burton_full_dst_nt[i])
        obrien[a, slot] = Float64(table.obrien_dst_nt[i])
        static_stack[a, slot] = Float64(table.static_stack_dst_nt[i])
        corr[a, slot] = analog_row_correction(table[i, :], cal)
    end
    return AnalogAnchors(issues, index, partition, cv_block, drv, latest, rate, coupling,
                      anchor_star, present, obs, served, frozen, raw_served, persistence,
                      burton, burton_full, obrien, static_stack, corr)
end

"""
    analog_check_correction!(anchors, table) -> Float64

Independent check of the cached ridge correction: for every unclamped base-table
row the frozen inertial-SINDy center must equal the frozen raw core plus the correction.
Returns the largest residual. This is an oracle, not a restatement — the
correction is recomputed from the calibration coefficients while the frozen
center and the frozen raw core come from the archived replay.
"""
function analog_check_correction!(anchors::AnalogAnchors, table::DataFrame)
    worst = 0.0
    for i in 1:nrow(table)
        a = anchors.index[table.issue_time_utc[i]]
        slot = ANALOG_STEP_SLOT[Int(table.model_step_hours[i])]
        frozen = anchors.frozen[a, slot]
        # The base table stores clamp(raw + corr); comparing on the interior of
        # the physical range keeps the identity free of the projection.
        (-1999.0 < frozen < 49.0) || continue
        residual = abs(frozen - (Float64(table.pred_dst_nt[i]) + anchors.corr[a, slot]))
        worst = max(worst, residual)
    end
    worst <= 1e-9 || error(
        "cached inertial-SINDy correction disagrees with the archived frozen center by $worst nT",
    )
    return worst
end

# ---------------------------------------------------------------------------
# Storm-safety cells (plan section 6-A2)
# ---------------------------------------------------------------------------

"""
    _analog_cell_labels_local(latest, rate, coupling) -> Vector{Symbol}

Plan section 6-A2 cells derived directly from the protocol text, kept as an
independent statement of the definition: the two ring-current depth cells, the
static stack causal regime of the issue, and the intense/extreme deepening cell
(Dst(t) ≤ −100 nT and rate < −15 nT/h, strict) whose six-hour behaviour carries
its own guard. It is the fallback when the analog ensemble statistics module is absent and
the cross-check partner when it is present.
"""
function _analog_cell_labels_local(latest::Real, rate::Real, coupling::Real)
    lat = Float64(latest)
    rat = Float64(rate)
    labels = Symbol[:all]
    lat <= ANALOG_CELL_DST_50_NT && push!(labels, :latest_le_m50)
    lat <= ANALOG_CELL_DST_100_NT && push!(labels, :latest_le_m100)
    push!(labels, static_stack_regime(lat, rat, Float64(coupling)))
    (lat <= ANALOG_INTENSE_DST_NT && rat < ANALOG_INTENSE_RATE_NT_PER_H) &&
        push!(labels, :intense_deepening)
    return labels
end

"""
    analog_cell_labels(latest, rate, coupling) -> Vector{Symbol}

Cells an issue belongs to, from issue-time state only. The analog ensemble statistics
module owns the definition when it is loaded so that the development tables and
the confirmatory gates cannot drift apart.
"""
analog_cell_labels(latest::Real, rate::Real, coupling::Real) =
    isdefined(SolarSINDy, :analog_regime_cells) ?
        Symbol.(getfield(SolarSINDy, :analog_regime_cells)(latest, rate, coupling)) :
        _analog_cell_labels_local(latest, rate, coupling)

"""
    analog_regime_cells_source() -> String

Which implementation of the cell labelling is in force, after cross-checking the
statistics module against the protocol text on a state grid that crosses every
threshold. A disagreement aborts immediately rather than after hours of scoring,
because a development selection and a confirmatory score computed under
different cell definitions would not be comparable.
"""
function analog_regime_cells_source()
    isdefined(SolarSINDy, :analog_regime_cells) || return "analog_common"
    reference = getfield(SolarSINDy, :analog_regime_cells)
    for latest in (-250.0, -120.0, -100.0, -60.0, -50.0, -20.0, 5.0),
        rate in (-40.0, -20.0, -15.0, -6.0, -1.0, 0.0, 4.0),
        coupling in (0.0, 0.5, 3.0)
        mine = Set(_analog_cell_labels_local(latest, rate, coupling))
        theirs = Set(Symbol.(reference(latest, rate, coupling)))
        mine == theirs || error(
            "SolarSINDy.analog_regime_cells disagrees with the plan section 6-A2 cell definition " *
            "at latest=$latest rate=$rate coupling=$coupling: $(sort(collect(theirs))) vs " *
            "$(sort(collect(mine)))",
        )
    end
    return "SolarSINDy.analog_regime_cells"
end

"""
    analog_cell_masks(anchors) -> NamedTuple

Per-anchor membership of every reported cell. Cells depend on issue-time state
alone, so they are computed once per anchor and reused by every configuration
and step.
"""
function analog_cell_masks(anchors::AnalogAnchors)
    n = length(anchors)
    masks = Dict{Symbol,BitVector}(name => falses(n) for name in ANALOG_CELL_NAMES)
    for i in 1:n
        for label in analog_cell_labels(anchors.latest[i], anchors.rate[i], anchors.coupling[i])
            haskey(masks, label) || error("cell labelling emitted the unknown cell $label")
            masks[label][i] = true
        end
    end
    all(masks[:all]) || error("the pooled cell must contain every anchor")
    return NamedTuple{ANALOG_CELL_NAMES}(Tuple(masks[name] for name in ANALOG_CELL_NAMES))
end

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

"Root-mean-square of a vector of errors; `NaN` when the sample is empty."
function analog_rmse(errors::AbstractVector{<:Real})
    isempty(errors) && return NaN
    total = 0.0
    for e in errors
        total += abs2(Float64(e))
    end
    return sqrt(total / length(errors))
end

"""
    analog_metric_rows(name, centers, anchors, masks, rows, fallback) -> Vector{NamedTuple}

Per-step, per-cell metrics of a candidate center matrix against the served inertial-SINDy
product. `rows` lists the anchor indices that are scored; a `(anchor, step)`
pair contributes only when the base table holds that row and the candidate
center is finite. `fallback` marks anchors served by inertial-SINDy because their analog
key was incomplete, and its per-step share is reported so a configuration
cannot look good by declining to forecast.
"""
function analog_metric_rows(name::AbstractString, centers::AbstractMatrix{Float64},
                         anchors::AnalogAnchors, masks, rows::AbstractVector{Int},
                         fallback::AbstractVector{Bool})
    out = NamedTuple[]
    for (slot, step) in enumerate(ANALOG_MODEL_STEPS)
        scored = [i for i in rows if anchors.present[i, slot] && isfinite(centers[i, slot])]
        expected = count(i -> anchors.present[i, slot], rows)
        length(scored) == expected || error(
            "$name lost $(expected - length(scored)) scoreable rows at step $step to " *
            "non-finite centers",
        )
        fallback_fraction = isempty(scored) ? NaN :
            count(i -> fallback[i], scored) / length(scored)
        for cell in ANALOG_CELL_NAMES
            mask = getproperty(masks, cell)
            idx = [i for i in scored if mask[i]]
            candidate_err = [anchors.obs[i, slot] - centers[i, slot] for i in idx]
            served_err = [anchors.obs[i, slot] - anchors.served[i, slot] for i in idx]
            rmse = analog_rmse(candidate_err)
            rmse_served = analog_rmse(served_err)
            push!(out, (
                config=String(name),
                model_step_hours=step,
                cell=String(cell),
                n=length(idx),
                rmse_nt=rmse,
                bias_nt=isempty(idx) ? NaN : mean(candidate_err),
                rmse_served_nt=rmse_served,
                bias_served_nt=isempty(idx) ? NaN : mean(served_err),
                loss_vs_served_nt=rmse - rmse_served,
                fallback_fraction=fallback_fraction,
            ))
        end
    end
    return out
end

"""
    analog_guard_flags(metrics) -> NamedTuple

Plan section 6-A2 storm-safety guards evaluated on the metric rows of one
configuration: no guarded cell with at least `ANALOG_GUARD_MIN_CELL_ROWS` rows may
lose more than 0.50 nT of RMSE to served inertial-SINDy, and the intense/extreme
deepening cell at six hours must lose no RMSE and keep its mean signed error
inside ±10 nT. Cells that are empty pass vacuously and their sizes are reported
so a vacuous pass is visible.
"""
function analog_guard_flags(metrics)
    cell_ok = true
    failures = String[]
    for row in metrics
        Symbol(row.cell) in ANALOG_GUARDED_CELLS || continue
        row.n >= ANALOG_GUARD_MIN_CELL_ROWS || continue
        isfinite(row.loss_vs_served_nt) || continue
        if row.loss_vs_served_nt > ANALOG_GUARD_MAX_CELL_LOSS_NT
            cell_ok = false
            push!(failures, @sprintf("%s@%dh:+%.3f", row.cell, row.model_step_hours,
                                     row.loss_vs_served_nt))
        end
    end
    intense = [row for row in metrics
               if row.cell == "intense_deepening" &&
                  row.model_step_hours == ANALOG_GUARD_INTENSE_STEP]
    intense_n = isempty(intense) ? 0 : first(intense).n
    intense_ok = true
    if !isempty(intense) && intense_n > 0
        row = first(intense)
        if !(isfinite(row.loss_vs_served_nt) && row.loss_vs_served_nt <= 0.0)
            intense_ok = false
            push!(failures, @sprintf("intense_deepening@6h_rmse_loss:%+.3f",
                                     row.loss_vs_served_nt))
        end
        if !(isfinite(row.bias_nt) && abs(row.bias_nt) <= ANALOG_GUARD_INTENSE_BIAS_NT)
            intense_ok = false
            push!(failures, @sprintf("intense_deepening@6h_bias:%+.2f", row.bias_nt))
        end
    end
    return (cell_guard_ok=cell_ok, intense_guard_ok=intense_ok,
            guards_ok=cell_ok && intense_ok, intense_rows=intense_n,
            guard_failures=join(failures, "|"))
end

"""
    analog_pooled_rmse(centers, anchors, rows, slot) -> (rmse, n)

Pooled RMSE of a candidate center matrix at one model step over the given
anchor rows.
"""
function analog_pooled_rmse(centers::AbstractMatrix{Float64}, anchors::AnalogAnchors,
                         rows::AbstractVector{Int}, slot::Int)
    total = 0.0
    n = 0
    for i in rows
        (anchors.present[i, slot] && isfinite(centers[i, slot])) || continue
        total += abs2(anchors.obs[i, slot] - centers[i, slot])
        n += 1
    end
    return (n == 0 ? NaN : sqrt(total / n), n)
end

# ---------------------------------------------------------------------------
# Run context and folds
# ---------------------------------------------------------------------------

"""
    AnalogContext

Everything both runners need that does not depend on the configuration: the
anchor view, the hourly frame and its driver lookup, the frozen inertial-SINDy core and
calibration, the issue-time analog features with their completeness and origin
eligibility masks, the cell masks, and the artifact hashes that make a resumed
run verifiable.
"""
mutable struct AnalogContext
    plan::AnalogRunPlan
    anchors::AnalogAnchors
    frame::DataFrame
    lookup::Dict{DateTime,_ANALOG_DRIVER_NT}
    core::Any
    cal::Any
    X::Matrix{Float64}
    ok::BitVector
    origin_ok::BitVector
    masks::Any
    direct_cache::Any
    base_table_sha::String
    frame_sha::String
    code_sha::String
    cells_source::String
    correction_residual_nt::Float64
end

"""
    AnalogFold

One fitting/scoring split. In the development runner a fold is an inner
cross-validation block: `train` holds the block's training archive (DEV anchors
whose maximum target precedes the block by at least 168 h) and `query` holds the
block's issues. In the confirmatory runner there is one fold whose `train` is
all of DEV and whose `query` is TEST.
"""
struct AnalogFold
    name::String
    train::Vector{Int}
    query::Vector{Int}
end

"""
    analog_build_context(plan; partitions, table_path, frame_path) -> AnalogContext

Load the base table and the hourly frame, build the anchor view, evaluate the
analog ensemble issue-time features once for every anchor, and record the artifact hashes.
The ridge-correction cache is checked against the archived frozen inertial-SINDy centers
before anything else runs.
"""
function analog_build_context(plan::AnalogRunPlan; partitions=("DEV",),
                           table_path::AbstractString=ANALOG_BASE_TABLE,
                           frame_path::AbstractString=ANALOG_BASE_HOURLY_FRAME)
    isfile(frame_path) || error("analog ensemble hourly frame is missing: $frame_path")
    cells_source = analog_regime_cells_source()
    core = load_operational_core(OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION)
    cal = read_inertial_sindy_calibration(
        operational_calibration_artifacts(OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION).point_csv,
    )
    frame = CSV.read(frame_path, DataFrame; types=Dict("time_utc" => DateTime))
    lookup = analog_frame_driver_lookup(frame)
    table = analog_read_base_table(; path=table_path, partitions=partitions)
    anchors = analog_build_anchors(table, cal)
    residual = analog_check_correction!(anchors, table)
    table = DataFrame()   # release the row view; the anchor view carries everything downstream
    X, ok = analog_feature_matrix(frame, anchors.issue)
    origin_ok = analog_origin_ok(lookup, anchors.issue)
    masks = analog_cell_masks(anchors)
    return AnalogContext(plan, anchors, frame, lookup, core, cal, X, ok, origin_ok, masks,
                      nothing, _analog_sha256_file(table_path), _analog_sha256_file(frame_path),
                      analog_code_signature(), cells_source, residual)
end

"""
    direct_gbm_features!(ctx) -> (X, ok)

Direct-GBM design matrix for every anchor, built on first use. The comparator is
the only consumer, so a development run that stops before it never pays for it.
"""
function direct_gbm_features!(ctx::AnalogContext)
    ctx.direct_cache === nothing || return ctx.direct_cache
    ctx.direct_cache = direct_gbm_features(ctx.frame, ctx.anchors.issue)
    return ctx.direct_cache
end

"""
    analog_dev_folds(ctx) -> Vector{AnalogFold}

Inner rolling-origin folds. For block `b` the training archive is every DEV
anchor whose maximum target (issue + 7 h) precedes the block's first issue by at
least 168 h, exactly the plan section 3 rule that `analog_block_windows` records
in the base manifest.
"""
function analog_dev_folds(ctx::AnalogContext)
    anchors = ctx.anchors
    folds = AnalogFold[]
    for block in ctx.plan.blocks
        query = [i for i in eachindex(anchors.issue) if anchors.cv_block[i] == block]
        isempty(query) && error("inner block $block has no DEV anchors")
        cutoff = minimum(anchors.issue[query]) - Hour(ANALOG_BASE_EMBARGO_HOURS)
        train = [i for i in eachindex(anchors.issue)
                 if anchors.partition[i] == "DEV" &&
                    anchors.issue[i] + Hour(ANALOG_MAX_STEP) <= cutoff]
        isempty(train) && error("inner block $block has an empty training archive")
        if ctx.plan.max_anchors > 0 && length(query) > ctx.plan.max_anchors
            query = query[1:ctx.plan.max_anchors]
        end
        push!(folds, AnalogFold("block_$(block)", train, query))
    end
    return folds
end

# ---------------------------------------------------------------------------
# Configuration results
# ---------------------------------------------------------------------------

"""
    AnalogConfigResult

Scored output of one configuration family. `raw` holds the *unclamped* ensemble
mean at each `(anchor, step)` for tail candidates and the model center for the
direct comparator; `fallback[i]` marks an anchor whose analog key was incomplete
and which is therefore served by the deployed inertial-SINDy product. `rows` lists every
anchor the configuration was asked to score, so a configuration cannot improve
its metrics by scoring fewer rows.
"""
struct AnalogConfigResult
    id::String
    family::String
    params::Dict{String,Any}
    raw::Matrix{Float64}
    fallback::Vector{Bool}
    rows::Vector{Int}
    applies_correction::Bool
end

function analog_empty_result(ctx::AnalogContext, id, family, params; applies_correction::Bool=true)
    n = length(ctx.anchors)
    return AnalogConfigResult(String(id), String(family), params,
                           fill(NaN, n, ANALOG_STEP_COUNT), fill(false, n), Int[],
                           applies_correction)
end

"""
    analog_config_centers(result, anchors, safeguards) -> Matrix{Float64}

Reported centers of one configuration. Tail candidates take the deployed inertial-SINDy
ridge correction and, when `safeguards` is set, the published inertial-SINDy point
operator; fallback anchors take the served product; the direct comparator
predicts the observation directly and takes neither stage.
"""
function analog_config_centers(result::AnalogConfigResult, anchors::AnalogAnchors, safeguards::Bool)
    centers = fill(NaN, size(result.raw))
    for i in result.rows, slot in 1:ANALOG_STEP_COUNT
        anchors.present[i, slot] || continue
        if result.fallback[i]
            centers[i, slot] = anchors.served[i, slot]
        elseif result.applies_correction
            centers[i, slot] = analog_center(
                result.raw[i, slot], anchors.corr[i, slot], anchors.latest[i],
                ANALOG_MODEL_STEPS[slot], anchors.rate[i], safeguards,
            )
        else
            centers[i, slot] = clamp(result.raw[i, slot], -2000.0, 50.0)
        end
    end
    return centers
end

"Reported raw column: the ensemble mean projected to the physical range."
analog_report_raw(result::AnalogConfigResult, i::Int, slot::Int) =
    clamp(result.raw[i, slot], -2000.0, 50.0)

# ---------------------------------------------------------------------------
# Artifact input–output and resume signatures
# ---------------------------------------------------------------------------

analog_oof_path(plan::AnalogRunPlan, id) = joinpath(plan.outdir, "oof_$(id).csv")
analog_summary_path(plan::AnalogRunPlan, id) = joinpath(plan.outdir, "summary_$(id).csv")
analog_signature_path(plan::AnalogRunPlan, id) = joinpath(plan.outdir, "sig_$(id).txt")

"""
    analog_config_signature(ctx, id, params) -> String

Resume key of one configuration: the configuration itself, the scale of the run,
the base table, the hourly frame and the analog ensemble sources. A persisted artifact is
reused only when this string matches, so a changed input or a changed kernel
invalidates it instead of silently mixing generations.
"""
function analog_config_signature(ctx::AnalogContext, id::AbstractString, params::AbstractDict)
    payload = Dict{String,Any}(
        "config" => String(id),
        "params" => Dict{String,Any}(String(k) => v for (k, v) in params),
        "plan" => ctx.plan.label,
        "blocks" => ctx.plan.blocks,
        "max_anchors" => ctx.plan.max_anchors,
        "base_table_sha256" => ctx.base_table_sha,
        "hourly_frame_sha256" => ctx.frame_sha,
        "code_sha256" => ctx.code_sha,
        "cells" => ctx.cells_source,
    )
    return _analog_sha256_text(JSON3.write(payload))
end

"""
    analog_config_is_current(ctx, id, params) -> Bool

Whether the persisted artifacts of a configuration can be reused.
"""
function analog_config_is_current(ctx::AnalogContext, id::AbstractString, params::AbstractDict)
    signature_path = analog_signature_path(ctx.plan, id)
    (isfile(signature_path) && isfile(analog_oof_path(ctx.plan, id)) &&
     isfile(analog_summary_path(ctx.plan, id))) || return false
    return strip(read(signature_path, String)) == analog_config_signature(ctx, id, params)
end

"""
    analog_write_config!(ctx, result, seconds) -> NamedTuple

Persist one configuration: the per-row out-of-fold centers, the per-step and
per-cell metrics of both safeguard variants, and the resume signature. Metrics
are returned so the caller does not have to read them back.
"""
function analog_write_config!(ctx::AnalogContext, result::AnalogConfigResult, seconds::Real)
    mkpath(ctx.plan.outdir)
    anchors = ctx.anchors
    centers_on = analog_config_centers(result, anchors, true)
    centers_off = analog_config_centers(result, anchors, false)
    rows = NamedTuple[]
    for i in result.rows, slot in 1:ANALOG_STEP_COUNT
        anchors.present[i, slot] || continue
        push!(rows, (
            issue_time_utc=string(anchors.issue[i]),
            cv_block=anchors.cv_block[i],
            model_step_hours=ANALOG_MODEL_STEPS[slot],
            fallback=result.fallback[i],
            raw_dst_nt=analog_report_raw(result, i, slot),
            center_s_on_dst_nt=centers_on[i, slot],
            center_s_off_dst_nt=centers_off[i, slot],
        ))
    end
    CSV.write(analog_oof_path(ctx.plan, result.id), DataFrame(rows))
    metrics = NamedTuple[]
    for (safeguards, centers) in ((true, centers_on), (false, centers_off))
        id = "$(result.id)_S$(safeguards ? "on" : "off")"
        for row in analog_metric_rows(id, centers, anchors, ctx.masks, result.rows, result.fallback)
            push!(metrics, merge(row, (
                family=result.family,
                safeguards=safeguards,
                params_json=JSON3.write(result.params),
                seconds=Float64(seconds),
            )))
        end
    end
    CSV.write(analog_summary_path(ctx.plan, result.id), DataFrame(metrics))
    write(analog_signature_path(ctx.plan, result.id),
          analog_config_signature(ctx, result.id, result.params))
    return (metrics=metrics, centers_on=centers_on, centers_off=centers_off)
end

"""
    analog_read_config(ctx, id) -> NamedTuple

Rebuild the center matrices and the metric rows of a persisted configuration.
Used by the resume path and by the composition stages, which need the selected
configuration's centers without recomputing the ensemble.
"""
function analog_read_config(ctx::AnalogContext, id::AbstractString)
    anchors = ctx.anchors
    n = length(anchors)
    centers_on = fill(NaN, n, ANALOG_STEP_COUNT)
    centers_off = fill(NaN, n, ANALOG_STEP_COUNT)
    raw = fill(NaN, n, ANALOG_STEP_COUNT)
    fallback = fill(false, n)
    seen = falses(n)
    oof = CSV.read(analog_oof_path(ctx.plan, id), DataFrame;
                   types=Dict("issue_time_utc" => DateTime))
    for r in 1:nrow(oof)
        i = get(anchors.index, oof.issue_time_utc[r], 0)
        i == 0 && error("persisted configuration $id holds an unknown issue time")
        slot = ANALOG_STEP_SLOT[Int(oof.model_step_hours[r])]
        raw[i, slot] = Float64(oof.raw_dst_nt[r])
        centers_on[i, slot] = Float64(oof.center_s_on_dst_nt[r])
        centers_off[i, slot] = Float64(oof.center_s_off_dst_nt[r])
        fallback[i] = Bool(oof.fallback[r])
        seen[i] = true
    end
    metrics = CSV.read(analog_summary_path(ctx.plan, id), DataFrame)
    return (rows=findall(seen), raw=raw, centers_on=centers_on, centers_off=centers_off,
            fallback=fallback, metrics=metrics)
end

# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

# --- configuration identities -------------------------------------------------
# The runner needs the identity and parameter record of a configuration *before*
# it is computed, to decide whether a persisted artifact can be reused. Building
# them here keeps the resume key and the artifact that answers to it in one place.

analog_id(family, weight_set, k) = "$(family)_$(weight_set)_K$(k)"
analog_params(family, weight_set, k, direct) = Dict{String,Any}(
    "tail" => String(family), "weight_set" => String(weight_set), "k" => Int(k),
    "direct" => Bool(direct),
)
gbm_driver_id(depth, rounds) = "T2_d$(depth)_r$(rounds)"
gbm_driver_params(depth, rounds) = Dict{String,Any}(
    "tail" => "T2", "depth" => Int(depth), "rounds" => Int(rounds), "eta" => GBM_ETA,
    "min_weight" => GBM_MIN_WEIGHT, "seed" => ANALOG_SEED,
)
direct_gbm_id(depth, rounds) = "DGBM_d$(depth)_r$(rounds)"
direct_gbm_params(depth, rounds) = Dict{String,Any}(
    "comparator" => "direct_gbm", "depth" => Int(depth), "rounds" => Int(rounds),
    "eta" => GBM_ETA, "min_weight" => GBM_MIN_WEIGHT, "seed" => ANALOG_SEED,
    "target" => DIRECT_GBM_TARGET, "target_anchor" => DIRECT_GBM_TARGET_ANCHOR,
)
const ANALOG_ORACLE_ID = "ORACLE_realized"
analog_oracle_params() = Dict{String,Any}("tail" => "realized_driver")


_analog_log(msg) = (println(msg); flush(stdout))

"Mark the anchors of a fold that cannot be forecast by a candidate and are served instead."
function _analog_mark_fallback!(result::AnalogConfigResult, anchors::AnalogAnchors,
                             fallback_rows::AbstractVector{Int})
    for i in fallback_rows
        result.fallback[i] = true
        for slot in 1:ANALOG_STEP_COUNT
            anchors.present[i, slot] || continue
            result.raw[i, slot] = anchors.raw_served[i, slot]
        end
    end
    return result
end

"""
    analog_run_analog_stage!(ctx, folds, weight_set; ks, direct, family) -> Dict{Int,AnalogConfigResult}

Analog-driver-continuation ensembles for one distance weight set. Per fold the
archive is searched once at `K = maximum(ks)` and each query anchor is rolled
out once per retrieved neighbour; because `analog_knn` returns neighbours in
increasing distance, the smaller ensembles are prefix means of the same rollout
set rather than separate searches. `direct = true` is the preregistered T1a
ablation, which copies the analog speed and density instead of continuing the
query's own values by the archived increments.
"""
function analog_run_analog_stage!(ctx::AnalogContext, folds, weight_set::Symbol;
                               ks::AbstractVector{<:Integer}, direct::Bool=false,
                               family::AbstractString="T1")
    anchors = ctx.anchors
    sorted_ks = sort(collect(Int, ks))
    kmax = last(sorted_ks)
    weights = analog_weights(weight_set)
    results = [
        analog_empty_result(ctx, analog_id(family, weight_set, k), family,
                         analog_params(family, weight_set, k, direct))
        for k in sorted_ks
    ]
    lib, xi = ctx.core.library, ctx.core.coefficients
    lookup = ctx.lookup
    for fold in folds
        started = time()
        archive = [i for i in fold.train if ctx.ok[i] && ctx.origin_ok[i]]
        length(archive) >= kmax || error(
            "fold $(fold.name) archive holds $(length(archive)) eligible origins, fewer than K = $kmax",
        )
        query = sort(fold.query)
        query_ok = [i for i in query if ctx.ok[i]]
        query_fallback = [i for i in query if !ctx.ok[i]]
        stats = analog_feature_stats(ctx.X[archive, :])
        archive_z = analog_standardize(ctx.X[archive, :], stats.mean, stats.sd)
        query_z = analog_standardize(ctx.X[query_ok, :], stats.mean, stats.sd)
        archive_times = anchors.issue[archive]
        neighbours = analog_knn(query_z, anchors.issue[query_ok], archive_z, archive_times, kmax;
                             weights=weights)
        Threads.@threads :static for j in eachindex(query_ok)
            i = query_ok[j]
            drv = anchors.drv[i]
            issue = anchors.issue[i]
            future = k -> get(lookup, issue + Hour(k - 1), nothing)
            members = [
                analog_member(drv, lookup, archive_times[neighbours[j, m]]; direct=direct)
                for m in 1:kmax
            ]
            out = Matrix{Float64}(undef, kmax, ANALOG_STEP_COUNT)
            analog_member_step_raw!(out, lib, xi, anchors.anchor_star[i], drv, future, members)
            prefix = Matrix{Float64}(undef, length(sorted_ks), ANALOG_STEP_COUNT)
            analog_prefix_means!(prefix, out, sorted_ks)
            for kk in eachindex(sorted_ks), slot in 1:ANALOG_STEP_COUNT
                results[kk].raw[i, slot] = prefix[kk, slot]
            end
        end
        for result in results
            _analog_mark_fallback!(result, anchors, query_fallback)
            append!(result.rows, query)
        end
        _analog_log(@sprintf(
            "    %s %s%s fold %s: archive %d, queries %d (%d served fallback), %.1f s",
            family, weight_set, direct ? " direct" : "", fold.name, length(archive),
            length(query), length(query_fallback), time() - started,
        ))
    end
    return Dict(k => results[kk] for (kk, k) in enumerate(sorted_ks))
end

"""
    analog_run_gdc_stage!(ctx, folds, depth, rounds) -> AnalogConfigResult

Boosted driver continuation (T2). Per fold one regressor is fitted for each
(driver variable, rollout step) pair on the training archive's issue-time
features, and the predicted conditional-mean trajectory drives a single SINDy
rollout. Speed and density are reconstructed from the predicted log increments
relative to the issue driver and are held inside the same physical bounds the
analog members use, so the two tails differ in how the continuation is chosen
and not in what counts as an admissible driver.
"""
function analog_run_gdc_stage!(ctx::AnalogContext, folds, depth::Int, rounds::Int)
    anchors = ctx.anchors
    result = analog_empty_result(ctx, gbm_driver_id(depth, rounds), "T2", gbm_driver_params(depth, rounds))
    lib, xi = ctx.core.library, ctx.core.coefficients
    lookup = ctx.lookup
    feature_names = String.(collect(ANALOG_FEATURE_NAMES))
    for fold in folds
        started = time()
        train = [i for i in fold.train if ctx.ok[i]]
        isempty(train) && error("fold $(fold.name) has no boosted-driver training rows")
        query = sort(fold.query)
        query_ok = [i for i in query if ctx.ok[i]]
        query_fallback = [i for i in query if !ctx.ok[i]]
        stats = analog_feature_stats(ctx.X[train, :])
        train_z = analog_standardize(ctx.X[train, :], stats.mean, stats.sd)
        query_z = analog_standardize(ctx.X[query_ok, :], stats.mean, stats.sd)
        nq = length(query_ok)
        predicted = (bz=zeros(nq, ANALOG_MAX_STEP), by=zeros(nq, ANALOG_MAX_STEP),
                     dlogv=zeros(nq, ANALOG_MAX_STEP), dlogn=zeros(nq, ANALOG_MAX_STEP))
        for k in 1:ANALOG_MAX_STEP
            targets = gbm_driver_targets(ctx.frame, anchors.issue[train], k)
            for (variable, y) in zip((:bz, :by, :dlogv, :dlogn), targets)
                keep = findall(isfinite, y)
                length(keep) >= 2 * GBM_MIN_WEIGHT || error(
                    "fold $(fold.name) has only $(length(keep)) finite $variable targets at step $k",
                )
                model = gbm_fit(train_z[keep, :], y[keep];
                                    max_depth=depth, nrounds=rounds, eta=GBM_ETA,
                                    min_weight=GBM_MIN_WEIGHT, seed=ANALOG_SEED,
                                    feature_names=feature_names)
                getproperty(predicted, variable)[:, k] = gbm_predict(model, query_z)
            end
        end
        Threads.@threads :static for j in 1:nq
            i = query_ok[j]
            drv = anchors.drv[i]
            issue = anchors.issue[i]
            future = k -> get(lookup, issue + Hour(k - 1), nothing)
            # The issue driver is the increment anchor; the floors keep the
            # logarithm defined on a degenerate archived record.
            log_v0 = log(max(drv.V, ANALOG_DRIVER_MIN_V_KMS))
            log_n0 = log(max(drv.n, ANALOG_DRIVER_MIN_N_CM3))
            member = function (k, last_known, kΔ)
                speed = max(exp(log_v0 + predicted.dlogv[j, k]), ANALOG_DRIVER_MIN_V_KMS)
                density = clamp(exp(log_n0 + predicted.dlogn[j, k]),
                                ANALOG_DRIVER_MIN_N_CM3, ANALOG_DRIVER_MAX_N_CM3)
                return (V=speed, Bz=predicted.bz[j, k], By=predicted.by[j, k],
                        n=density, Pdyn=dynamic_pressure(density, speed))
            end
            out = Matrix{Float64}(undef, 1, ANALOG_STEP_COUNT)
            analog_member_step_raw!(out, lib, xi, anchors.anchor_star[i], drv, future, [member])
            for slot in 1:ANALOG_STEP_COUNT
                result.raw[i, slot] = out[1, slot]
            end
        end
        _analog_mark_fallback!(result, anchors, query_fallback)
        append!(result.rows, query)
        _analog_log(@sprintf(
            "    T2 d%d r%d fold %s: train %d, queries %d (%d served fallback), %.1f s",
            depth, rounds, fold.name, length(train), length(query), length(query_fallback),
            time() - started,
        ))
    end
    return result
end

"""
    direct_gbm_anchor(anchors, i) -> Float64

Issue-time Dst the direct-GBM increment is measured from and added back to.
"""
@inline function direct_gbm_anchor(anchors::AnalogAnchors, i::Integer)
    latest = anchors.latest[i]
    isfinite(latest) || error(
        "direct-GBM anchor at $(anchors.issue[i]) has no finite latest Dst, so its " *
        "increment target is undefined",
    )
    return latest
end

"""
    direct_gbm_target(anchors, rows, slot) -> Vector{Float64}

Direct-GBM regression target on `rows`: the change of Dst over the lead,
`Dst(t+h) - Dst(t)`. [`direct_gbm_center`](@ref) inverts it, so the two are only
ever correct as a pair.
"""
direct_gbm_target(anchors::AnalogAnchors, rows, slot::Integer) =
    [anchors.obs[i, slot] - direct_gbm_anchor(anchors, i) for i in rows]

"Center of a fitted direct-GBM increment at anchor `i`."
direct_gbm_center(anchors::AnalogAnchors, i::Integer, increment::Real) =
    Float64(increment) + direct_gbm_anchor(anchors, i)

"""
    direct_gbm_check_anchor(features, anchors, rows)

Fail closed unless the issue-time Dst the increment is measured from is exactly
the `dst0` column of the design matrix. The comparator is persisted for
deployment and a loader can only invert the increment with a quantity it reads
from its own feature vector, so the two must be the same number.
"""
function direct_gbm_check_anchor(features::AbstractMatrix, anchors::AnalogAnchors, rows)
    column = analog_feature_index(:dst0)
    for i in rows
        features[i, column] == anchors.latest[i] || error(
            "direct-GBM increment anchor disagrees with the design matrix at " *
            "$(anchors.issue[i]): dst0 = $(features[i, column]) but " *
            "latest_dst_nt = $(anchors.latest[i])",
        )
    end
    return nothing
end

"""
    analog_run_direct_stage!(ctx, folds, depth, rounds) -> AnalogConfigResult

Direct machine-learning comparator (plan section 5): one boosted regressor per
model step predicting Dst at the target hour from the issue-time analog features
plus the Dst and coupling lags, fitted on the same training archive as the
candidate. It receives neither the inertial-SINDy correction nor the inertial-SINDy safeguards,
because it is not a inertial-SINDy derivative; anchors whose lag block is incomplete take
the same served-inertial-SINDy fallback as the candidates, so both are scored on the same
rows.

The regressed quantity is the increment `Dst(t+h) - Dst(t)` and the reported
center is the fitted increment plus the issue-time Dst, so a persisted model is
only a forecast once `latest_dst_nt` is added back. The level is not regressed
directly: the histogram binning of the boosted learner puts the whole deep tail
of `dst0` into one bin, so a level fit cannot resolve a storm anchor from the
quiet anchors that share that bin and its deepest attainable prediction is close
to the bin's mean.

`on_model`, when given, is called as `on_model(fold, slot, model, feature_names)`
after each fit and only observes it, so persisting the comparator for deployment
cannot move a score.
"""
function analog_run_direct_stage!(ctx::AnalogContext, folds, depth::Int, rounds::Int;
                               on_model=nothing)
    anchors = ctx.anchors
    result = analog_empty_result(ctx, direct_gbm_id(depth, rounds), "DGBM",
                              direct_gbm_params(depth, rounds); applies_correction=false)
    features, usable = direct_gbm_features!(ctx)
    feature_names = String.(direct_gbm_feature_names())
    for fold in folds
        started = time()
        query = sort(fold.query)
        query_ok = [i for i in query if usable[i]]
        query_fallback = [i for i in query if !usable[i]]
        query_features = features[query_ok, :]
        direct_gbm_check_anchor(features, anchors, query_ok)
        for slot in 1:ANALOG_STEP_COUNT
            train = [i for i in fold.train if usable[i] && anchors.present[i, slot]]
            length(train) >= 2 * GBM_MIN_WEIGHT || error(
                "fold $(fold.name) has only $(length(train)) direct-GBM training rows at slot $slot",
            )
            direct_gbm_check_anchor(features, anchors, train)
            model = gbm_fit(features[train, :], direct_gbm_target(anchors, train, slot);
                                max_depth=depth, nrounds=rounds, eta=GBM_ETA,
                                min_weight=GBM_MIN_WEIGHT, seed=ANALOG_SEED,
                                feature_names=feature_names)
            on_model === nothing || on_model(fold, slot, model, feature_names)
            increments = gbm_predict(model, query_features)
            for (j, i) in enumerate(query_ok)
                result.raw[i, slot] = direct_gbm_center(anchors, i, increments[j])
            end
        end
        _analog_mark_fallback!(result, anchors, query_fallback)
        append!(result.rows, query)
        _analog_log(@sprintf(
            "    direct GBM d%d r%d fold %s: queries %d (%d served fallback), %.1f s",
            depth, rounds, fold.name, length(query), length(query_fallback), time() - started,
        ))
    end
    return result
end

"""
    analog_run_oracle_stage!(ctx, folds; baselines) -> NamedTuple

Realized-future-driver oracle: the tail is the driver that was actually
observed. It is noncausal and is never a candidate; it measures how much of the
remaining error a perfect driver continuation could remove through the unchanged
core, which is the headroom the plan's per-lead effect sizes are set against.
Every query anchor is scored, because the oracle needs no issue-time analog key.

With `baselines = true` the same realized driver sequence is also integrated
through Burton, Burton-full and O'Brien–McPherron, giving the SINDy-removal
reference of the plan's ablation list.
"""
function analog_run_oracle_stage!(ctx::AnalogContext, folds; baselines::Bool=false)
    anchors = ctx.anchors
    result = analog_empty_result(ctx, ANALOG_ORACLE_ID, "ORACLE", analog_oracle_params())
    n = length(anchors)
    baseline_centers = baselines ?
        (burton=fill(NaN, n, ANALOG_STEP_COUNT), burton_full=fill(NaN, n, ANALOG_STEP_COUNT),
         obrien=fill(NaN, n, ANALOG_STEP_COUNT)) : nothing
    lib, xi = ctx.core.library, ctx.core.coefficients
    lookup = ctx.lookup
    for fold in folds
        started = time()
        query = sort(fold.query)
        Threads.@threads :static for idx in eachindex(query)
            i = query[idx]
            drv = anchors.drv[i]
            issue = anchors.issue[i]
            future = k -> get(lookup, issue + Hour(k - 1), nothing)
            out = Matrix{Float64}(undef, 1, ANALOG_STEP_COUNT)
            scratch = baselines ?
                (burton=Matrix{Float64}(undef, 1, ANALOG_STEP_COUNT),
                 burton_full=Matrix{Float64}(undef, 1, ANALOG_STEP_COUNT),
                 obrien=Matrix{Float64}(undef, 1, ANALOG_STEP_COUNT)) : nothing
            analog_member_step_raw!(out, lib, xi, anchors.anchor_star[i], drv, future,
                                 [analog_realized_driver_member(future)]; baselines=scratch)
            for slot in 1:ANALOG_STEP_COUNT
                result.raw[i, slot] = out[1, slot]
                baselines || continue
                baseline_centers.burton[i, slot] = scratch.burton[1, slot]
                baseline_centers.burton_full[i, slot] = scratch.burton_full[1, slot]
                baseline_centers.obrien[i, slot] = scratch.obrien[1, slot]
            end
        end
        append!(result.rows, query)
        _analog_log(@sprintf("    realized-driver oracle fold %s: queries %d, %.1f s",
                          fold.name, length(query), time() - started))
    end
    return (result=result, baselines=baseline_centers)
end

# ---------------------------------------------------------------------------
# Selection rule (plan section 4)
# ---------------------------------------------------------------------------

"""
    AnalogCandidate

One row of the selection trace: a configuration together with its safeguard
option, the mean pooled out-of-fold RMSE over the selection steps, and the
storm-safety guard verdict.
"""
struct AnalogCandidate
    id::String
    base_id::String
    family::String
    safeguards::Bool
    k::Int
    mean_rmse_nt::Float64
    step_rmse::Vector{Float64}
    guards::NamedTuple
end

"Tie-break key of the selection rule: smaller mean RMSE, then smaller K, then S = on."
_analog_candidate_key(c::AnalogCandidate) = (c.mean_rmse_nt, c.k, c.safeguards ? 0 : 1, c.id)

"""
    analog_candidates(metrics) -> Vector{AnalogCandidate}

Build the selection trace from the pooled metric rows of every candidate
configuration. Only the pooled `all` cell at the selection steps enters the
score; the cell rows enter through the guards.
"""
function analog_candidates(metrics)
    grouped = Dict{String,Vector{Any}}()
    for row in metrics
        push!(get!(grouped, String(row.config), Any[]), row)
    end
    candidates = AnalogCandidate[]
    for (id, rows) in grouped
        family = String(first(rows).family)
        safeguards = Bool(first(rows).safeguards)
        params = JSON3.read(String(first(rows).params_json))
        k = haskey(params, :k) ? Int(params[:k]) : typemax(Int)
        base_id = replace(id, r"_S(on|off)$" => "")
        step_rmse = Float64[]
        for step in ANALOG_SELECTION_STEPS
            hit = [r for r in rows if r.cell == "all" && r.model_step_hours == step]
            length(hit) == 1 || error(
                "configuration $id has $(length(hit)) pooled rows at step $step",
            )
            push!(step_rmse, Float64(first(hit).rmse_nt))
        end
        push!(candidates, AnalogCandidate(id, base_id, family, safeguards, k,
                                       mean(step_rmse), step_rmse, analog_guard_flags(rows)))
    end
    sort!(candidates; by=_analog_candidate_key)
    return candidates
end

"""
    analog_select(candidates; require_guards = true) -> Union{AnalogCandidate,Nothing}

Apply the preregistered selection rule: the guard-passing configuration with the
smallest mean pooled out-of-fold RMSE over model steps {2, 3, 6}, ties broken
toward the smaller ensemble and then toward safeguards on. Returns `nothing`
when no configuration passes the guards, which is a reportable outcome and not a
licence to relax them.

`require_guards = false` exists only so a smoke run on a few thousand anchors
can exercise the composition stages when the storm cells are too small to be
informative. It is never used by the development or confirmatory runs, whose
callers pass the default, and any artifact produced under it is labelled.
"""
function analog_select(candidates; require_guards::Bool=true)
    passing = [c for c in candidates if c.guards.guards_ok && isfinite(c.mean_rmse_nt)]
    if isempty(passing)
        require_guards && return nothing
        finite = [c for c in candidates if isfinite(c.mean_rmse_nt)]
        isempty(finite) && return nothing
        @warn "no configuration passed the storm-safety guards; selecting on RMSE alone " *
              "because the guards were not required (smoke path only)"
        return first(sort(finite; by=_analog_candidate_key))
    end
    return first(sort(passing; by=_analog_candidate_key))
end

"Selection-trace rows in ranked order."
function analog_selection_rows(candidates, selected)
    rows = NamedTuple[]
    for (rank, c) in enumerate(candidates)
        push!(rows, (
            rank=rank, config=c.id, base_config=c.base_id, family=c.family,
            safeguards=c.safeguards, k=(c.k == typemax(Int) ? -1 : c.k),
            mean_rmse_steps_2_3_6_nt=c.mean_rmse_nt,
            rmse_step_2_nt=c.step_rmse[1], rmse_step_3_nt=c.step_rmse[2],
            rmse_step_6_nt=c.step_rmse[3],
            cell_guard_ok=c.guards.cell_guard_ok,
            intense_guard_ok=c.guards.intense_guard_ok,
            guards_ok=c.guards.guards_ok,
            intense_rows=c.guards.intense_rows,
            guard_failures=c.guards.guard_failures,
            selected=(selected !== nothing && c.id == selected.id),
        ))
    end
    return rows
end

# ---------------------------------------------------------------------------
# C1 — lead-aware tail composition
# ---------------------------------------------------------------------------

"""
    analog_lat_weights(ctx, centers, rows) -> (weights, table)

Per model step, the composition weight `w` minimising the pooled out-of-fold
RMSE of `w · tail-center + (1 − w) · frozen-tail inertial-SINDy`. Ties resolve toward the
smaller weight, the more conservative composition. `w = 1` reproduces the tail
center and `w = 0` reproduces frozen inertial-SINDy exactly, which is the executable
identity for this stage.
"""
function analog_lat_weights(ctx::AnalogContext, centers::AbstractMatrix{Float64},
                         rows::AbstractVector{Int})
    anchors = ctx.anchors
    weights = zeros(Float64, ANALOG_STEP_COUNT)
    table = NamedTuple[]
    for (slot, step) in enumerate(ANALOG_MODEL_STEPS)
        best_w = NaN
        best_rmse = Inf
        entry = Dict{String,Float64}()
        n_scored = 0
        for w in ANALOG_LAT_WEIGHT_GRID
            total = 0.0
            n = 0
            for i in rows
                (anchors.present[i, slot] && isfinite(centers[i, slot])) || continue
                blended = w * centers[i, slot] + (1 - w) * anchors.frozen[i, slot]
                total += abs2(anchors.obs[i, slot] - blended)
                n += 1
            end
            rmse = n == 0 ? NaN : sqrt(total / n)
            n_scored = n
            entry[@sprintf("rmse_w%.2f_nt", w)] = rmse
            if isfinite(rmse) && rmse < best_rmse
                best_rmse = rmse
                best_w = w
            end
        end
        isfinite(best_w) || error("no finite lead-aware composition weight at step $step")
        weights[slot] = best_w
        push!(table, (
            model_step_hours=step, n=n_scored, selected_weight=best_w,
            rmse_selected_nt=best_rmse,
            rmse_w0_00_nt=entry["rmse_w0.00_nt"], rmse_w0_25_nt=entry["rmse_w0.25_nt"],
            rmse_w0_50_nt=entry["rmse_w0.50_nt"], rmse_w0_75_nt=entry["rmse_w0.75_nt"],
            rmse_w1_00_nt=entry["rmse_w1.00_nt"],
        ))
    end
    return (weights=weights, table=table)
end

"""
    analog_apply_lat(ctx, centers, weights, rows) -> Matrix{Float64}

Blend a tail center with frozen-tail inertial-SINDy using per-step weights.
"""
function analog_apply_lat(ctx::AnalogContext, centers::AbstractMatrix{Float64},
                       weights::AbstractVector{Float64}, rows::AbstractVector{Int})
    anchors = ctx.anchors
    out = fill(NaN, size(centers))
    for i in rows, slot in 1:ANALOG_STEP_COUNT
        (anchors.present[i, slot] && isfinite(centers[i, slot])) || continue
        w = weights[slot]
        out[i, slot] = w * centers[i, slot] + (1 - w) * anchors.frozen[i, slot]
    end
    return out
end

# ---------------------------------------------------------------------------
# E-layers (plan section 4, E1 and E2)
# ---------------------------------------------------------------------------

"Cap of an E-layer correction at model step h (nT)."
analog_e_cap(h::Integer) = 5.0 + 5.0 * Int(h)

"Names of the innovation error-state inputs, in column order."
const ANALOG_E_FEATURE_NAMES = (
    :innovation_1h, :innovation_2h, :innovation_3h, :innovation_4h, :innovation_5h,
    :innovation_6h, :dst0, :ddst1, :vbs0,
)

"""
    analog_innovation_features(ctx, centers, rows) -> (E, ok)

Innovation error-state inputs at every anchor: the six matured one-step
innovations of the composed center plus the issue-time Dst, its one-hour rate,
and the rectified coupling.

The one-step forecast issued at `t − j` targets `t − j + 1`, so it is matured at
issue `t` for every `j ≥ 1`; the six lags used here are therefore all observable
at issue time and the layer stays causal. An anchor whose six predecessors are
not all scored has no defined input and is returned with `ok = false`, so it
keeps the uncorrected center rather than an imputed one.
"""
function analog_innovation_features(ctx::AnalogContext, centers::AbstractMatrix{Float64},
                                 rows::AbstractVector{Int})
    anchors = ctx.anchors
    n = length(anchors)
    slot1 = ANALOG_STEP_SLOT[1]
    innovation = fill(NaN, n)
    for i in rows
        (anchors.present[i, slot1] && isfinite(centers[i, slot1])) || continue
        innovation[i] = anchors.obs[i, slot1] - centers[i, slot1]
    end
    features = fill(NaN, n, length(ANALOG_E_FEATURE_NAMES))
    ok = falses(n)
    for i in rows
        complete = true
        for lag in 1:ANALOG_E_INNOVATION_LAGS
            j = get(anchors.index, anchors.issue[i] - Hour(lag), 0)
            if j == 0 || !isfinite(innovation[j])
                complete = false
                break
            end
            features[i, lag] = innovation[j]
        end
        complete || continue
        drv = anchors.drv[i]
        features[i, ANALOG_E_INNOVATION_LAGS + 1] = anchors.latest[i]
        features[i, ANALOG_E_INNOVATION_LAGS + 2] = anchors.rate[i]
        features[i, ANALOG_E_INNOVATION_LAGS + 3] = drv.V * max(0.0, -drv.Bz) / 1000.0
        ok[i] = true
    end
    return (features, ok)
end

"Standardised ridge with an intercept; the penalty acts on the standardised slopes only."
struct AnalogRidge
    mean::Vector{Float64}
    sd::Vector{Float64}
    beta::Vector{Float64}
    intercept::Float64
    lambda::Float64
end

function analog_ridge_fit(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, lambda::Real)
    size(X, 1) == length(y) || throw(DimensionMismatch("ridge design and target disagree"))
    size(X, 1) > size(X, 2) || throw(ArgumentError(
        "ridge fit needs more rows than columns, got $(size(X, 1)) × $(size(X, 2))",
    ))
    Float64(lambda) > 0 || throw(ArgumentError("ridge penalty must be positive"))
    design = Float64.(X)
    target = Float64.(y)
    all(isfinite, design) && all(isfinite, target) || throw(ArgumentError(
        "ridge inputs must be finite",
    ))
    μ = vec(mean(design; dims=1))
    σ = max.(vec(std(design; dims=1)), 1e-9)
    Z = (design .- transpose(μ)) ./ transpose(σ)
    ȳ = mean(target)
    beta = (transpose(Z) * Z + Float64(lambda) * I) \ (transpose(Z) * (target .- ȳ))
    return AnalogRidge(μ, σ, beta, ȳ, Float64(lambda))
end

function analog_ridge_predict(model::AnalogRidge, X::AbstractMatrix{<:Real})
    size(X, 2) == length(model.beta) || throw(DimensionMismatch(
        "ridge model expects $(length(model.beta)) columns, received $(size(X, 2))",
    ))
    return ((Float64.(X) .- transpose(model.mean)) ./ transpose(model.sd)) * model.beta .+
           model.intercept
end

"""
    analog_e_design(kind, efeatures, adc, centers, rows, slot) -> Matrix

Design matrix of an E-layer. E1 uses the innovation error state alone; E2 adds
the analog feature vector and the center being corrected, which makes it the
information-equivalent competitor to the direct machine-learning comparator.
"""
function analog_e_design(kind::Symbol, efeatures::AbstractMatrix{Float64},
                      adc::AbstractMatrix{Float64}, centers::AbstractMatrix{Float64},
                      rows::AbstractVector{Int}, slot::Int)
    kind === :E1 && return efeatures[rows, :]
    # The issue-time state columns of the innovation block (`dst0`, `ddst1`,
    # `vbs0`) are the identical quantities the analog feature vector already
    # carries, so E2 takes the innovations, the analog vector and the center
    # instead of repeating three columns verbatim.
    kind === :E2 && return hcat(efeatures[rows, 1:ANALOG_E_INNOVATION_LAGS], adc[rows, :],
                                [centers[i, slot] for i in rows])
    throw(ArgumentError("unknown E-layer $kind"))
end

"Column names of an E-layer design matrix, for the boosted variant."
function analog_e_feature_names(kind::Symbol)
    kind === :E1 && return String.(collect(ANALOG_E_FEATURE_NAMES))
    kind === :E2 && return vcat(
        String.(collect(ANALOG_E_FEATURE_NAMES)[1:ANALOG_E_INNOVATION_LAGS]),
        String.(collect(ANALOG_FEATURE_NAMES)), ["center_dst_nt"],
    )
    throw(ArgumentError("unknown E-layer $kind"))
end

"""
    analog_e_layer_predictions(ctx, e_folds, base_centers, efeatures, eok, kind, param) -> Matrix

Out-of-fold capped E-layer corrections. Each fold fits on rows that precede the
fold's queries by at least the 168 h target embargo, so the correction applied
to a query has never seen that query's own innovation.

`on_model`, when given, is called as `on_model(fold, slot, model, cap)` after
each fit. It only observes: the returned corrections are computed from the same
fitted object either way, so a caller that persists the model for deployment
cannot move a score.
"""
function analog_e_layer_predictions(ctx::AnalogContext, e_folds, base_centers::AbstractMatrix{Float64},
                                 efeatures::AbstractMatrix{Float64}, eok::AbstractVector{Bool},
                                 kind::Symbol, param; on_model=nothing)
    anchors = ctx.anchors
    delta = fill(NaN, size(base_centers))
    minimum_rows = 10 * length(analog_e_feature_names(kind))
    for fold in e_folds, slot in 1:ANALOG_STEP_COUNT
        step = ANALOG_MODEL_STEPS[slot]
        usable(i) = eok[i] && anchors.present[i, slot] && isfinite(base_centers[i, slot])
        train = [i for i in fold.train if usable(i)]
        query = [i for i in fold.query if usable(i)]
        (length(train) >= minimum_rows && !isempty(query)) || continue
        design = analog_e_design(kind, efeatures, ctx.X, base_centers, train, slot)
        target = [anchors.obs[i, slot] - base_centers[i, slot] for i in train]
        query_design = analog_e_design(kind, efeatures, ctx.X, base_centers, query, slot)
        cap = analog_e_cap(step)
        predictions = if kind === :E1
            model = analog_ridge_fit(design, target, param)
            on_model === nothing || on_model(fold, slot, model, cap)
            analog_ridge_predict(model, query_design)
        else
            model = gbm_fit(design, target; max_depth=ANALOG_E2_DEPTH, nrounds=ANALOG_E2_ROUNDS,
                                eta=GBM_ETA, min_weight=GBM_MIN_WEIGHT, seed=ANALOG_SEED,
                                feature_names=analog_e_feature_names(kind))
            on_model === nothing || on_model(fold, slot, model, cap)
            gbm_predict(model, query_design)
        end
        for (j, i) in enumerate(query)
            delta[i, slot] = clamp(predictions[j], -cap, cap)
        end
    end
    return delta
end

"""
    analog_run_e_layer(ctx, e_folds, base_centers, rows, kind; params, label) -> NamedTuple

Fit, score and adjudicate one optional error layer. A step keeps the layer only
when it improves the pooled out-of-fold RMSE on the rows the layer can actually
correct *and* the storm-safety guards still hold at that step; otherwise the
step keeps the identity. The audit rows record every parameter tried, the
evaluable sample size, both RMSEs and the verdict, so a rejected layer is as
visible as an accepted one.
"""
function analog_run_e_layer(ctx::AnalogContext, e_folds, base_centers::AbstractMatrix{Float64},
                         rows::AbstractVector{Int}, kind::Symbol; params,
                         label::AbstractString=String(kind),
                         eval_rows::AbstractVector{Int}=rows)
    anchors = ctx.anchors
    # Innovations are built over every scored row so that a query at the start of
    # a fold can still see its six predecessors; the layer is only evaluated and
    # guarded on the rows a fold can actually correct.
    efeatures, eok = analog_innovation_features(ctx, base_centers, rows)
    deltas = Dict{Any,Matrix{Float64}}()
    for param in params
        deltas[param] = analog_e_layer_predictions(ctx, e_folds, base_centers, efeatures, eok,
                                                kind, param)
    end
    audit = NamedTuple[]
    corrected = copy(base_centers)
    accepted = falses(ANALOG_STEP_COUNT)
    chosen = Vector{Any}(undef, ANALOG_STEP_COUNT)
    fill!(chosen, nothing)
    for (slot, step) in enumerate(ANALOG_MODEL_STEPS)
        best_param = nothing
        best_rmse = Inf
        summary = NamedTuple[]
        for param in params
            delta = deltas[param]
            evaluable = [i for i in eval_rows if isfinite(delta[i, slot])]
            base_rmse = analog_rmse([anchors.obs[i, slot] - base_centers[i, slot] for i in evaluable])
            layer_rmse = analog_rmse([
                anchors.obs[i, slot] - (base_centers[i, slot] + delta[i, slot])
                for i in evaluable
            ])
            push!(summary, (param=param, n=length(evaluable), base=base_rmse, layer=layer_rmse))
            if isfinite(layer_rmse) && layer_rmse < best_rmse
                best_rmse = layer_rmse
                best_param = param
            end
        end
        guards = (guards_ok=false, cell_guard_ok=false, intense_guard_ok=false,
                  intense_rows=0, guard_failures="not_evaluated")
        take = false
        if best_param !== nothing
            trial = copy(base_centers)
            delta = deltas[best_param]
            for i in eval_rows
                isfinite(delta[i, slot]) || continue
                trial[i, slot] = base_centers[i, slot] + delta[i, slot]
            end
            step_metrics = [r for r in analog_metric_rows("$(label)_step$(step)", trial, anchors,
                                                       ctx.masks, eval_rows,
                                                       falses(length(anchors)))
                            if r.model_step_hours == step]
            guards = analog_guard_flags(step_metrics)
            best_summary = first(s for s in summary if s.param == best_param)
            take = guards.guards_ok && isfinite(best_summary.base) &&
                   best_summary.base - best_summary.layer > 0
            if take
                for i in eval_rows
                    isfinite(delta[i, slot]) || continue
                    corrected[i, slot] = trial[i, slot]
                end
                accepted[slot] = true
                chosen[slot] = best_param
            end
        end
        for s in summary
            # The guards are evaluated on the best parameter alone, so only that
            # row may carry a guard verdict; the others report that no guard was
            # ever computed for them instead of inheriting the winner's verdict.
            selected_param = best_param !== nothing && s.param == best_param
            push!(audit, (
                layer=String(label), model_step_hours=step, param=string(s.param),
                n_evaluable=s.n, rmse_base_nt=s.base, rmse_layer_nt=s.layer,
                gain_nt=s.base - s.layer,
                selected_param=selected_param,
                accepted=(take && selected_param),
                guards_ok=(selected_param ? guards.guards_ok : false),
                guard_failures=(selected_param ? guards.guard_failures : "not_evaluated"),
            ))
        end
    end
    return (centers=corrected, audit=audit, accepted=accepted, params=chosen)
end

"""
    analog_e_layer_folds(ctx, folds) -> Vector

Expanding-window folds for the E-layers over the inner blocks: block `b` is
corrected by a layer fitted on the pooled out-of-fold rows of the earlier
blocks, embargoed by 168 h. The first block therefore has no layer and is
excluded from the layer's evaluation, which is reported rather than hidden.
"""
function analog_e_layer_folds(ctx::AnalogContext, folds)
    out = NamedTuple[]
    for (position, fold) in enumerate(folds)
        position == 1 && continue
        cutoff = minimum(ctx.anchors.issue[fold.query]) - Hour(ANALOG_BASE_EMBARGO_HOURS)
        train = Int[]
        for earlier in folds[1:position - 1]
            append!(train, [i for i in earlier.query
                            if ctx.anchors.issue[i] + Hour(ANALOG_MAX_STEP) <= cutoff])
        end
        isempty(train) && continue
        push!(out, (name=fold.name, train=train, query=fold.query))
    end
    return out
end

# ---------------------------------------------------------------------------
# Manifest and report helpers
# ---------------------------------------------------------------------------

"Accumulating manifest of inputs, outputs, configurations and stage timings."
mutable struct AnalogManifest
    rows::Vector{NamedTuple}
end
AnalogManifest() = AnalogManifest(NamedTuple[])

function analog_manifest!(manifest::AnalogManifest, kind::AbstractString, name::AbstractString,
                       count::Real=NaN, value::AbstractString="")
    push!(manifest.rows, (entry_type=String(kind), name=String(name),
                          count=Float64(count), value=String(value)))
    return manifest
end

"""
    analog_write_manifest(path, manifest, ctx) -> path

Write the manifest after stamping the SHA-256 of every artifact in the run
directory, so the recorded state matches the files on disk at the moment the run
finished.
"""
function analog_write_manifest(path::AbstractString, manifest::AnalogManifest, ctx::AnalogContext)
    analog_manifest!(manifest, "input_sha256", "analog_base_table.csv", NaN, ctx.base_table_sha)
    analog_manifest!(manifest, "input_sha256", "analog_hourly_frame.csv", NaN, ctx.frame_sha)
    analog_manifest!(manifest, "input_sha256", "analog_source_code", NaN, ctx.code_sha)
    analog_manifest!(manifest, "environment", "cells", NaN, ctx.cells_source)
    analog_manifest!(manifest, "environment", "julia_threads", Threads.nthreads(), "")
    analog_manifest!(manifest, "identity", "inertial_sindy_correction_max_abs_nt",
                  ctx.correction_residual_nt, "")
    for file in sort(readdir(ctx.plan.outdir))
        full = joinpath(ctx.plan.outdir, file)
        (isfile(full) && full != abspath(path)) || continue
        analog_manifest!(manifest, "output_sha256", file, filesize(full), _analog_sha256_file(full))
    end
    CSV.write(path, DataFrame(manifest.rows))
    return path
end

"""
    analog_markdown_table(io, header, rows)

Emit one GitHub-flavoured markdown table. Reports built from these tables carry
numbers only; the interpretation belongs in the planner's decision record, not
in a generated file.
"""
function analog_markdown_table(io::IO, header::AbstractVector{<:AbstractString}, rows)
    println(io, "| ", join(header, " | "), " |")
    println(io, "|", repeat("---|", length(header)))
    for row in rows
        println(io, "| ", join(row, " | "), " |")
    end
    println(io)
end

_analog_fmt(x::Real; digits::Int=3) = isfinite(x) ? string(round(Float64(x); digits=digits)) : "NaN"
_analog_fmt(x) = string(x)

"""
    analog_write_final_config!(ctx, id, family, params, centers, rows, fallback, safeguards,
                            seconds) -> Vector{NamedTuple}

Persist a *composed* center — a lead-aware blend or an error-layer output —
whose safeguard option is already fixed by the selected configuration. The two
safeguard columns of the row file therefore carry the same value, and the
metrics record which option was in force. Unlike a stage result, the centers are
written exactly as supplied: the served-fallback substitution has already been
made by the composition stage.
"""
function analog_write_final_config!(ctx::AnalogContext, id::AbstractString, family::AbstractString,
                                 params::AbstractDict, centers::AbstractMatrix{Float64},
                                 rows::AbstractVector{Int}, fallback::AbstractVector{Bool},
                                 safeguards::Bool, seconds::Real)
    mkpath(ctx.plan.outdir)
    anchors = ctx.anchors
    out = NamedTuple[]
    for i in rows, slot in 1:ANALOG_STEP_COUNT
        anchors.present[i, slot] || continue
        push!(out, (
            issue_time_utc=string(anchors.issue[i]),
            cv_block=anchors.cv_block[i],
            model_step_hours=ANALOG_MODEL_STEPS[slot],
            fallback=fallback[i],
            raw_dst_nt=centers[i, slot],
            center_s_on_dst_nt=centers[i, slot],
            center_s_off_dst_nt=centers[i, slot],
        ))
    end
    CSV.write(analog_oof_path(ctx.plan, id), DataFrame(out))
    metrics = [
        merge(row, (family=String(family), safeguards=safeguards,
                    params_json=JSON3.write(params), seconds=Float64(seconds)))
        for row in analog_metric_rows(id, centers, anchors, ctx.masks, rows, fallback)
    ]
    CSV.write(analog_summary_path(ctx.plan, id), DataFrame(metrics))
    write(analog_signature_path(ctx.plan, id), analog_config_signature(ctx, id, params))
    return metrics
end

"Convert the persisted metric table of a configuration back into metric rows."
analog_metric_namedtuples(table::DataFrame) = [NamedTuple(row) for row in eachrow(table)]
