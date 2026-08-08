# assimilation_fair_test.jl — FAIR test of online EKF coefficient adaptation, fixing the two confounds
# that biased the earlier check (assimilation_forecast_value.jl) toward "adaptation hurts":
#   (1) it used the minimal 3-term library (a poor base model)         -> here: the FULL operational library;
#   (2) it initialised coefficients least-squares-fit on the test storm -> here: the DEPLOYED discovery
#       coefficients, fit on the TRAIN-split storms (solar cycles 20-23), hence genuinely out-of-sample for
#       the cycle-25 (2023-2025) storms tested here.
#
# Data note: the storm catalog and the archived omni_extracted.csv come from different OMNI vintages, so the
# catalog's ROW indices and recorded depths do not match this file for recent storms. We therefore extract
# each storm window by TIME from a single self-consistent df (parse+clean of one file) and verify each window
# is internally a real deep storm, rather than trusting cross-vintage row indices. The catalog is used only
# to locate storm centres (min_dst_time) and to confirm the split is "test".
#
# Historical development question: does adapting a small physically-motivated coefficient subset of the
# current V2.1 20/11 core online (decay rate and/or the
# dominant coupling scale) improve the ONE-STEP-AHEAD Dst* forecast on held-out storms, versus holding all
# coefficients fixed at the deployed values? We score the prediction made BEFORE each observation. To give
# adaptation its best fair chance, we sweep the coefficient process noise q_coeff and report the BEST adapted
# result against fixed (so a negative result is conservative). This is a
# one-step extension diagnostic and does not reproduce the deployed V2.1 tail or
# 26-feature calibration. Its bidirectional gap filling is retrospective, so this
# script cannot establish live deployability. The later powered and served-tail
# replays supersede its promotion interpretation and reject EKF promotion.

using SolarSINDy, DataFrames, Statistics, LinearAlgebra, Printf, Dates

const PKG  = pkgdir(SolarSINDy)
const PROJ = normpath(joinpath(PKG, ".."))
const EXTRACTED = joinpath(PROJ, "paper_v2_monitor", "data", "omni_extracted.csv")
const CATALOG   = joinpath(PKG, "data", "storm_catalog.csv")

# forward+backward fill of NaN in a column, then median fallback if all-NaN
function fillnan!(x::Vector{Float64})
    n = length(x); last = NaN
    for i in 1:n; isfinite(x[i]) ? (last = x[i]) : (isfinite(last) && (x[i] = last)); end
    last = NaN
    for i in n:-1:1; isfinite(x[i]) ? (last = x[i]) : (isfinite(last) && (x[i] = last)); end
    any(!isfinite, x) && (x .= ifelse.(isfinite.(x), x, 0.0))
    return x
end

function main()
    df = parse_omni2(EXTRACTED; year_start=2022, year_end=2025)
    clean_omni_data!(df)
    catalog = load_storm_catalog(CATALOG)
    # held-out storms: split=="test" AND cycle 25 (2023+), located by time, deepest first
    cand = filter(e -> e.split == "test" && year(e.min_dst_time) >= 2023, catalog)
    sort!(cand, by = e -> e.min_dst)
    deepest = cand[1:min(6, length(cand))]

    core = load_operational_core(:v2)
    lib = core.library; term_names = get_term_names(lib); ξ0 = copy(core.coefficients)
    length(term_names) == 20 && count(!=(0.0), ξ0) == 11 &&
        !("n*V^2" in term_names) ||
        error("fair EKF diagnostic did not load the current 20/11 V2.1 core")
    i_decay = findfirst(==("Dst_star"), term_names)
    act = [i for i in eachindex(ξ0) if ξ0[i] != 0 && i != i_decay]
    i_inj = act[argmax(abs.(ξ0[act]))]
    @printf("full library: %d terms, %d active; adapt decay='%s', injection='%s'\n",
            length(lib), count(!=(0), ξ0), term_names[i_decay], term_names[i_inj])

    # time-window extraction from the self-consistent df (NaN-filled drivers + obs)
    function window(e)
        m = (df.datetime .>= e.min_dst_time - Hour(36)) .& (df.datetime .<= e.min_dst_time + Hour(72))
        sub = df[m, :]
        cols = Dict(c => fillnan!(Float64.(coalesce.(sub[!, c], NaN))) for c in
                    [:V, :Bz, :By, :n, :Pdyn, :Dst, :Dst_star])
        return cols
    end

    function onestep_rmse(w, adapt_idx; q_coeff = 1e-6)
        obs = w[:Dst_star]; npts = length(obs)
        drivers = [(V=w[:V][k], Bz=w[:Bz][k], By=w[:By][k], n=w[:n][k], Pdyn=w[:Pdyn][k]) for k in 1:npts]
        f = init_assimilation(lib, ξ0, adapt_idx, obs[1]; q_coeff = q_coeff)
        err = Float64[]
        for k in 1:npts-1
            assimilation_predict!(f, drivers[k])
            push!(err, current_dst(f) - obs[k+1])
            assimilation_update!(f, obs[k+1])
        end
        sqrt(mean(err .^ 2))
    end

    qsweep = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    @printf("\n%-22s %7s | %-7s | %-22s | %-22s\n",
            "storm (min_dst_time)","winMin","fixed","adapt decay (bestq:rmse)","adapt dec+inj (bestq:rmse)")
    tf = Float64[]; td = Float64[]; tdi = Float64[]
    dmat = Vector{Vector{Float64}}()   # per-storm RMSE across qsweep (adapt-decay) for the single-q analysis
    for e in deepest
        w = window(e); wmin = minimum(w[:Dst_star])
        wmin > -80 && continue   # internal deep-storm check (vintage-independent)
        rf = onestep_rmse(w, Int[])
        dvals = [onestep_rmse(w, [i_decay]; q_coeff=q) for q in qsweep]
        ivals = [onestep_rmse(w, [i_decay, i_inj]; q_coeff=q) for q in qsweep]
        bd, bdq = minimum(dvals), qsweep[argmin(dvals)]
        bi, biq = minimum(ivals), qsweep[argmin(ivals)]
        push!(tf, rf); push!(td, bd); push!(tdi, bi); push!(dmat, dvals)
        @printf("%-22s %7.0f | %7.2f | q=%.0e: %7.2f     | q=%.0e: %7.2f\n",
                string(e.min_dst_time), wmin, rf, bdq, bd, biq, bi)
    end
    isempty(tf) && (println("no qualifying deep windows found"); return)
    @printf("\nmean 1-step Dst* RMSE [nT] over %d held-out cycle-25 storms:\n", length(tf))
    @printf("  fixed = %.2f   adapt-decay(per-storm best q) = %.2f   adapt-decay+inj(per-storm best q) = %.2f\n",
            mean(tf), mean(td), mean(tdi))

    # Shared-q check: useful as a raw-core sensitivity screen, but still selected and
    # scored on the same retrospective storm set and therefore not a deployment gate.
    @printf("\nadapt-decay mean RMSE at a SINGLE shared q (raw-core diagnostic; vs fixed = %.2f):\n", mean(tf))
    qmeans = [mean(getindex.(dmat, j)) for j in eachindex(qsweep)]
    for (j, q) in enumerate(qsweep)
        @printf("  q=%.0e : %.2f %s\n", q, qmeans[j], qmeans[j] < mean(tf) - 0.05 ? "(beats fixed)" : "")
    end
    bestq_single = minimum(qmeans)

    if bestq_single < mean(tf) - 0.05
        @printf("\n→ RAW-CORE ONE-STEP SIGNAL: a shared q (%.0e) lowers mean 1-step RMSE %.2f→%.2f nT on these held-out storms\n",
                qsweep[argmin(qmeans)], mean(tf), bestq_single)
        @printf("  with the revised 20/11 core. This is not deployable evidence: q was selected on the scored set,\n")
        @printf("  preprocessing is retrospective, and the served V2.1 tail is absent. The later powered and exact\n")
        @printf("  served-tail EKF replays answer the promotion question and reject EKF deployment.\n")
    else
        @printf("\n→ Per-storm best q helps, but NO single global q robustly beats fixed: the gain needs per-storm\n")
        @printf("  tuning. This raw-core diagnostic provides no promotion evidence; later V2.1 replays also reject EKF deployment.\n")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
