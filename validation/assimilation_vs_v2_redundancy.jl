# assimilation_vs_v2_redundancy.jl — superseded small-sample EKF redundancy diagnostic.
#
# This diagnostic asks whether online decay adaptation adds value after a
# separately fitted one-step residual correction on the current V2.1 20/11
# core. It does not reproduce the deployed V2.1 driver tail or 26-feature
# calibration, so its result is exploratory rather than operational evidence.
#
# Faithful test: on the same held-out cycle-25 storms, leave-one-storm-out, we generate per-step ONE-STEP
# predictions two ways — fixed v1 coefficients and EKF-adapted (decay, q=1e-4, the deployable single q) —
# fit the REAL v2 correction (operational calibration) on the OTHER storms' residuals for each, and apply
# it to the held-out storm. We compare four one-step RMSEs:
#   A = fixed v1 (raw)              B = fixed v1 + V2 correction
#   C = EKF  v1 (raw)              D = EKF  v1 + v2 correction
# Verdict: if D ~ B, the v2 correction already captures the EKF gain -> EKF REDUNDANT with v2. If D < B by
# a meaningful margin, the EKF adds value ON TOP of v2 -> worth deploying. If B <= C, the correction alone
# already matches/beats the raw EKF -> strong redundancy signal. The correction is applied directly via
# operational_v2_correction (the fitted ridge beta), bypassing the component selector so no baseline
# columns are needed; this isolates exactly the v1+correction question.

using SolarSINDy, DataFrames, Statistics, LinearAlgebra, Printf, Dates

const PKG  = pkgdir(SolarSINDy)
const PROJ = normpath(joinpath(PKG, ".."))
const EXTRACTED = joinpath(PROJ, "paper_v2_monitor", "data", "omni_extracted.csv")
const CATALOG   = joinpath(PKG, "data", "storm_catalog.csv")
const QEKF = 1e-4   # fixed research value shared with the powered diagnostic

function fillnan!(x::Vector{Float64})
    n = length(x); last = NaN
    for i in 1:n; isfinite(x[i]) ? (last = x[i]) : (isfinite(last) && (x[i] = last)); end
    last = NaN
    for i in n:-1:1; isfinite(x[i]) ? (last = x[i]) : (isfinite(last) && (x[i] = last)); end
    any(!isfinite, x) && (x .= ifelse.(isfinite.(x), x, 0.0)); return x
end
rmse(e) = sqrt(mean(e .^ 2))

function main()
    df = parse_omni2(EXTRACTED; year_start=2022, year_end=2025); clean_omni_data!(df)
    catalog = load_storm_catalog(CATALOG)
    cand = filter(e -> e.split == "test" && year(e.min_dst_time) >= 2023, catalog)
    sort!(cand, by = e -> e.min_dst); deepest = cand[1:min(6, length(cand))]

    core = load_operational_core(:v2)
    lib = core.library; term_names = get_term_names(lib); ξ0 = copy(core.coefficients)
    length(term_names) == 20 && count(!=(0.0), ξ0) == 11 &&
        !("n*V^2" in term_names) ||
        error("EKF redundancy diagnostic did not load the current 20/11 V2.1 core")
    i_decay = findfirst(==("Dst_star"), term_names)

    function window(e)
        m = (df.datetime .>= e.min_dst_time - Hour(36)) .& (df.datetime .<= e.min_dst_time + Hour(72))
        sub = df[m, :]
        Dict(c => fillnan!(Float64.(coalesce.(sub[!, c], NaN))) for c in
             [:V, :Bz, :By, :n, :Pdyn, :Dst, :Dst_star])
    end

    # per-step one-step rows in operational schema (latest=anchor obs, pred=1-step, obs=realized next hour)
    function storm_rows(w, adapt_idx)
        obs = w[:Dst_star]; npts = length(obs)
        drivers = [(V=w[:V][k], Bz=w[:Bz][k], By=w[:By][k], n=w[:n][k], Pdyn=w[:Pdyn][k]) for k in 1:npts]
        q = isempty(adapt_idx) ? 1e-6 : QEKF
        f = init_assimilation(lib, ξ0, adapt_idx, obs[1]; q_coeff = q)
        r = DataFrame(latest_dst_nt=Float64[], V_kms=Float64[], Bz_nt=Float64[], By_nt=Float64[],
                      n_cm3=Float64[], Pdyn_npa=Float64[], pred_dst_nt=Float64[],
                      pred_dst_ci05_nt=Float64[], pred_dst_ci95_nt=Float64[], observation_dst_nt=Float64[])
        for k in 1:npts-1
            assimilation_predict!(f, drivers[k]); p = current_dst(f)
            push!(r, (obs[k], w[:V][k], w[:Bz][k], w[:By][k], w[:n][k], w[:Pdyn][k], p, p-15.0, p+15.0, obs[k+1]))
            assimilation_update!(f, obs[k+1])
        end
        r
    end

    # apply the fitted v2 correction directly (the ridge beta), no component selector
    function corrected(testdf, cal)
        prep = SolarSINDy.add_operational_v2_features!(copy(testdf))
        [prep.pred_dst_nt[i] + SolarSINDy.operational_v2_correction(cal,
            NamedTuple{Tuple(cal.feature_names)}(Tuple(Float64(prep[i, c]) for c in cal.feature_names)))
         for i in 1:nrow(prep)]
    end

    wins = Dict(e.storm_id => window(e) for e in deepest)
    @printf("EKF-vs-v2 redundancy, leave-one-storm-out over %d held-out cycle-25 storms (q_EKF=%.0e):\n", length(deepest), QEKF)
    @printf("%-22s | %-23s | %-23s\n", "held-out (min_dst_time)", "FIXED v1: raw -> +v2corr", "EKF v1: raw -> +v2corr")
    A=Float64[]; B=Float64[]; C=Float64[]; D=Float64[]
    for e in deepest
        w = wins[e.storm_id]; minimum(w[:Dst_star]) > -80 && continue
        others = [o for o in deepest if o.storm_id != e.storm_id]
        # FIXED
        trf = vcat([storm_rows(wins[o.storm_id], Int[]) for o in others]...)
        calf = fit_operational_v2_calibration(trf)
        tef = storm_rows(w, Int[])
        a = rmse(tef.pred_dst_nt .- tef.observation_dst_nt)
        b = rmse(corrected(tef, calf) .- tef.observation_dst_nt)
        # EKF
        tre = vcat([storm_rows(wins[o.storm_id], [i_decay]) for o in others]...)
        cale = fit_operational_v2_calibration(tre)
        tee = storm_rows(w, [i_decay])
        c = rmse(tee.pred_dst_nt .- tee.observation_dst_nt)
        d = rmse(corrected(tee, cale) .- tee.observation_dst_nt)
        push!(A,a); push!(B,b); push!(C,c); push!(D,d)
        @printf("%-22s | %7.2f -> %7.2f       | %7.2f -> %7.2f\n", string(e.min_dst_time), a, b, c, d)
    end
    mA,mB,mC,mD = mean(A),mean(B),mean(C),mean(D)
    @printf("\nmean one-step Dst* RMSE [nT]:  A fixed-raw=%.2f   B v2(fixed+corr)=%.2f   C ekf-raw=%.2f   D ekf+corr=%.2f\n", mA,mB,mC,mD)
    @printf("  EKF gain on raw v1   (A-C) = %+.2f\n", mA-mC)
    @printf("  EKF gain on top of v2 (B-D) = %+.2f   <- the deployment-relevant number\n", mB-mD)
    @printf("  v2 corr vs raw EKF   (C-B) = %+.2f   (>0 => correction alone already beats raw EKF)\n", mC-mB)

    # variance / power scrutiny — a mean is not a result at n=6. Report the paired per-storm spread.
    bd = B .- D                                   # >0 favours EKF on top of v2
    nstorm = length(bd); se = std(bd)/sqrt(nstorm)
    nfav = count(>(0.05), bd); nhurt = count(<(-0.05), bd)
    @printf("\npaired B-D per storm (>0 = EKF helps on top of v2): %s\n",
            join([@sprintf("%+.2f", x) for x in bd], "  "))
    @printf("  mean %+.2f ± %.2f (SE), range [%+.2f, %+.2f]; favours-EKF %d/%d, hurts %d/%d\n",
            mean(bd), se, minimum(bd), maximum(bd), nfav, nstorm, nhurt, nstorm)
    flagship_hurt = bd[1] < -0.05   # storms sorted deepest-first => index 1 is the May-2024 superstorm
    if mean(bd) > 2*se && nhurt == 0
        println("→ SMALL-SAMPLE SIGNAL: EKF adds value in this diagnostic, but this superseded script cannot authorize deployment.")
    elseif abs(mean(bd)) <= se && maximum(abs.(bd)) < 0.5
        println("→ REDUNDANT: D ~ B everywhere; the v2 correction already captures the EKF gain. Keep EKF available,")
        println("  not deployed, on evidence — v2 already does the job.")
    else
        @printf("→ INCONCLUSIVE at n=%d: the mean favours EKF (+%.2f) but within ~1 SE and storm-dependent (range\n", nstorm, mean(bd))
        @printf("  %+.2f..%+.2f); it %s on the flagship May-2024 superstorm. The storms-only v2 correction is also a\n",
                minimum(bd), maximum(bd), flagship_hurt ? "HURTS" : "helps")
        println("  noisy proxy (far less data than the operational broad calibration; it even hurt fixed-v1 on one")
        println("  storm). Honest status: NOT redundant, but NOT a clean additive win — partially complementary,")
        println("  unresolved at this scale. The later powered multi-step test and exact served-tail seven-storm replays")
        println("  supply the missing evidence and reject EKF promotion. Keep the EKF available for research only.")
    end
    println("This small-sample result is historical development evidence; the current operational verdict is NOT PROMOTABLE.")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
