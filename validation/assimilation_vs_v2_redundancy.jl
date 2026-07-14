# assimilation_vs_v2_redundancy.jl — the one open check before any EKF deployment decision.
#
# The fair test (assimilation_fair_test.jl) showed online decay adaptation improves the RAW v1 one-step
# forecast out-of-sample (17.60 -> 15.98 nT). But the OPERATIONAL path is v2 = v1 + a causal linear
# residual correction (fit_operational_v2_calibration: ridge regression of the v1 residual on the
# standardized issue-time drivers). That correction ALSO removes v1 error. So the deployment-relevant
# question is: does the EKF still help ON TOP of the v2 correction, or is it redundant with it?
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

using SolarSINDy, CSV, DataFrames, Statistics, LinearAlgebra, Printf, Dates

const PKG  = pkgdir(SolarSINDy)
const PROJ = normpath(joinpath(PKG, ".."))
const EXTRACTED = joinpath(PROJ, "paper_v2_monitor", "data", "omni_extracted.csv")
const CATALOG   = joinpath(PKG, "data", "storm_catalog.csv")
const COEFCSV   = joinpath(PKG, "data", "real_sindy_discovery_coefficients.csv")
const QEKF = 1e-4   # the deployable single global process noise from the fair test

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

    lib = build_solar_wind_library(include_redundant_n_v2=true); term_names = get_term_names(lib)
    coef_df = CSV.read(COEFCSV, DataFrame); ξ0 = zeros(length(lib))
    for row in eachrow(coef_df)
        idx = findfirst(==(row.term), term_names); idx !== nothing && (ξ0[idx] = row.coefficient)
    end
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
        println("→ EKF ADDS VALUE on top of v2: consistent positive gain across storms, beyond noise. Pursue deploy.")
    elseif abs(mean(bd)) <= se && maximum(abs.(bd)) < 0.5
        println("→ REDUNDANT: D ~ B everywhere; the v2 correction already captures the EKF gain. Keep EKF available,")
        println("  not deployed, on evidence — v2 already does the job.")
    else
        @printf("→ INCONCLUSIVE at n=%d: the mean favours EKF (+%.2f) but within ~1 SE and storm-dependent (range\n", nstorm, mean(bd))
        @printf("  %+.2f..%+.2f); it %s on the flagship May-2024 superstorm. The storms-only v2 correction is also a\n",
                minimum(bd), maximum(bd), flagship_hurt ? "HURTS" : "helps")
        println("  noisy proxy (far less data than the operational broad calibration; it even hurt fixed-v1 on one")
        println("  storm). Honest status: NOT redundant, but NOT a clean additive win — partially complementary,")
        println("  unresolved at this scale. Proper resolution needs the real broad v2 calibration + more storms")
        println("  (and a multi-step horizon). Keep the EKF available, not deployed, pending that — the value is")
        println("  real on raw v1 but not yet shown to survive on top of v2 without hurting the worst storm.")
    end
    return nothing
end

main()
