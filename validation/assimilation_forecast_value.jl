# assimilation_forecast_value.jl — does online EKF coefficient adaptation improve the OPERATIONAL
# (short-horizon) Dst forecast? Decisive, reproducible check on the held-out May 2024 storm.
#
# Both filters assimilate Dst* every step; only `adapt_idx` differs, so the comparison isolates
# the value of adapting the SINDy coefficients online. We score the one-step-ahead prediction
# (predict BEFORE seeing the observation) — the operational quantity.
#
# Finding (run below): in THIS setup adaptation does not help — 1-step RMSE ~23.3 nT fixed vs ~27.8
# (adapt injection) / ~32.8 (adapt both).
#
# ⚠ CAVEAT (added on reflection): this experiment is CONFOUNDED and BIASED toward "adaptation hurts":
#   (1) it uses the minimal 3-term library (a poor base model), not the operational full library;
#   (2) the initial coefficients are least-squares-fit on the May 2024 storm ITSELF, so the fixed
#       baseline is already storm-optimal and adaptation can only drift away from it.
# So this does NOT cleanly establish the operational value of the EKF. A FAIR test must use the full
# library and OUT-OF-SAMPLE initial coefficients (fit on a prior period), then check whether adaptation
# helps track the held-out storm — that requires OMNI (V,Bz,By,n) which this CSV lacks. Until then, the
# honest position is: keep the EKF available (predict/update verified), do not deploy by DEFAULT
# (the v2 correction layer already adapts to recent residuals), and treat its operational value as
# OPEN pending the fair test — not as "shown to degrade the forecast."

using SolarSINDy, Printf, CSV, DataFrames, Statistics, LinearAlgebra

const DATA = joinpath(@__DIR__, "..", "data", "may2024_reconstruction.csv")

function main()
    df = CSV.read(DATA, DataFrame)
    V = Float64.(df.V); Bz = Float64.(df.Bz); obs = Float64.(df.Dst_obs)
    Bs = max.(-Bz, 0.0); n = length(obs)
    dDst = diff(obs)
    Θ = hcat(ones(n - 1), obs[1:end-1], (V .* Bs)[1:end-1])
    ξ0 = Θ \ dDst                                   # in-sample minimal-library start coefficients
    lib = build_minimal_library()                   # ["1", "Dst_star", "V*Bs"]
    drivers = [(V=V[k], Bz=Bz[k], By=0.0, n=5.0, Pdyn=2.0) for k in 1:n]

    function onestep_rmse(adapt_idx)
        f = init_assimilation(lib, ξ0, adapt_idx, obs[1])
        err = Float64[]
        for k in 1:n-1
            assimilation_predict!(f, drivers[k])
            push!(err, current_dst(f) - obs[k+1])   # 1-step prediction vs next observation
            assimilation_update!(f, obs[k+1])
        end
        sqrt(mean(err .^ 2))
    end

    rf  = onestep_rmse(Int[])     # fixed coefficients
    ri  = onestep_rmse([3])       # adapt V*Bs injection
    rb  = onestep_rmse([2, 3])    # adapt decay + injection
    @printf("May 2024 one-step Dst forecast RMSE [nT]: fixed=%.2f  adapt-injection=%.2f  adapt-both=%.2f\n", rf, ri, rb)
    println(rf <= ri && rf <= rb ?
            "→ Coefficient adaptation does NOT improve the operational forecast; keep EKF available, not deployed." :
            "→ Adaptation improved the forecast here — reconsider deploying.")
    return nothing
end

main()
