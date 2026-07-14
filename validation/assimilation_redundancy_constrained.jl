# assimilation_redundancy_constrained.jl — does a STABILITY-CONSTRAINED EKF earn the 1-step gain WITHOUT
# the multi-step blow-up that sank the unconstrained filter?
#
# assimilation_redundancy_power.jl verified the unconstrained EKF is harmful at multi-step because the
# online filter drives the decay coefficient across [-0.51, +0.54]; a POSITIVE decay coefficient is a
# dynamically unstable ODE, so free-running rollout diverges. The fix tested here: an opt-in projected EKF
# (init_assimilation(...; coeff_bounds)) that holds the adapted decay coefficient <= cap < 0 after every
# update. We sweep the cap (just-stable -1e-3, mild -0.02, fixed-strength -0.048) and, for each, repeat the
# powered multi-step comparison (n=31 cycle-25 storms, broad leakage-free v2 calibration, horizons 1/2/3/6 h):
#   A=fixed raw  B=v2(fixed+corr)  C=ekf raw  D=ekf+corr ; paired B-D = EKF gain on top of v2 (>0 favours EKF).
# The fixed pass + its calibration are computed once (cap-independent). We also print the CONSTRAINED decay
# range to confirm the box binds. Verdict (variance-aware): a cap is a WIN only if D <= B at every horizon
# (no multi-step harm, flagship not blown up) while keeping the 1-step behaviour — i.e. the constraint
# removes the harm. If even the constrained filter only matches v2, the EKF is (cleanly) redundant; if it
# still diverges, the decay box was not the whole story.
#
# RESULT (run below, n=31 storms) — the constraint WORKS; the EKF becomes deployable:
#   cap = -0.048 (the discovered physical decay; adapted decay free to STRENGTHEN, never weaker):
#     horizon   A fix   B v2   C ekf   D e+c   B-D (EKF gain on v2)
#       1 h     9.66    9.14   8.42    8.16    +0.99 ± 0.33   (23/8 storms; flagship +7.25)
#       2 h    13.09   13.13  12.58   12.69    +0.44 ± 0.34
#       3 h    16.59   17.17  16.39   16.92    +0.25 ± 0.36
#       6 h    26.57   27.97  26.22   27.52    +0.46 ± 0.44   (flagship +7.60)
#   The decay box binds (range [-0.559, -0.048], vs [-0.51,+0.54] unconstrained), so the multi-step
#   DIVERGENCE is gone: 6 h B-D goes -35.56 (unconstrained) -> +0.46 (constrained); flagship -394 -> +7.6.
#   1-step gain is SIGNIFICANT (+0.99 ± 0.33, >2.5 SE). At the -0.048 cap multi-step is NEUTRAL (B-D within
#   ~1 SE of 0). IMPORTANT — this multi-step safety is CAP-DEPENDENT, not general: the looser caps leave
#   SIGNIFICANT residual multi-step harm (6 h B-D = -2.67 ± 0.42 at cap -0.001, -1.29 ± 0.39 at -0.02 —
#   several SE below 0), so the constraint STRENGTH matters and only the -0.048 cap removes the harm. That
#   cap is the principled choice (the discovered physical decay value) AND the one that works; it is not a
#   free parameter tuned to win. The 1 h gain (~+1.0 nT) is robust across ALL caps. VERDICT: deploy-worthy
#   AT cap=-0.048 — wiring the constrained EKF (init_assimilation(...; coeff_bounds=[(-Inf,-0.048)])) under
#   v2 improves the 1 h forecast ~1 nT with no multi-step harm; a weaker cap must NOT be used (it reintroduces
#   multi-step harm). Operational value concentrated at 1 h.

using SolarSINDy, CSV, DataFrames, Statistics, LinearAlgebra, Printf, Dates

const PKG  = pkgdir(SolarSINDy)
const PROJ = normpath(joinpath(PKG, ".."))
const EXTRACTED = joinpath(PROJ, "paper_v2_monitor", "data", "omni_extracted.csv")
const CATALOG   = joinpath(PKG, "data", "storm_catalog.csv")
const COEFCSV   = joinpath(PKG, "data", "real_sindy_discovery_coefficients.csv")
const QEKF = 1e-4
const HORIZONS = [1, 2, 3, 6]
const DEPTH = -80.0
const CAPS = [-1e-3, -0.02, -0.048]      # decay-coefficient upper bounds to sweep (all < 0 = stable)

function fillnan!(x::Vector{Float64})
    n=length(x); l=NaN; for i in 1:n; isfinite(x[i]) ? (l=x[i]) : (isfinite(l)&&(x[i]=l)); end
    l=NaN; for i in n:-1:1; isfinite(x[i]) ? (l=x[i]) : (isfinite(l)&&(x[i]=l)); end
    any(!isfinite,x)&&(x.=ifelse.(isfinite.(x),x,0.0)); x
end
rmse(e)=sqrt(mean(e.^2))
emptyrows()=DataFrame(latest_dst_nt=Float64[],V_kms=Float64[],Bz_nt=Float64[],By_nt=Float64[],n_cm3=Float64[],
                      Pdyn_npa=Float64[],pred_dst_nt=Float64[],pred_dst_ci05_nt=Float64[],pred_dst_ci95_nt=Float64[],observation_dst_nt=Float64[])

function main()
    df = parse_omni2(EXTRACTED; year_start=2022, year_end=2025); clean_omni_data!(df)
    for c in [:V,:Bz,:By,:n,:Pdyn]; df[!,c]=fillnan!(Float64.(coalesce.(df[!,c],NaN))); end
    dst = Float64.(coalesce.(df.Dst_star, NaN)); N = nrow(df)
    cat = load_storm_catalog(CATALOG)
    storms = sort(filter(e->e.split=="test" && year(e.min_dst_time)>=2023 && e.min_dst<DEPTH, cat), by=e->e.min_dst)
    flagship = storms[1].storm_id
    sid = zeros(Int,N)
    for s in storms; lo=s.min_dst_time-Hour(36); hi=s.min_dst_time+Hour(72)
        @inbounds for k in 1:N; (df.datetime[k]>=lo && df.datetime[k]<=hi)&&(sid[k]=s.storm_id); end; end
    lib=build_solar_wind_library(include_redundant_n_v2=true); tn=get_term_names(lib); coef=CSV.read(COEFCSV,DataFrame); ξ0=zeros(length(lib))
    for r in eachrow(coef); i=findfirst(==(r.term),tn); i!==nothing&&(ξ0[i]=r.coefficient); end
    i_decay=findfirst(==("Dst_star"),tn)
    drivers=[(V=df.V[k],Bz=df.Bz[k],By=df.By[k],n=df.n[k],Pdyn=df.Pdyn[k]) for k in 1:N]
    @printf("storms n=%d; flagship id=%d (%s, %.0f); fixed decay=%.5f\n",
            length(storms), flagship, string(storms[1].min_dst_time), storms[1].min_dst, ξ0[i_decay])

    function run(adapt, cap)
        q = isempty(adapt) ? 1e-6 : QEKF
        cb = (isempty(adapt) || cap===nothing) ? nothing : [(-Inf, cap)]
        f = init_assimilation(lib, ξ0, adapt, isfinite(dst[1]) ? dst[1] : 0.0; q_coeff=q, coeff_bounds=cb)
        calib=emptyrows(); srows=Dict{Tuple{Int,Int},DataFrame}(); cmin=Inf; cmax=-Inf
        for k in 1:N-1
            if !isempty(adapt); cd=current_coeffs(f)[1]; cmin=min(cmin,cd); cmax=max(cmax,cd); end
            if sid[k]!=0 && isfinite(dst[k])
                for Nh in HORIZONS
                    (k+Nh<=N && isfinite(dst[k+Nh])) || continue
                    g=deepcopy(f); for _ in 1:Nh; assimilation_predict!(g,drivers[k]); end
                    p=current_dst(g); key=(sid[k],Nh); haskey(srows,key)||(srows[key]=emptyrows())
                    push!(srows[key],(dst[k],df.V[k],df.Bz[k],df.By[k],df.n[k],df.Pdyn[k],p,p-15.0,p+15.0,dst[k+Nh]))
                end
            end
            assimilation_predict!(f,drivers[k]); p1=current_dst(f)
            if sid[k]==0 && isfinite(dst[k]) && isfinite(dst[k+1])
                push!(calib,(dst[k],df.V[k],df.Bz[k],df.By[k],df.n[k],df.Pdyn[k],p1,p1-15.0,p1+15.0,dst[k+1]))
            end
            assimilation_update!(f,dst[k+1])
        end
        calib, srows, (cmin,cmax)
    end

    corrected(testdf,cal)=begin
        prep=SolarSINDy.add_operational_v2_features!(copy(testdf))
        [prep.pred_dst_nt[i]+SolarSINDy.operational_v2_correction(cal,
            NamedTuple{Tuple(cal.feature_names)}(Tuple(Float64(prep[i,c]) for c in cal.feature_names))) for i in 1:nrow(prep)]
    end

    calib_f, storm_f, _ = run(Int[], nothing); cal_f = fit_operational_v2_calibration(calib_f)

    for cap in CAPS
        calib_e, storm_e, cr = run([i_decay], cap); cal_e = fit_operational_v2_calibration(calib_e)
        @printf("\n=== decay cap = %.4f  (constrained adapted-decay range [%.4f, %.4f]; was [-0.51,+0.54] unconstrained) ===\n",
                cap, cr[1], cr[2])
        @printf("%-5s %4s | %6s %6s %6s %6s | %-20s\n","horiz","n","A fix","B v2","C ekf","D e+c","B-D mean±SE (fav/hurt) flagship")
        for Nh in HORIZONS
            A=Float64[];B=Float64[];C=Float64[];D=Float64[];bd=Float64[]; fav=0;hurt=0; flag=NaN
            for s in storms
                kf=(s.storm_id,Nh); (haskey(storm_f,kf)&&haskey(storm_e,kf))||continue
                tf=storm_f[kf]; te=storm_e[kf]; (nrow(tf)<5||nrow(te)<5)&&continue
                a=rmse(tf.pred_dst_nt.-tf.observation_dst_nt); b=rmse(corrected(tf,cal_f).-tf.observation_dst_nt)
                c=rmse(te.pred_dst_nt.-te.observation_dst_nt); d=rmse(corrected(te,cal_e).-te.observation_dst_nt)
                push!(A,a);push!(B,b);push!(C,c);push!(D,d);push!(bd,b-d)
                (b-d)>0.05&&(fav+=1);(b-d)<-0.05&&(hurt+=1); s.storm_id==flagship&&(flag=b-d)
            end
            n=length(bd); se=n>1 ? std(bd)/sqrt(n) : NaN
            @printf("%-3dh  %4d | %6.2f %6.2f %6.2f %6.2f | %+.2f ± %.2f (%d/%d) flag %+.2f\n",
                    Nh,n,mean(A),mean(B),mean(C),mean(D),mean(bd),se,fav,hurt,flag)
        end
    end
    println("\nRead: a cap WINS if D<=B (B-D>=0) at every horizon (harm removed) — then the constrained EKF is")
    println("safe to deploy under v2 (and helps where B-D>2SE). If D~B, harm removed but EKF redundant. If C")
    println("still explodes at 6h, the decay box was not the whole story.")
    return nothing
end

main()
