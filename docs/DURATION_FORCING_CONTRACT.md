# Duration-conditioned forcing B: pure research kernel contract v1

Frozen before implementation, 23 October 2026. Parent `75dc4bb25eda95ba70e0bc31ae330761f12ddd56`; branch `codex/duration-forcing-20261023`. Root authorized only pure kernels, synthetic fit/save/reload/as-of forecast tests and focused Julia1.12.6/1.13.1 verification. No actual fit, actual new outcome, future data, serving change, commit, push or full harness launch is authorized by this stage.

This supplements `manual/DST_NEXT_MECHANISM_CANDIDATE_CONTRACT_20261015.md:184–223` without altering the original A18-fit protocol, its negative initial result or any existing scientific deadline. A+B is a separate proposed experiment. Improving actual forecast skill and novelty remain unproved.

## Mathematical and source boundary

The supplied hourly q is the nonnegative thresholded injection already computed from completed raw speed/Bz. South S means Bz<0, otherwise North N. N requires q=0; finite negative q is invalid. A missing/invalid required hour or a timestamp gap breaks a run. An unknown initial run age is excluded from all risk/global counts until a witnessed flip establishes age1 or 49 contiguous known hours establish the capped49+ age. A missing hour is not a known non-flip or a synthesized transition. A final hour without an eligible next hour is right-censored.

Every training source hour and known-next transition/first-S mark must have nominal completion AND supplied assumed receipt <= parameter boundary minus168h. Equality is eligible; +1ms is ineligible. Input tags denote [tag,tag+1h); supplied receipt must not precede completion. This kernel stage supports declared hypothetical archival availability only and does not invent measured receipts. Fit preprocessing uses only eligible transitions; future data and later receipts cannot influence fitted parameters. Source provenance is caller-supplied and separately distinguished from the eligible training-prefix digest.

For each state s, age a∈1:49 and q bin b (one N bin, four S bins), record risk count n and flip count d. Global p_s is total flips/total risks over the same eligible known-age transitions. For beta∈{5,20}, h=(d+beta*p_s)/(n+beta), represented as an exact Rational{BigInt}. Empty cells shrink to p_s. A state with zero global support is unavailable rather than assigned a zero/default hazard.

Root accepted these previously unspecified conventions before code:

1. South q edges are nearest-rank inverse-empirical-CDF values at ranks ceil(N/4),ceil(N/2),ceil(3N/4), from eligible known-age South risk rows. Stable chronological order resolves value ties. q equal to an edge belongs to the lowest matching bin; repeated edges leave empty bins rather than discarding samples.
2. Unknown initial age excludes the row from every cell and global risk/flip count. First-S mark samples are the q of the next South hour of an eligible known-age North→South risk transition, with the same completed/received cutoff for both hours.
3. First-S samples are stably sorted by q then chronological order and partitioned into four rank blocks ending at ceil(k*N/4),k=1:4. Keep four separate arithmetic centroids with their actual block counts/masses, even when centroids coincide. At least four samples are needed to define all four centroids; otherwise marks are explicitly unavailable when a positive-probability N→S branch requires them. No arbitrary centroid or global probability is fabricated.

## Forecast and interval-observer integration

The latest completed driver must be the exact required hour immediately before the nominal issue, with an eligible assumed receipt. Missing latest data or unknown age returns explicit unavailable. Do not substitute an older driver.

From the observed current S/N, known capped age and q, enumerate every positive-support stay/flip path for1–6 future hours. S stay retains q and increments capped age; S flip becomes N age1,q0. N stay increments age with q0; N flip becomes S age1 and splits into four separately identified first-S marks. Retain duplicate-centroid branch identities and masses. No outcome-based pruning or random sampling. Unsupported state/mark support on any positive-mass required path makes the forecast explicitly unavailable; never drop that path and renormalize survivors.

For each path, propagate the existing A integrated Gaussian endpoint/interval kernel, using f_j=−g_j*q for two modes, fixed latest pressure and no future observations. Retain conditional endpoint covariance and each interval's affine offset/two gain coefficients. Historical base state and historical gain basis must refer to the same nominal issue. A caller may evaluate the affine output at fixed nonnegative gains; the original A fitter and actual permit remain untouched. A synthetic independent gain-fit oracle will exercise A's verified two-column NNLS on the new B affine design.

Return exact rational path mass and interval mixture mean, conditional variance, between-forcing variance and total variance via the law of total variance. Target observation variance is an explicit nonnegative input, never silently omitted or presumed calibrated. Future pressure uncertainty and parameter uncertainty remain absent and must be named in the result scope. The model is a research component, not an operational/probabilistic skill claim.

The covariance, zero-forcing base mean and zero-forcing interval offset depend on depth and the initial state, not on a future forcing branch. Compute these once per depth through the unchanged full A kernel. Precompute the one-hour mode coefficients once; share them while enumerating branch gain-basis recurrences. This preserves every branch and its covariance while avoiding a full BigFloat covariance calculation at every tree node. This new B path uses Float64 affine recurrences and must match independently evaluated full-A per-branch means/bases at the frozen `atol=5e-11,rtol=5e-14` on the declared synthetic range (initial means/covariances up to1e4 in magnitude, q up to100, gains up to100, horizons1–6, A's three tau pairs/sigmas and zero-lambda Brownian limits). The old A numerical path, tolerances and artifact identities remain unchanged. Nonfinite or unsupported arithmetic must reject rather than clip. No tolerance is selected from actual results.

The h6 tree bound follows (N',S')=(N+S,4N+S). Starting N gives terminal counts5,13,41,121,365,1093; starting S gives2,7,20,61,182,547. Including the initial node, at most1639 nodes are visited from N. This is a combinatorial bound; measured resources remain to be collected on synthetic controls after correctness.

## Artifacts and failure behavior

Persist only bounded data-only JSON with exact round-trip Float64 values, integer risk/flip counts, explicit unsupported fields, boundary/cutoff, source/training-prefix/code identities and a deterministic model digest. Use a new exclusively claimed directory, immutable manifest and a completion marker written last. Load requires the caller's expected manifest digest, exact membership, every payload digest, semantic validation and unchanged loaded code/A dependency identity. Unknown fields, missing completion, tampering, impossible counts/clocks, changed code and duplicates must fail closed. No deserialization of executable objects or unpinned artifact acceptance.

The fitted model's containers are immutable; public operations recheck source/dependency and model identities. A changed caller source SHA honestly changes artifact provenance even when numerical fitted parameters agree. Tests distinguish prefix causality from dishonest attribution of mutated full raw bytes to an old source digest.

## Independent pre-code oracles and required tests

`DURATION_FORCING_PRECODE_ORACLES_V1.json` freezes literal/rational expectations generated without the implementation:

- Complete South lengths {1,2,4,4}: age risks4/3/2/2, flips1/1/0/2, unshrunk remaining-S probabilities2/3,2/3,0 from age2; beta5/20 exact hazards.
- A censored South length3 adds exactly two known stays and no terminal flip/stay. A50-hour initial same-state prefix admits exactly one age49 transition;49 hours admits none.
- Stable nearest-rank/repeated-edge bins; four separate duplicate centroids retain their masses; explicit insufficient support.
- Alternating S/N fixture with four first-S marks {0,0,2,2}: all hazards1; h1/h2 means−11.5/−12 for fixed P0, initial zero mean/covariance, lambda0 and gain(1,0). With process sigma(1,0) and target observation variance0, h2 conditional variance4/3, forcing variance1,total7/3; adding target observation variance1 makes total10/3.
- Threshold example E∈{0,2},c=1/2: E[max(E−c,0)]=3/4 differs from max(E[E]−c,0)=1/2. This does not imply that averaging already transformed q through linear dynamics changes the mean; path enumeration is also needed for temporal dependence and mixture variance.
- Literal cutoff equality/+1ms, missing hours and observation ages49 boundaries; source and future receipt mutations; save/reload/relocation and digest/semantic corruption controls.
- Exact rational mass1 and the h6 combinatorial bounds; deterministic results across calls and native runtimes; independent two-column synthetic gain recovery and interval-vs-endpoint mutation sensitivity.

Focused tests and a synthetic end-to-end example must pass on Julia1.12.6 and1.13.1. Preserve failures and fix source causes; do not adjust oracles/tolerances after a failure. No fullPkg/H/profile/actual scientific execution occurs until root review.

## Prior overlap boundary

The existing mechanism contract already records prior integrated filtering, delayed updates, multi-reservoir Dst response, analog wind ensembles and physics-memory boosting. Root additionally read Salman et al.2018, DOI10.1029/2018SW002056, methods2.3–2.4 ([primary preprint](https://arxiv.org/html/1812.05423v1)): shock-conditioned historical analog ensembles predicting sustained southward Bz probabilities. Duration/southward probability forecasting is therefore not a novelty claim. Root identified Omondi et al. preprint DOI10.5194/egusphere-2025-6298 as a further overlap lead; no full-method conclusion is made here. A complete current primary-source overlap and ablation review remains required.
