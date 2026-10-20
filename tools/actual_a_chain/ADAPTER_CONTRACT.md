# Candidate A training/replay adapter: preimplementation contract

15 October2026. Coordinator review is required before adapter coding, raw-data application or real fitting. This is a separate research candidate, not an amendment to the frozen Sandu run, locked venue contract, ten-expert serving model,0.60 SINDy-family floor, conformal sidecar or paper. No live endpoint, model bundle, calibration identity, source receipt or manuscript changes are proposed. Existing negative experiments remain visible. A credible improvement and any uniqueness remain unverified hypotheses.

## Pinned foundation and dependency boundary

Use the existing SHA-pinned author-linked57-column archive,430248 unique contiguous hours,1976-11-01T00 through2025-11-30T23, SHA256 `593354e93e8f6c1486dc46de3ae96d4281d1129f9d5d39933347c2702122d248`; captured format SHA256 `ab7cc12ea18d178ed3b529d603b867c9e948df419275a4cea56a8da08042feee`. Reuse `SanduFullChannelFeatures.read_author_hourly` and its raw channel map/censor policy, without modifying the published feature or experiment modules. Its parser checks the complete byte digest and rejects a year above2025 before parsing that row's physical fields. The adapter must additionally assert the exact coverage/count/schema and freeze the parser source hash before any application. No2026 measurement path exists.

Bind the isolated kernel657cf432ffdcc09f3058a57e79e327c159292fe6e98318d4f0dc6bc917b4a04c and replay08f8088e55b3b10bb189d4f89360dd72e30e2c5f0f162f7c687e92a7f4ce1165. Their synthetic tests are a foundation, not integration clearance. Preserve `C:/Temp/da-20261015/CONTRACT.md`, its affine proof, B correction, all before-source failures and2,272 independent/347 own assertions. The complete source/dependency/configuration map, tested package map, Julia version and numerical environment must be frozen for the adapter's own chain. Native comparator code/results remain immutable while their current run finishes. Interpret no A result before those comparators complete their required verification.

## Hour and receipt algebra

Raw tag r denotes mean over[r,r+1h). The nominal issue/model boundary is integer hour i, equal to origin o+1h. Driver rows are available at nominal completion. Scenario L0 has Dst receipt r+1h; L1 has Dst receipt r+2h. These are declared hypothetical archival availability scenarios, never measured publication or actual live verification times.

|Quantity|L0|L1|
|---|---|---|
|Freshest raw V/Bz/P tag|i-1h|i-1h|
|Required anchor Dst tag|i-1h|i-2h|
|Anchor nominal completion|i|i-1h|
|Anchor assumed receipt|i|i|
|h target mean interval|[i+h-1h,i+h)|same|
|h target tag/completion|i+h-1h / i+h|same|
|Assumed target receipt|i+h|i+h+1h|

h is4 or6 and is lead to target-hour END, not its start or an instantaneous state. Integer-boundary archival ingestion is explicitly processed before the nominal issue snapshot, using the prototype's named hypothetical-inclusive policy. Actual future production receipts would require strict completed capture AND verification before actual issuance; archival equalities must never be relabeled actual evidence. Checked millisecond arithmetic and explicit clock fields are mandatory.

Example i=2020-01-01T00: both scenarios use driver tag2019-12-31T23; L0 Dst tag23 and L1 Dst tag22. h4 target is2020-01-01[03,04), received04 inL0 or05 inL1; h6 is[05,06), received06 or07. Target Dst is original censored data, never imputed. No observed future wind enters a scenario rollout. L1 shifts Dst availability only; it must not stale the fresh drivers or shift target labels.

For a known hour r, q=max(V_r*max(-Bz_r,0)/1000-0.49,0) mV/m, pressure offset H(P_r)=7.26sqrt(max(P_r,0))-11 nT and observation y_r=Dst_r-H(P_r). P is the actual raw pressure channel in nPa, not a fabricated n/V reconstruction. All three used driver values must be finite and V nonnegative; negative P follows the frozen max(P,0) formula and is separately counted. q uses hourly means as a declared coarse model, without invented subhourly fluctuations. n and the other real channels remain available to the stronger comparators but are not required when absent from A's equations.

Primary future forcing holds the latest completed q and P fixed across six future intervals. Forcing for each mode is f_j=-g_j*q; full-Dst target mean adds H(P_latest) to the latent interval mean. Pressure corrections for historical observations use their own historical hour's available P. There are no future observation updates. The two gains are shared by h4/h6 and not updated online.

## Training, selection and embargo

Initial fitting starts at the actual archive start; validation issues are[2020-01-01,2023-01-01), final development issues are[2023-01-01,2026-01-01), truncated by the actual November2025 target endpoint. These years are already exposed development, not fresh confirmation.

For a fitting boundary B, require issue<=B-168h, target completion<=B-168h AND assumed label receipt<=B-168h, plus issue<B. Both h4 and h6 labels must satisfy the rule on each **joint gain-fitting origin**. Equality is admitted; +1ms is rejected. Initial B=2020-01-01; final refit B=2023-01-01. Validation selection labels also require both maturity clocks<=2023-01-01 minus168h. No imputed outcome, held-out fitted transformation or post2025 field is admissible.

The latest individually eligible issues are:

|Boundary|L0 h4/h6|L1 h4/h6|
|---|---|---|
|2020-01-01|2019-12-24T20 / T18|2019-12-24T19 / T17|
|2023-01-01|2022-12-24T20 / T18|2022-12-24T19 / T17|

Joint two-lead fitting uses the earlier h6 boundary. Direct per-lead baselines retain their complete permissible per-lead training pools; do not weaken them by silently deleting the h4 tail or A-ineligible predictors. Report exact training masks/counts and the distinction from A's joint intersection. An additional strict-common-training sensitivity would require its own frozen refit, not replacement of the strong primary baselines.

Allowed state updates at validation/test issue use only observations revealed by that issue, including measurements newer than the parameter-fitting cutoff. This is causal inference input, not parameter refitting. Replaying the whole historical dataset to create a past issue may not assimilate labels that become available after that issue. Final gains refit only on the pre2023 eligible pool; no2023–2025 error selects a setting.

## Eighteen fixed primary settings and two-gain fit

Enumerate tau pairs(2,12),(4,24),(8,48)h; common per-mode sigma1/3/10 nT/sqrt(h); eta0/0.25. R=1 nT², initial mean0 and covariance10000I. That is exactly18 primary settings per latency scenario, in the listed lexicographic order: **36 visible initial A fits acrossL0/L1**, followed by at most one selected refit perL. This is a separate A-only panel; the old A+B two-beta count is not its experimental protocol. No B, online gain adaptation or hyperparameter search is added. Preserve every setting, fitted gain and validation panel.

For fixed tau/sigma and the causal missingness/observation schedule, obtain each joint target prediction as a_i+z_i' g from affine replay. Include the constant pressure offset in a_i, not z_i. A persistence-fallback row has a_i=the common anchor, z_i=0 and remains in loss/counts. Use no intercept or unconstrained residual correction. Estimate g1,g2>=0 by weighted least squares:

    L = mean_all[(a + Zg - y)^2]
        + eta * mean_target_Dst<=-50[(a + Zg - y)^2].

The all term pools both leads on the same joint origins (equal lead counts). If no target-storm entry exists, the storm term is exactly0 and both eta variants remain explicitly tied. Per-entry weights are1/N + eta*I_storm/N_storm; target-defined storm membership is used only in fit/evaluation, never as an issue-time input.

The two-variable NNLS solver enumerates the origin, both nonnegative axis optima and a feasible unconstrained least-squares optimum. **First filter each candidate by feasibility and KKT against the full original normalized design; only then compare objectives and apply minimum-norm/lexicographic ties among survivors.** An empty surviving set is a fitting failure. A KKT-failing tied candidate must not cause rejection of a different valid optimum. Use stable SVD, not an explicit inverse of normal equations. Normalize sqrt(weight)*Z and sqrt(weight)*(y-a) by one common finite maximum absolute entry (no absolute floor1); an allzero problem returns minimum-norm g=0. Reject a nonzero finite entry disappearing to0 during weighting/normalization, and reject nonfinite gains/results instead of silently accepting unsupported scale.

Let m=size(A,1), the number of weighted target rows, including fallback rows; it is not the number of scalar matrix cells. Fixed rank cutoff is max(m,2)*eps(Float64)*largest_singular_value. Fixed objective tie bound is64eps(Float64) times the maximum of zero-gain residual SSE and the compared SSEs, with no absolute floor. For normalized full A,b and gradient A'*(A*g-b), the fixed KKT bound is128eps(Float64)*max(m,2)*norm(A,2)*(norm(A,2)*norm(g,2)+norm(b,2)); require nonnegative gains, zero-coordinate gradient>=-bound and positive-coordinate absolute gradient<=bound. Compute this check with the actual full normalized matrix, not a rank-truncated reconstructed matrix. Rank/objective boundaries and KKT tolerances must be tested by independent scaled/rank/active-set fixtures before qualification; no threshold changes after a failed real fit. At least100 joint origins with nonzero A design are required; otherwise mark fitting failure, not a successful all-persistence fitted model.

Selection chooses at most one setting per latency scenario for the shared h4/h6 trajectory. First require validation pooled RMSE no worse than matched tunedGBM29 at **both** h4 and h6 on the exact joined, per-lead scoreable/mature validation cohorts, including every fallback. Joint-label intersection is a gain-fitting rule, not permission to shrink a per-lead scoring cohort. Among qualifying settings minimize the mean of the two lead-specific target-storm RMSE ratios A/GBM29. Fixed exact-score ties use the18-setting order. With a zero reference storm RMSE, define a zero candidate RMSE ratio as1 and a positive candidate RMSE ratio asInf; no finite normalized candidate means explicit none. A missing storm cell blocks the storm-selection reference instead of silently dropping a lead. Do not use a numeric tolerance to turn a worse pooled RMSE into a qualifier.

If no setting satisfies the two pooled constraints, preserve an explicit none/negative result and all18 panels. Do not select the least-bad nonqualifier from test results, label persistence a fitted A success, or refit a purported winner. Missing/corrupt reference artifacts are a blocked selection gate, distinct from a valid no-qualifier result. Report both lead-specific pooled/storm RMSE and every setting regardless. Refit only a selected setting's gains on the eligible pre2023 joint pool. This coordinator-required rule replaces the rejected allSSE+.25storm validation-selection proposal; the eta-weighted **gain-fitting** objective above remains unchanged.

The reference predictions must come from the selected GBM29 model trained only on the initial pre2020 eligible pool with the initial pre2020 imputer/scaling, never the final pre2023 refit. Current `SanduMatchedDevelopment._selection` keeps GBM models/predictions transiently and saves grid metrics, configuration and train/validation row IDs, not a complete prediction artifact. Consequently a separate exact-input initial-model replay/refit qualification is mandatory before the A selection gate can execute. Preserve the original Sandu run and create a new source/config/fit/feature/mask-bound validation-reference identity. Recompute all eight grid predictions and their selection in that isolated replay, retain every per-row prediction and fallback, independently recover the declared metrics, and investigate any mismatch with the old grid receipt. Summary metrics alone cannot establish point-prediction identity. A final-model prediction file, different-history GBM or old latency summary is not a substitute. Apply the same pre2020 rule to the newL1 reference.

## Common cohort and missing histories

For each L/h, freeze rows by exact issue/anchor/target keys with a finite **required** delayed anchor and actual target, independently of method errors. If that required anchor is missing, the row is non-scoreable for every method; do not silently choose an older finite anchor. The common persistence prediction is that same raw anchor. Keep a complete non-scoreable ledger, including unavailable archive boundaries.

A requires168 consecutive valid known-driver hours before model admission. A missing/invalid required q or P marks the current row persistence fallback. Restart the prior at the start of the next valid segment, retain the grid/gap in evidence, and require another168 contiguous valid-driver hours. Missing Dst alone means no likelihood for that bin, while known forcing still propagates; it does not fabricate a Dst value or reset state. An eligible finite anchor remains required at issue. A missing observation's final-archive status is explicit, not a future receipt invented for a nonexistent measurement.

All settings share this deterministic admission/reset policy. No gap is concatenated away. Fallback reasons include warmup, required-known-driver gap, replay-budget failure and unavailable finite design. Exceptions, invalid models, inconsistent receipts or negative covariance are experiment failures, not catch-all persistence. Candidates never delete rows on which a stronger baseline predicts. Complete model/fallback counts and prefix/reset identities are reported.

## Streaming and durable checkpoints beyond4096 intervals

The prototype4096 bound is a per-call memory guard, not a limit on total training history. For the deterministic L0/L1 archive, stream completed hours and retain the pending delayed-observation tail plus a checkpoint before that tail. A bin is sealed only after its declared receipt opportunity has passed and its original value/missing status is known. The maximum fixed L=1 delay makes a small tail sufficient; independently prove boundary behavior rather than dropping the final unassimilated bin at a chunk edge.

Maintain a rolling168-hour driver-validity/reset ledger. Stream affine baseline/two coefficient columns for h4/h6 to hash-bound chunk files, rather than retaining all430248 historical state checkpoints. Covariance/gain schedules are reusable for the two eta objectives because eta/g do not change them; there are nine distinct tau/sigma schedules perL. Baseline and basis must carry through every chunk identically to a full-prefix reference. A fitted replay reruns this causal stream with fixed gains. Its future-target queue contains at most six bins, plus the declared receipt delay for label eligibility. Input parser memory is separate and measured; this first adapter may hold its verified17-channel matrix but may not call that fully streamed raw input.

For later actual late/revised receipts, use immutable disk checkpoints at fixed24-hour boundaries, a bounded active tail, and the complete append-only receipt/selected-revision index. Each checkpoint binds initial/reset identity, configuration/gains, source prefix, selected-likelihood prefix, boundary/asof and parent hash. A changed old likelihood invalidates descendants: load a checkpoint before the earliest changed bin and replay forward in bounded chunks using the latest eligible capture per bin as of the new issue. Never assimilate the revision again at the present endpoint, discard it because it predates the memory window, or modify an issued forecast. This actual-evidence extension requires separate journal-adapter tests and is not silently claimed by the archive adapter.

If a required checkpoint/source interval is absent or a preregistered replay budget is exceeded, retain the original issued history and use an explicit current-row persistence fallback. No approximate truncated replay may be called exact. Resource gates measure live memory/handles, chunk-size invariance, long-history scaling and revising before the active window. The prototype's measured kernel/cycle timings are reference evidence only, not a promise for the full adapter.

## Matched comparator controls

At L0, the existing frozen native run may be joined only after it completes and independently matches raw-source SHA, all issue/target/anchor keys, actual targets, cutoff contracts and common row hashes. Keep all six native full/raw families/recipes, strong tuned GBM29 and persistence; retain their imputation/extra-channel advantages and fallback statuses. A uses fewer channels; this is disclosed, not a reason to weaken them. Existing1963-start memory/ARX results remain a separate unequal-history panel, with all negative results preserved.

L1 requires a newly qualified delay-aware baseline view/refit before any matched claim: delay every Dst-dependent feature/transform while keeping current completed drivers fresh; retain original target values. Do not reuse L0 predictions by relabeling issue times. Full native/GBM family definitions, grids, early-stop/refit-tree rule and source-trained imputer recipe remain unchanged; bind the new receipt-aware feature sources/masks. Refits use the same1976 source start and L1 label-receipt embargo. No comparator gets future q/P, contemporaneous unavailable Dst or an altered missingness cohort.

Research contrasts must name the compared lead, latency, common cohort and method identity. All h4/h6 pooled, target<=-50/<=-100, anchor-storm and fallback panels remain visible. Any method-selection multiplicity or confirmatory superiority rule requires another declared protocol before a formal claim; unadjusted exploratory intervals do not establish universal dominance. Mechanism ablations from the original candidate contract still need their own fixed fitting/selection implementation before attributing improvement to the interval operator. No A verdict is permitted from infrastructure or synthetic tests.

## Required synthetic chain before real-data authorization

1. Independent integer-millisecond L0/L1 clocks and cutoff fixture: both horizon targets, joint-mask intersection, exact-cutoff acceptance and+1ms rejection, end-of-source no2026 read, delayed Dst/fresh wind and missing-grid controls.
2. Raw-to-q/pressure-corrected-mean scalar fixtures with actual column-name/index checks, fill/nonfill censor counts and no target imputation; arbitrary-prefix/future mutation invariance.
3. Affine design versus direct candidate replay for many constrained gains, every tau/sigma and both delays; independent dense Gaussian reference on short histories and independent two-variable NNLS/KKT boundary/rank cases. A deliberately endpoint-based observation must fail the integrated-mean oracle.
4. Long synthetic sequences exceeding4096/8192 hours, chunk sizes1/7/24/511/4096, a delayed observation straddling each chunk boundary, missing-driver resets and late revisions older than the active window. Match full-prefix point/covariance/selected-factor identities and retain issued forecasts unchanged.
5. Fully synthetic source->clocks->streamed basis->NNLS->validation selection->refit->save->reload->prediction. Model/checkpoint artifacts store finite exact-roundtrip gains/configuration, source/kernel/parser/contract hashes, fit/selection masks, actual maximum permitted label clocks, training coverage, KKT/roundtrip receipts and immutable model identity. Corrupt coefficients, source, masks, clocks or model hashes must fail closed; wrong-state/checkpoint bindings must be rejected. No serving identity is reused.
6. Independent prediction/metric/artifact reader, saved deterministic paired draw arrays, realistic memory/time/handle profiles, clean-checkout source closure and the mandatory integration Pkg.test/examples/dev-harness chain. Preserve all failures and rerun affected gates after a fix. Only then may the coordinator explicitly authorize a separate exposed-data fit.

This contract authorizes no code or data execution by itself. It does not change the locked current paper or product and does not claim the prototype is a complete forecaster.
