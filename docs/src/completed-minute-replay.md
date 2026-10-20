# Completed-source minute replay

`CompletedMinuteReplay` constructs an explicit retrospective information set
for 30-minute ground dB/dt targets. Source tags and analysis availability have
separate fields. Ground source delay also delays future labels; additional
label processing is counted afterward.

```@example completed_minute
using Dates, SolarSINDy
using SolarSINDy.CompletedMinuteReplay
policy = ReplayPolicy(ground_extra_ms=120_000)
timing = issue_timing(DateTime(2024,5,10,12), policy)
@assert timing.latest_ground_center == DateTime(2024,5,10,11,57)
@assert timing.label_assumed_ready == DateTime(2024,5,10,12,32,45)
@assert timing.sufficient_phase_period_minutes == 33
@assert !timing.measured_receipts_available
nothing # hide
```

With zero extra delay, a whole-minute issue uses the preceding ground center
and OMNI arrival-start row. The 30 target centers are the next 30 minutes;
their last ground filter completes 45 seconds after the final center. The
earliest whole-minute feedback is therefore 31 minutes after issue.

`admit_features` accepts dictionaries of ground `(X,Y)` vectors in nT and wind
`(V,Bz,By,n)` rows in km/s, nT, nT and cm^-3. It accesses only the admitted
prefix. `admit_target` takes an explicit `asof` and checks readiness before
accessing target vectors. `replay_case` joins the two results. Every admission
result has `accepted`, `reason` and `value` fields; rejected inputs retain an
explicit missing, invalid or not-ready reason.

These helpers implement a declared analysis convention. NASA's OMNI arrival
times and final ground archives do not establish actual historical receipt or
revision vintage. The module preserves that distinction in its output. It
constructs cases rather than fitting or serving a model; changed information
sets require a complete new numerical replay before use in paper results.

```@docs
SolarSINDy.CompletedMinuteReplay
SolarSINDy.CompletedMinuteReplay.ReplayPolicy
SolarSINDy.CompletedMinuteReplay.issue_timing
SolarSINDy.CompletedMinuteReplay.minute_phase
SolarSINDy.CompletedMinuteReplay.training_eligible
SolarSINDy.CompletedMinuteReplay.admit_features
SolarSINDy.CompletedMinuteReplay.admit_target
SolarSINDy.CompletedMinuteReplay.replay_case
```
