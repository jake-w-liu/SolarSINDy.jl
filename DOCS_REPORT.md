# Documentation Verification Report

Date: 2026-08-19

Supersedes the 2026-08-10 report, whose coverage figures (132 exports, zero missing docstrings,
"Remaining Gaps: None") were measured before the V2.2 research chain, the V2.3 study machinery and
the V2.4e serving layer were exported, and whose recorded clean build could not have been
reproduced: `docs/Manifest.toml` still resolved `SolarSINDy` at `0.1.0` without `EvoTrees` or
`Logging`, so the documented build command failed at instantiation.

## Environment

`docs/Manifest.toml` is re-resolved against the current `Project.toml`. The manifest now records
`SolarSINDy` at `0.2.1` with `EvoTrees` and `Logging` in its dependency list (`julia_version =
"1.12.6"`). The documented command sequence is

```text
julia --project=docs -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

and both steps complete with exit code 0.

## Coverage

Measured on the current tree with `names(SolarSINDy)` and `Base.Docs.meta(SolarSINDy)`:

| Quantity | Count |
|---|---|
| Exported bindings (excluding the module name) | 427 |
| Exported bindings carrying a docstring, directly or through an alias | 380 |
| Exported bindings with no docstring at all | 47 |
| Source files contributing documented exports | 35 |

The manual reaches the documented set through `@autodocs` blocks keyed on source file with
`Private = false`, so every exported binding of a documented file is rendered without being listed
by hand and a new export cannot silently fall out of the manual. Three entries remain explicit: the
module docstring, the internal `AlarmCooldownState` returned by alarm evaluation, and `parse_omni2`,
whose docstring lives on the aliased `parse_omni2_csv`.

Pages (`docs/make.jl`):

| Page | Contents |
|---|---|
| `api.md` | utilities, candidate libraries, sparse identification, data loading and cleaning, storm selection, baselines, metrics |
| `forecast-api.md` | forecast issuance and rollout, alarms, conformal intervals, assimilation, monitoring and real-time feeds |
| `operational-api.md` | versioned cores and calibration artifacts, static V2.2 stack serving, V2.3 analog serving |
| `operational-v24-api.md` | V2.4e super-learner serving and depth-safe severity |
| `operational-v22-research-api.md` | the offline V2.2 research chain (arrival, driver, core path, boost, error models, history, residual, shadow chain) |
| `operational-v23-api.md` | V2.3 study features, analog archive, gradient-boosted comparator, statistics |
| `index.md`, `examples.md`, `live-verification.md`, `ekf-v3-decision.md` | narrative pages |

## Build

`docs/make.jl` runs with `checkdocs = :exports`, `doctest = true` and `warnonly = false`. The build
completes with **exit code 0, zero errors and zero warnings**. The previous structure failed this
same command with `258 docstrings not included in the manual` and terminated before rendering.

Every generated page is below Documenter's 100 KiB advisory; the largest is
`operational-v22-research-api.html` at 95.5 KiB (the pre-split `api.html` was 134.3 KiB and
`operational-api.html` 106.1 KiB, both of which raised size warnings). Generated `docs/build/`
content remains ignored and untracked.

## Remaining Gaps

1. **47 exported bindings carry no docstring.** They are the V2.2 research-chain and V2.3/V2.4
   configuration constants (`OPERATIONAL_V22_ARRIVAL_*`, `OPERATIONAL_V22_DRIVER_*`,
   `OPERATIONAL_V22_ERROR_*`, `OPERATIONAL_V22_HISTORY_*`, `OPERATIONAL_V22_RESIDUAL_*`,
   `OPERATIONAL_V22_SHADOW_*`, `OPERATIONAL_V22_BOOST_*`, `OPERATIONAL_V22_CORE_PATH_*`,
   `V23_MEMBER_MAX_N_CM3`, `V24_SERVED_GUARDED_IDENTITY`, `V24_SERVING_ANALOG_WEIGHT_SET`,
   `V24_SERVING_DST_CEIL_NT`, `V24_SERVING_EXPERT_COUNT`, `V24_SERVING_GUARD_DEPTH_NT`,
   `V24_SERVING_GUARD_REFERENCE_NONE`, `V24_SERVING_POOLED`) plus
   `operational_v22_h1_innovation`. `checkdocs = :exports` does not fail on them — there is no
   docstring to place — so the build is green while the constants are undocumented. Writing those
   docstrings edits `src/`, which is outside this pass.
2. The narrative pages (`index.md`, `examples.md`, `live-verification.md`) are maintained by hand and
   are not covered by any automated check beyond Documenter's cross-reference and doctest stages.

Docs status: environment re-resolved, build green at `warnonly = false`, coverage measured rather
than asserted, and the two gaps above recorded rather than described as none.
