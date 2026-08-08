# Documentation Verification Report

Date: 2026-08-08

## Coverage

All 132 exported bindings have Julia doc bindings and appear exactly once in
the API reference. The V2.1 migration audit added the twelve versioned-core
bindings to the operational reference, documented both version constants, and
resolved the `AlarmCooldownState` cross-reference by including that internal
return type in the manual.

The reference is divided between `docs/src/api.md` (61 core bindings) and
`docs/src/operational-api.md` (71 exported operational bindings, plus the
documented internal cooldown-state return type). This keeps both rendered pages
below Documenter's 100 KiB advisory threshold without omitting or duplicating
an exported symbol.

## Build

The exact command

```text
julia --project=docs docs/make.jl
```

completed with exit code 0. `docs/make.jl` explicitly enables
`checkdocs=:exports`, `doctest=true`, and `warnonly=false`. The doctest and
document-check stages completed with zero errors and zero warnings. The build
rendered the home, core API, operational API, examples, live-verification, and
EKF-decision pages. A separate Julia binding audit counted 132 exports and zero
missing docstrings. Generated `docs/build/` content remains ignored and is not
tracked.

## Remaining Gaps

None.

Docs complete: 14 issues detected → 14 confirmed → 14 fixed, 0 require user
action.
