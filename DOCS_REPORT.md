# Documentation repair verification, 18 October 2026

Strict CI run 39122321358 on b390755368dbc042a6e3a2f77d84062037141f37
failed with 25 documented exported implementation bindings missing from the
manual. Their public wrapper entries were already present. The repair adds
the existing numerical, replay, feature, stream, training and artifact
docstrings through explicit implementation-module references.

| Check | Result |
| --- | --- |
| Public interval namespace bindings represented in the manual | 19/19 |
| Previously missing implementation docstrings | 25 included |
| `checkdocs=:exports` | Retained |
| `doctest=true` | Retained |
| `warnonly=false` | Retained |
| Native Windows Julia 1.12.6 build | Exit 0, 158.11 seconds |
| Documenter errors or warnings | Zero |
| Frozen tracked files changed during build | Zero |

The build retained dependency precompilation warnings from GeometryBasics;
they are not Documenter manual warnings and were not hidden or filtered out
of the raw log. This report states the checked manual coverage, not a claim
that every internal helper has a docstring.

Build receipt SHA-256: 5ccdca94a1ea2c6ec22e8d94eab1de4ad39c66581310954c772d6d6d2817d5ce.
Full raw build log SHA-256: 25a70dcae72bc28d668a7f6720dabcf1f2bf7cd369146a57bf20e8b050a3cc86.
Manual source SHA-256: 51802c94df83645b9bb3d21035c2fd40b0af13b5d9ae3666f3fbc96f48af9da0.

All computational source, test, example, workflow and environment files are
unchanged from b390755. The full package/experiment/harness and six-cell CI
qualifications of that computational candidate remain pending at this point.
No empirical forecast-skill, novelty, actual-archive or deployment claim follows
from this documentation build.


## Consolidated disk-fitting candidate

The earlier build above applies to commit 41f44f6 and its 19 public interval
exports. The consolidated candidate retains those references and adds two
maintained disk-fitting exports with their implementation docstrings. Its
21-export manual, full package tests, experiments, canonical harness and native
six-cell matrix require fresh qualification of the actual merged commit. The
earlier 19-export build is not relabeled as a run on the expanded interface.


## Recipe metadata repair

The consolidated 21-export manual at 8d13a3b passed strict Ubuntu CI run
39126089371, job117671411714, log SHA-256
806a5b614d950422b422b930aa781bcb1cd11d42c53022478aa62ad0a97ed50f.
The later acaf046f recipe repair adds explicit numeric-representation semantics
to the manual. A fresh strict build is required for that final committed source;
the 8d13a3b pass is retained as historical evidence only.


## B reader correction documentation checkpoint

The fixed-header, slot/certificate and pending-container corrections preserve
the existing public exports, field sets and numerical semantics. Added helpers
are private. Both portable suites and positive-first public regressions pass on
the current computational bytes. Fresh strict documentation builds on the final
combined source are still required; prior documentation receipts are not relabeled
as runs on this correction. Actual archival admission and forecast-skill claims
remain outside this component checkpoint.


## Matched342 API and official Expat documentation lock update

The additive matched342 manual, implementation docstrings and generated
doctest build under both Windows Julia1.12.6 and1.13.1 with
checkdocs=:exports, doctest=true and warnonly=false unchanged. The 1.13 build
initially failed because Application Control blocked the installed official
Expat_jll2.7.3+0 DLL. A byte-for-byte comparison with its official archive ruled
out local corruption. The current official2.9.0+0 update changes exactly that
dependency's version and tree hash in docs/Manifest.toml. In an isolated exact
source copy, strict1.13 and1.12 builds exit0 without Documenter errors or warnings.
Their raw logs retain the upstream compiler/import warnings. No security setting
was changed, and no scientific Project.toml or Manifest.toml was modified.

The updated-lock Julia1.13 log SHA256 is
fed299f6573ea0e928539249b2be5dcc5e817d8b345405a9bc4975748e54fcf7;
Julia1.12 log SHA256 is
9e579b209dadfab7063412ac126e8e9747afbf7c22450b084f976756926621da.
The blocked original build remains a failed historical attempt, not a waived
check. Final committed full-package, experiment, harness and CI checks remain
required before release.


## Physical artifact-root clarification

The matched342 manual now states that artifact paths must have ordinary
physical ancestors and that callers should resolve their own newly created
temporary directory before writing an artifact. Alias paths remain rejected.
Only tests, their example and documentation change in this correction; all
production modules and exported signatures are byte-identical to f05. Fresh
strict builds on both final-source native runtimes remain required. The
documentation-only official Expat_jll2.9.0+0 lock is retained, and no security
setting or scientific dependency lock is changed.


## Archival source-format namespace

The manual documents the package namespace, explicit inclusive ground and
half-open wind periods, per-body byte/row limits, raw and selected witnesses,
validity masks and reasons. Unavailable payload zeros are interpreted only
through their masks. Body parsing supplies no provider-clock, cross-body,
source-admission or predictive-performance claim. Strict final-head builds
remain required before publication; no Documenter setting is weakened.


## Independent reader API documentation

The archival-format manual and namespace docstring describe independent
reconstruction, immutable selected and raw records, digests and the remaining
support-clock and catalog boundaries. The example compares both readers on
generated text. Strict final-source documentation builds are still required.
