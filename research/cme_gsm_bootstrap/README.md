These sources are integrated into the package and the separate four-job
`.github/workflows/cme-gsm-bootstrap.yml` workflow. Integration is not native or
scientific qualification. Production matrix admission still requires all four
genuine successful native jobs, three byte-equality controls and independent
retained-log/artifact verification before reader setup or any study GSM values.

The original external preparation package contained `PRECODE.json` and wrote
`frozen_population_years.toml` before the Julia implementation. That historical
precode/recovery evidence is separate from this integrated source directory.
The committed derivative preserves all 551 original IDs and UTC MO-start
timestamps in original order, and the sorted year universe 1995 through 2025
(31 years). It contains no observations, speed values, quality values or labels.
The original population and protocol SHA-256 values are bound in the derivative.
The precode's `literal_writer_oracle_sha256` binds the JSON oracle file; the
literal 12 CSV bytes `2,1\n1,2\n2,2\n` separately hash to
`4dc41bae5c2d906887e2b55d8586d810afa33f2f4ed7dea49f83dfa419321fb2`.

The generator uses only Julia Random/SHA/TOML/Dates standard libraries. Production
generation is exactly Julia 1.12.6, `MersenneTwister(170281)`, 10,000 outer draws,
31 inner slots, one scalar `rand(rng, 1:K)` per slot. It writes UTF-8 ASCII CSV
without a header or BOM, comma separators, LF after each draw, and 1-based indices.
Four artifact members bind matrix bytes, source, runtime and full year lineage.
No actual production RNG matrix hash is predeclared here. Admit only the bytes
and hash retained by the successful native primary and all three controls.

The literal fake-draw test fixes bytes and scalar-call order independently of
Julia RNG output. The native test separately reconstructs each real RNG row as a
tuple, compares exact serialized lines, checks no overwrite, and mutates copied
artifacts to test matrix, lineage, source, year order and year omission rejection.
The original preparation tests were unexecuted at authoring. Current validation
status must come from exact-source native receipts; static review is not a Julia
syntax, runtime, numerical, cross-platform or scientific qualification.

`run_native_bootstrap.py` is the integrated owner. It requires its sibling
files committed under `research/cme_gsm_bootstrap/` in a clean repository. It pins
and reuses the existing repository-native descendant owner and physical-memory
helper, verifies exact HEAD bytes before and after, and refuses to run below
8 GiB available physical memory. It owns one Julia child with a 900-second limit,
uses standard libraries only, retains logs, partial results and receipts, and
checks Julia's measured peak RSS against 8 GiB. This last check is retrospective;
it is not an OS-enforced memory cap. This authoring note supplies no native run.

The full-file pins form a dependency chain: the bootstrap owner pins the memory
helper, which itself pins the native owner. A helper-owner pin edit therefore
also changes the memory helper's full-file hash. The package-invoked controls in
`test/cme_gsm_bootstrap/` check these transitive references and exercise the real
source preflight on bounded clean generated repositories, including the original
stale-memory-pin rejection. They perform no Julia or observational work.

The original preparation workflow was an inert `.yml.template`; the integrated
workflow is now active on the declared repository events. Its four jobs are Linux
1.12.6 candidate primary, Windows and Intel macOS 1.12.6 controls, and Linux
1.13.1 control. Standard `macos-15-intel` total RAM is documented as 14 GB by
https://docs.github.com/en/actions/reference/runners/github-hosted-runners;
every host must still pass the actual available-memory test. The integrated
workflow pins action revisions. macOS provisions Python3.14 before the owner;
Windows/Linux use Python3.12.

The integrated workflow triggers on push to `main` and the existing
`codex/catalog-calibration-integration-20261027` candidate, and retains manual
dispatch for later use. A dispatch-only new workflow cannot bootstrap before
default-branch registration under the official GitHub event rules:
https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_dispatch.
The external preparation record retained the original13 reviewed files and froze
`PRECODE_AMENDMENT_V1.json` before that trigger correction. Those historical
receipts are not new runtime authority; the551-event TOML remains unchanged.

Runtime controls distinguish same-version OS equality, measured1.13.1 native RNG
generation, and reproduction of the frozen primary through a reader. Julia's
official Random documentation warns that version changes can change streams
and recommends retaining generated random data for exact reproducibility:
https://docs.julialang.org/en/v1/stdlib/Random/#Reproducibility.
V2 says cross-runtime/platform controls must reproduce the matrix or fail before
outcome access, but does not explicitly say a new-version RNG must regenerate
identical bytes. The current draft takes the conservative native-regeneration
interpretation: all three controls still fail if generated bytes differ. This
is stronger than necessary for every statistic consuming the frozen primary;
it is not a guaranteed RNG property. Reader parity is a distinct measurement;
the current entry validates the primary but has no independent reader-parity
receipt. Replacing the1.13.1 equality gate with measured generation plus required
reader parity needs an explicit reviewed protocol interpretation/amendment.
This trigger correction changes no Julia source or native-owner gate.

Remaining concrete steps: qualify the complete current source and preserve full
lineage; retain the pinned helper paths or independently re-review changed
helpers; run all four genuine native jobs; retain
their raw logs and API-digest-verified artifacts; independently verify every
matrix hash and all three exact-byte comparisons; then freeze the measured
primary matrix before any GSM values. Cross-version equality has not been
measured. A failed or unequal control blocks outcome access; it never replaces
or silently regenerates the primary. The bootstrap artifact grants no source,
software, scientific or paper-result authority and does not replace full package
tests, experiments, the harness audit or CI18.
