# Source-bound reference import into Candidate A

Pre-implementation plan, 2026-10-18 UTC. This does not qualify an importer,
authorize A fitting, or change the frozen 18-setting recipe or evidence gates.
The existing NEXT_CHAIN_CONTRACT.md remains authoritative.

## Verified gap and starting evidence

The current training entry rejects every reference except a NamedTuple with
`evidence_kind == :synthetic_initial_reference`. Reopened source confirms there
is no serialized real-reference importer. The original L0 reconstruction has
completed all 16 grid models, but independent actual-artifact closure is still
pending; its native completion alone is insufficient for consumption by A.

Current reader source is 9d426ff388d8058451643da4ad024f45f2982d88. Its exact
520-file manifest is READER_COMMITTED_SOURCE_V4.json, SHA256
92bb65549b0a820650a96aad953e44c34c3e05ed5fc42402d76e3bd9e590d825.
Its full local/native release gates remain pending. The original L0 stage
terminal has SHA256
615b326652ad3f13d6db088a17bc993be6106e6ff6cd6ec404e8bf92c540388a;
the reference completion is
a7b578c27778348707e0c8d5f8f88a2122914bf29d5f92efe368b6a47f3193ed.
These identities are provenance, not numerical acceptance of the new artifacts.

## Proposed interface and trust boundary

Add a separate tool module that constructs an immutable artifact-reference
descriptor from absolute roots and caller-pinned original, reconstructed and
independent-reader completion digests. Do not accept a convenient Boolean or
actual-reference NamedTuple tag as proof. Keep the current synthetic entry and
its behavior intact. Refactor only the shared selection arithmetic into a
private helper; a typed importer selection method must reverify its serialized
proof and input/output identities before calling that helper, and again after
selection. The default NamedTuple entry must still reject an actual tag.

The descriptor stores roots and pinned digests, not mutable parsed metadata or
prediction arrays. Rebuild the selection rows from verified serialized bytes at
each call. Returned metadata records the proof/mode and source identities.
Rehashed changes to clocks, source/mode, imputation, grid order, chosen member,
model/scaling/mask files, per-lead keysets or fallback points must be rejected.
Input/output roots remain separate, ordinary, canonical-spelled and bounded;
all referenced reader control and prepared-plan files must be hash-bound.
Reader software qualification, artifact numerical closure and original runtime
execution lineage are distinct checks; never label the data-only replay as an
execution-lineage qualification. L0 admission grants no L1 support.

## Common rows without a joint-cohort shortcut

Construct the exact expected mature selection keyset independently for each
lead from raw censored Dst and checked clocks. The baseline common row index is
the ordered UNION of those two keysets. This union retains every required
lead-specific row; it is never the h4/h6 intersection used for joint gain fitting.
For every union issue, reconstruct both raw target slots, scoreable flags and
completion/receipt clocks, including a non-scoreable or immature opposite slot.
Only an eligible slot requires a finite joined GBM prediction. An ineligible
slot is explicit and must never enter metrics. The 168-hour embargo means all
union-row target slots remain inside the reconstructed pre-selection prefix.
Prove that bound before indexing. Do not read later test labels to fill slots.

Verify the selected initial model/scaling/mask identities and exact per-lead
joined prediction vectors. Confirm persistence fallback equals the required raw
anchor exactly. Keep the original GBM grid metrics separate from the matched
A-join metrics. Build the baseline row identity expected by _check_rows and
_identity; candidate panels must be projected onto exactly this union before
selection. Projection must reject duplicates, missing required keys and
conflicting raw values and may not drop a difficult eligible row.

Keep all existing selection arithmetic: all 18 candidates; both pooled RMSE
constraints without tolerance promotion; fixed normalized storm objective;
exact grid-order ties; explicit missing-storm and no-qualifier results. Do not
manufacture a winner or enable final fitting/serving.

## Independent pre-code oracles and verification

Before implementation, freeze literal small raw panels and expected h4/h6
clocks, eligibility, union keys, missing-opposite-slot flags and fallback values.
Include unequal per-lead cohorts, exact embargo equality and +1 ms exclusion,
no-storm, zero-reference storm error, exact ties, no qualifier, and a candidate
that would win only after incorrectly taking the joint intersection.

Exercise the complete serialized generated-reference reader/importer chain,
using the existing independently verified nonconstant fixture. Synthetic mode
must remain synthetic; relabeling it as actual must reject. Add rehashed
corruptions for all required bindings and post-load disk/descriptor mutation.
Include missing receipt, model/scaling/mask corruption, omitted/duplicate/extra
required keys, nonfinite eligible points, wrong source/latency, relocation,
case/link aliases, bounded malformed input and no successful completion on error.

Run targeted tests with independent expected values, full package tests and
experiments through the canonical harness, and all six native platform/runtime
cells. Measure longer generated-panel memory/time/handles; do not claim bounded
full-history orchestration from a small importer test. Actual importer use
requires its own completed software gates plus the independent actual L0
artifact closure. All 18 initial A fits, selected final refitting, saved model
closure and separate L1 baseline/18 fits remain later mandatory stages.
# Selection subwindow addendum

The reviewed generated reference starts its A selection one hour after the
original training boundary. The artifact route accepts this audited subwindow;
fit-target and receipt maturity remain bounded by the original training boundary
minus 168 hours. The public synthetic NamedTuple route retains its original
equal-boundary requirement. The pre-implementation literal clock oracle
`a_reference_subwindow_clock_oracle_v1.json` distinguishes the legal 168-hour
label from a 167-hour label that would incorrectly pass a subwindow cutoff.

`load_reference` requires separate caller-pinned export and independent-audit
completion digests. Rehashing export metadata cannot replace that audit trust
anchor. These caller pins must come from trusted completed execution records;
arbitrary caller-fabricated pins are not execution attestations.

