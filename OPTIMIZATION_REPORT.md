# Duration point-mean computation

The objective is efficient expected affine point designs under the unchanged B
forcing law and A interval operator. Exact wind-state probability masses give
each future E[q] after the nonlinear injection threshold. Linearity then permits
expected gain-basis propagation. State/age/mark identities, duplicate marks,
unequal masses, positive-support failures and exact mass one are preserved.
This does not produce mixture variance, uncertainty intervals or forecast-skill
evidence. No actual data, actual fits or new outcomes were consumed.

## Synthetic warmed measurements

Julia1.13.1, identical synthetic model/current/checkpoint/basis and six-hour source
inputs, BenchmarkTools1.8.0, three warmups and1000single-evaluation samples per row.
The original/selected full rows use module hashes f2dfca48/862c6bf8; the alternative
and mean rows use7720ad10. The mean functions are byte-identical across the latter
two variants; whole-source validation reads slightly different module bytes.
These are prototype comparisons; final selected-source receipts are separate.
Values are medians; median GC fraction was zero for these rows, which does
not imply absence of GC in every sample.

| Computation | Median ms | Allocated MB | Allocations |
|---|---:|---:|---:|
| Original exhaustive complete forecast |19.8903|24.349903|495134|
| Selected corrected exhaustive complete forecast |20.11685|24.450518|495188|
| Separate mean-only complete forecast |3.1132|1.326304|17532|
| Return-widened alternative exhaustive complete forecast |20.4909|24.452230|495213|
| Return-widened alternative affine design |7.8396|7.133796|157131|
| Mean-only affine design |2.18485|1.119312|17259|

The mean-only complete computation is about6.46times faster with94.6%less allocated
memory than the selected exhaustive computation on this fixture. This is not an
actual-panel timing or forecast-accuracy comparison. The exhaustive correction is
a cold-compilation remedy, with approximately1%higher warmed median time than the
original in this run. The affine-return-widened alternative was not adopted: it
provided no clear combined cold benefit and had a higher warmed median.

## Verification and limits

The exact pre-code Fraction/Brownian fixtures precede implementation. Direct
public h1:6,N/S cold controls and exact old/new summaries/components/path order/
metadata comparisons passed both native runtimes before adaptation. The original
source is retained as a SHA-pinned reference. All-corner focused controls cover
the three tau pairs and three sigmas, both betas/starts, ages2/48/49, horizons1:6,
current-q/centroid equality, covariance-domain boundaries and missing support.
Final-source focused/docs/mutation/profile receipts are maintained separately.
Root-owned complete Pkg.test/experiments/harness/CI gates remain required.

CPU and allocation profiles identify source/dependency verification and JSON
model identity as substantial remaining mean-only cost, followed by exact mass
addition and expectation arithmetic. Full evaluation remains dominated by branch
components and BigFloat mixture work. No validation cache or bypass was added.
Type diagnostics show the deliberate widened evaluator result boundary; no
universal type-stability or full-optimization claim is made. Output containers
are immutable; local Dict/Vector workspaces are call-owned, and the unchanged A
precision lock guards BigFloat operations.


## B schema correction resource scope

The verified correction strengthens metadata grammar and independent readback;
it is not presented as a performance optimization. Fresh4096-row/full36 runs on
both runtimes preserve all90 numerical binary files exactly and retain zero
reader fits. Repeated full-source validation and panel encoding remain measured
work. Earlier counter-only comparisons establish repeated linear-sized work,
not a complete complexity or bottleneck attribution. The actual72-panel joint
selector profile remains required before choosing resource changes or scaling
to the archival training population. Full combined-source resource and startup
checks remain open; concurrency-qualified timings are not universal latency
guarantees. No actual archival fits were started through these checks.


## Matched342 collector, replay and row-storage resource checks

The original public collector allocated about1.456GB for1024 rows through
repeated concatenation; the original maximum-tree replay allocated about739.65MB
at1024 rows through repeated tuple boxing. A preallocated feature matrix and a
per-call typed edge table preserve the exact feature values, bins, Float32
accumulation order and outputs. Paired original/corrected controls pass3030
comparisons plus74 edge cases on both runtimes before adoption. A tested vector
of vectors alternative was rejected after its measured latency regression.

Final Julia1.12 component measurements give a1024-row feature median17.327ms
with22558931 allocated bytes and maximum-geometry1024-row replay median7.976ms
with736245 bytes. These are concurrency-qualified synthetic measurements, not
full-archive fitting times or a universal latency guarantee. Final Julia1.13
storage controls include the full500000-row bound. Its immutable packed row
index allocates about4.00MB to construct and avoids sample-count specialization.
Ordered4096-index access costs5.5microseconds versus0.1microsecond for the old
tuple fixture; complete encoder checking costs147.9 versus138.05microseconds.
The storage change therefore trades some access time for bounded type and
storage behavior; it is not claimed to improve every operation. Full500000-row
fitting and actual-panel performance are not measured here.


## Temporary-root correction resource scope

Canonicalizing a caller-owned fixture directory repairs portability without
changing any production algorithm or resource ceiling. The complete repaired
component passes both runtimes, including ordinary collector/replay allocation
checks and the full500000-row storage bound. The twelve additional checks
exercise ordinary paths and explicitly rejected aliases; no platform skip is
introduced. No latency or scientific-performance gain is claimed for this
filesystem correction. Earlier source profiles retain their original scope.


## Bounded archival raw-row storage

Every admitted raw record occupies 48 immutable bytes; the unchanged 50000-row
limit bounds one ledger to 2400000 bytes. The selected witness storage remains
unchanged. Exact binary64 and signed-zero oracles precede implementation.
Both package-bound ordinary blank-body calls retain the original 16MiB
allocation ceiling. The SHA reader uses a fixed-buffer IOBuffer route already
checked against 20 independent byte vectors on both runtimes. Diagnostic
timings are concurrency-qualified, and no universal latency or global
multi-station catalog capacity is inferred from these per-body checks.
