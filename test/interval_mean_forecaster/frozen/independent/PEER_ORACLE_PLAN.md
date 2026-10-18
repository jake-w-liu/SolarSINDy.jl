# Candidate A independent oracle plan

This scratch-only audit uses no real fitting, new observational data, repository
edits, historical-output edits, or novelty/performance verdict. The implementation
owner is separately creating `C:/Temp/da-20261015/DstIntervalObserver.jl`.

The independent oracle augments the continuous state with interval integrals.
An affine matrix exponential propagates their means. A separate matrix
exponential solves the covariance ODE with the Kronecker operator
`I⊗A + A⊗I`; this avoids using the candidate's closed-form scalar moments and
avoids the growing exponential block in Van Loan at large decay rates.
No eigenvalue or covariance clipping is used.

Fixed numeric checks include exact Brownian limits, the stated lambda=ln2
fixture, lambda values0,1e-14,1e-9,ln2,0.25,7,100,10000, multiple durations,
zero/positive process noise, one/two modes, signed correlated PSD priors,
constant-input semigroup composition, and a hand-solved noisy Brownian
posterior. Primary matrix comparisons will use atol5e-11/rtol5e-11; explicit
hand scalar examples use tighter tolerances where appropriate. These thresholds
are chosen before any candidate comparison. A discrepancy is investigated
with the retained original source and expectations, not resolved by loosening.

For delayed observations, a single augmented state retains all exact historical
bin integrals and the final endpoint. Selection admits one latest eligible
likelihood per bin; one dense Gaussian block solve conditions all selected
observations at once. It is independent of a chronological filter replay and
its checkpoints. Pressure-corrected observation values are `sum(x)`; full-Dst
pressure reconstruction is outside this first kernel contract.

Fixtures will include receipt/verification eligibility at one-millisecond
boundaries, late historical-bin arrivals, replaced revisions, identical repeated
receipts, input-array permutations, changed future values/receipt IDs, and
immutable issued predictions. Every gain setting uses fixed covariances and
the same observations/receipt selection; basis replay must establish affine
means in gains and unchanged covariances.

Before interpreting prefix tests, bind whether the public API accepts full
record arrays and filters them or instead requires caller-sliced known-driver
prefixes. No implicit receipt assumption will be substituted. Equality in
strict actual mode is excluded; separately named hypothetical inclusive mode
must be explicit.

Self-oracle checks initially passed62 assertions. A comment-only correction
changed the stated two-bin Brownian integral cross covariance from5/6 to1/2;
the implemented expected matrix was already1/2 and all62 actual tests passed.
