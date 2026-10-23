# Completed-source minute replay contract

This research module separates archive tags, assumed source availability,
scheduled issue, target centers, label readiness and measured receipt evidence.
It supplies clock and case construction, rather than a fitted forecaster.
The legacy paper tables and live defaults remain separate artifacts.

The explicit analysis convention assigns a ground-center filter 45 seconds
of future support and an OMNI target-arrival interval one minute. The former
follows the centered-filter source declaration and reference INTERMAGNET
support; it does not identify every provider's exact coefficients. NASA states
that HRO minute tags identify interval starts and target arrivals, rather than
spacecraft observation or publication times:

- https://tech-man.intermagnet.org/stable/chapters/oneminuteimos/samplingfiltering.html
- https://omniweb.gsfc.nasa.gov/html/sc_merge_data1.html

These are counterfactual analysis-availability assumptions. Final archived
products do not establish actual historical receipt or revision vintage.
Every constructed case retains `measured_receipts_available=false`.

## Clocks and information sets

An issue lies on a whole UTC minute. With zero extra delay, the latest admitted
ground center and OMNI arrival start are both the preceding minute. Thirty
ground rates end at that ground center. Calendar features belong to the issue,
not the latest source anchor. The target centers are issue+1 through issue+30
minutes, and the target is the largest horizontal vector-difference magnitude
over those centers. Filtering makes their physical supports overlap; this is
not a claim about an unfiltered physical warning interval.

Ground extra delay applies to both historical features and future target
vectors. Label extra delay is additional processing after ground availability.
Each extra delay is an integer number of milliseconds between zero and one
day. The zero-extra-delay label is ready at issue+30 minutes+45 seconds. Its
first scheduled feedback is issue+31 minutes. A sufficient fixed phase period
is the ceiling, in minutes, of 30 minutes+45 seconds+ground extra+label extra.
Unknown or unbounded actual receipts require a different admission contract.

`admit_ground_features` admits the four ground summaries without any wind
input. Its row contains the issue, ground anchor, rate summaries and explicit
availability convention. OMNI and additional label delays do not change the
ground feature row. Ground-only case eligibility therefore does not depend
on wind missingness. Both feature APIs share the checked history-span admission
and summary calculation; mixed admission retains its prior validation order.

`admit_features` reads only admitted history, using `(X,Y)` in nT and wind
`(V,Bz,By,n)` in km/s, nT, nT and cm^-3. It can operate on a source prefix
containing no target values. `admit_target` requires explicit `asof` and checks
readiness before reading a future vector. Missing/invalid histories, wind,
targets and early labels have distinct rejection reasons. Temporal fitting
additionally requires label readiness strictly before the next split issue.

Clock shifts and minute rounding reject Int64 overflow. Input vectors are read
by order rather than assuming their first array index is one. Functions leave
source dictionaries unchanged and hold at most 30 rates per feature/label call.
The complete ground history requires 31 vectors. Feature admission checks its
oldest vector before reading any source key, rejecting an unrepresentable
history even if a dictionary contains keys produced by wrapped arithmetic.

## Independent expectations and verification

The literal clock and feature fixtures precede this implementation. They cover
one-millisecond readiness/split boundaries, year changes, pre-epoch phases,
asymmetric source/label delays, exact source anchors and algebraic polynomial
ground rates. The independent feature example has sample standard deviation
sqrt(310), checked with fixed absolute tolerance 1e-13; clocks, keys, counts,
readiness decisions and the other hand-exact example values use equality.

A retained predecessor failed the stronger same-ground-source label-delay
contract. Another retained reproduction exposed unchecked extreme-date minute
rounding. Both have regression expectations here. Source-clock mutations and
future-value changes must be detected; changing target values must leave issue
features unchanged. Missing and invalid values never become fabricated zeros.
The Float64 source range excludes nonzero scalars that underflow to zero as
well as nonfinite conversions. Signed zeros and representable subnormals remain
valid. In particular, conversion cannot admit an originally negative speed or
density as zero. Literal power-of-two and BigFloat range controls exercise
ground-history, target and wind admission independently.
A separate unbounded-integer fixture fixes nine extreme history cases,
including the first representable issue and its preceding minute under four
ground delays. Omitting the history-span guard must reproduce the failure.

Full package tests, the actual experiment entry, strict documentation, the
unchanged canonical harness and native cross-platform CI remain qualification
requirements. Scientific use additionally requires rebuilding actual cohorts,
refitting changed models, recalibrating, replaying feedback and regenerating
paper evidence. The module does not qualify old numerical results or establish
forecast skill, novelty, actual receipts or operational readiness.
