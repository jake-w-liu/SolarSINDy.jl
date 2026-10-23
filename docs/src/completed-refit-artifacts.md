# Completed-source ridge and calibration artifacts

`SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts` provides immutable
synthetic research artifacts for completed-source ridge fitting and full
chronological calibration. Every handle reports
`source_bound_provenance_verified=false`. Actual source parsers, selected tree
refits, governed model selection and matched baseline studies remain separate
requirements. A declared role or `purpose=:refitted` does not establish them.

The two ridge conventions are distinct. Mixed8 uses the fixed eight features,
sample standard deviation plus `1e-9`, penalty1 and a penalized intercept.
Ground11 uses the unchanged eleven reference features, sample standard deviation
with zero/nonfinite scales replaced by1, an unpenalized intercept and an explicit
candidate from the five fixed penalties. A selected ground refit requires the
separate verified selection chain and is rejected here.

Only the two synthetic source formats are accepted. A source audit binds approved
body bytes, native parsing and an independent parser reconstruction. A feature
schedule retains every requested issue and rejection without consulting target
projections. A fit input retains the complete accepted/excluded split partition;
label readiness must be strictly before its declared boundary before a target
is queried. Ground11 never opens wind. Natural/intersection tags isolate
identities; a tag alone does not certify common-case membership across methods.

The fixed offline jobs in `research/dbdt_completed_source/` perform actual ridge
computation and independent input, augmented-QR and directed numerical checks.
The owned native receipt records executable/library/argv/configuration and
before/after controls. Its trusted external pin is required separately from the
copy inside the artifact. These receipts describe an audited trusted runner;
they do not prove execution on a hostile host.

Publication requires a new ordinary destination, exclusive member creation and
an exclusive completion marker written last. Existing complete or incomplete
outputs cannot be overwritten or resumed. Bounded canonical JSON rejects unknown
fields, duplicate/noncanonical bytes, unsupported scalars, links, extra members
and invalid dependency graphs. Float64 payloads retain finite hexadecimal bits,
including signed zeros and subnormals.

Data-only loading validates the original native/independent graph, exact member
bytes, source identities, fixed relevant code identities and complete cohort
partition. It does not require a clean current Git checkout or native launch
flags, fit a model, follow recorded source/executable paths, or replace saved
features or scores using another platform's math library. Caller resolution
paths are explicit ordinary local paths. Recorded Windows and POSIX ownership
tuples are validated against their recorded architecture/OS tags; accepting their
metadata is distinct from platform execution qualification.

Calibration derives every prediction from the loaded point, retains all signed
scores in chronology and preserves the exact stable sorted permutation.
Readiness is strict before the declared next issue. October-development and
October-December-final purposes require their respective fixed split bounds;
changing a string cannot upgrade a calibration. A matching numerical runtime
reproduces saved prediction/log-target/score bits. Foreign arithmetic is checked
against the exact-real expression on the same encoded ridge parameters/features,
using the directed normalization-loss and dot/reduction enclosure; authoritative
score and rank bytes are retained.

`completed_calibration` returns a fresh legacy calibration plus a separate
immutable artifact pin. The legacy point identifier is the point manifest SHA.
It remains synthetic and does not upgrade the event adapter's independent
provenance claim or establish coverage, skill, a matched study or a serving path.

```@example refit_namespace
using SolarSINDy
R = SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
@assert !(:ground_cases in names(R))
@assert !(:CompletedRefitArtifacts in names(SolarSINDy.CompletedSourceDBDT))
nothing # hide
```

```@docs
SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts.RefitProtocol
SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts.RidgeRecipe
SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts.open_audited_sources
SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts.seal_feature_schedule
SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts.build_fit_input
SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts.fit_point
SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts.check_ridge_fit
SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts.publish_point
SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts.load_point
SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts.predict_log
SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts.calibrate_point
SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts.publish_calibration
SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts.load_calibration
SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts.completed_calibration
```
