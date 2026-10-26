# Archival source formats

```@docs
SolarSINDy.ArchiveFormatParsers
```

`ArchiveFormatParsers.parse_ground` accepts IAGA-2002 text for one declared
station and quasi-definitive product. `parse_wind` accepts the five-column OMNI
HAPI CSV order Time, By, Bz, speed and density. Both require explicit whole-minute
start and stop times within 32 days; the ground interval includes the stop and
the wind interval excludes it. One body is limited to 8 MiB and 50,000 raw rows.

`row(body,i)` returns the selected original witness for each sorted unique tag.
`raw_row(body,i)` returns every raw record in source order, including partial
and repeated records. Immutable masks and reason bits distinguish valid values
from missing, unobserved and unusable values. An unavailable payload zero is
interpreted only through its mask. `complete_values` returns `nothing` for an
incomplete vector. Complementary partial records are never combined into a
complete vector, and conflicting valid duplicate components reject the body.

The body, selected-record and raw-record digests bind exact bytes. The readers
accept text and declarations; archive source admission, provider support clocks,
cross-body reconciliation and forecast performance require separate checks.

## Independent reconstruction

```@docs
SolarSINDy.ArchiveFormatAudit
```

`ArchiveFormatAudit.audit_ground` and `audit_wind` reconstruct the same raw
source text through a separate decimal lexer, grouped duplicate checks and
binary encoding. `record` and `raw_record` expose immutable selected and raw
records. `selected_records` and `raw_records` retain their packed bytes, while
`selected_digest` and `raw_digest` bind those bytes. Both readers preserve
provider tags; support clocks, cross-body catalog reconciliation and archive
source admission require separate checks.

## Cross-body catalogs

```@docs
SolarSINDy.ArchiveSourceCatalog
SolarSINDy.ArchiveCatalogAudit
```

`build_catalog` and `audit_catalog` accept one declared stream with at most32
bodies and1600000 raw rows. Canonical body identities make selected original
witnesses independent of input enumeration order. Every raw row participates
in duplicate-consistency checks. A selected witness is an actual complete row
when available, or an original incomplete row; partial rows are never combined.
`witness` and `audit_witness` retain the original line and body identity.
`locate` and `audit_locate` require exact-minute tags and return `nothing` for
absent rows. Missing declared minutes differ from uncovered gaps between bodies.
The catalog retains every original immutable body ledger. It grants no file
permission, provider-clock interpretation, archive admission or forecast claim.

## Completed-source catalog windows

`ArchiveCompletedWindows` admits features through bounded views of the original
catalog witnesses. A feature request reads31completed ground vectors and, for
mixed8, one completed wind vector. Ground11 ignores the wind argument. Partial
selected vectors remain unavailable; the adapter never stitches measurements.

Target access checks strict split readiness and assumed label maturity before
creating its31-vector target view. An immature or split-ineligible request has
an empty source query trace. Window iteration, scans, off-minute queries, queries
outside the declared support and more than256calls per view are rejected.

Returned feature vectors and ordered read traces are immutable tuples. The
independent route uses the independently reconstructed catalog and a separate
window implementation; both share the feature primitive whose numerical
expectations are frozen separately. Source pins retain assumed availability,
unknown historical receipts and false source-bound provenance. These adapters
grant no file permission, actual-source artifact, fit or forecasting qualification.

```@docs
SolarSINDy.ArchiveCompletedWindows
SolarSINDy.ArchiveCompletedWindows.prepare_catalog_issue
SolarSINDy.ArchiveCompletedWindows.evaluate_catalog_label
SolarSINDy.ArchiveCompletedWindowAudit
SolarSINDy.ArchiveCompletedWindowAudit.audit_catalog_issue
SolarSINDy.ArchiveCompletedWindowAudit.audit_catalog_label
```

## Pinned source reconstruction

The offline source job accepts a canonical request with an explicit file list,
byte sizes, SHA256 pins, stream declarations and source intervals. It checks
request shape, paths, stream membership and bounded minute intervals before
reading dataset bytes. Ground-only requests reject wind descriptors. Generated
requests admit only generated fixture files under their own input directory.

Actual-cache execution also requires four caller-supplied path/hash pairs:
authorization, frozen source-admission plan, exact package source proof and
completed software qualification. The loaded worker must occur in that proof.
Recorded paths never launch code or discover additional datasets. Historical
provider receipt availability remains false for this archival-cache route.

Each fresh snapshot preserves original text,48-byte raw records and selected
80-byte catalog witnesses in shards of at most8192rows. Members are bounded by
8MiB, the snapshot by1GiB and the member count by1024. The completion marker is
written last; failed attempts remain incomplete. Primary and independent jobs
reconstruct separately. A completed single-role snapshot grants no independent
source qualification, eligible-case count, fit or forecasting verdict.

```@docs
SolarSINDy.ArchiveCatalogSourceJob
SolarSINDy.ArchiveCatalogSourceJob.Authority
SolarSINDy.ArchiveCatalogSourceJob.reconstruct
```

## Sealed case schedules

`ArchiveCatalogCases` retains every explicitly scheduled issue, including
feature and label exclusions. Features are sealed before labels are requested.
The caller's issue vector is copied before validation and source access.
Schedules contain at most600000 exact-minute, strictly increasing issues.

Each160-byte feature record carries issue and readiness clocks, admission
status, reason, numerical values and the ordered source-support digest. Each
64-byte label record carries the matching clocks, independent admission status,
target and support digest. Rejected rows expose no numerical values. Mixed8
records use zero-filled unused feature columns. Payloads are immutable strings;
`case_arrays` returns separate caller-owned arrays for the accepted intersection.

The label audit checks the sealed ground-source hashes, station, strict split
readiness and assumed maturity. Labels may be available for a feature-rejected
issue; `case_counts` reports those exclusions separately. Joining labels from
another schedule is rejected. These pure APIs perform no file access or model
fit and retain false measured-receipt and source-bound-provenance flags.

```@docs
SolarSINDy.ArchiveCatalogCases
SolarSINDy.ArchiveCatalogCases.CatalogFeatureSchedule
SolarSINDy.ArchiveCatalogCases.CatalogLabelAudit
SolarSINDy.ArchiveCatalogCases.seal_catalog_features
SolarSINDy.ArchiveCatalogCases.audit_catalog_labels
SolarSINDy.ArchiveCatalogCases.feature_row
SolarSINDy.ArchiveCatalogCases.label_row
SolarSINDy.ArchiveCatalogCases.case_counts
SolarSINDy.ArchiveCatalogCases.case_arrays
```

## Source-bound case artifacts

The case job publishes a complete feature schedule before requesting any labels.
It retains every issue and each separate exclusion. Its bounded snapshot reader
checks canonical metadata, ordinary paths, exact membership, sizes and hashes,
then reconstructs the catalogs from preserved original text. Recorded cache
paths are never opened. Feature and label members have at most8192rows; the
existing8MiB/member,1GiB/artifact and1024-member limits remain in force.

Generated requests accept generated snapshots and no actual authority. Archival
requests require a distinct feature-only grant bound to the exact request,
locked schedule plan, eight-job source agreement, loaded package inventory,
complete software qualification and a predeclared floating comparison policy.
A source-reconstruction grant does not authorize features. Neither grant
authorizes model fitting or historical provider receipt claims.

Publication uses fresh exclusive files and writes its completion marker last.
The reader recomputes and compares every feature and label byte before returning
`VerifiedCases`. Split projection checks issue membership and strict readiness
before each persisted target access. It returns owned arrays and performs no fit.

```@docs
SolarSINDy.ArchiveCatalogCaseJob
SolarSINDy.ArchiveCatalogCaseJob.Authority
SolarSINDy.ArchiveCatalogCaseJob.VerifiedCases
SolarSINDy.ArchiveCatalogCaseJob.publish
SolarSINDy.ArchiveCatalogCaseJob.read_verified
SolarSINDy.ArchiveCatalogCaseJob.split_arrays
```

## Comparing native artifact outputs

The pure comparison reader checks caller-supplied binary records across native
jobs. It owns its buffers, validates the entire default-policy issue schedule,
and requires exact status, exclusion, padding, unused-column and support bytes.
Direct wind fields remain bit exact. Other accepted values use a fixed
generated-tested rounding criterion. Its admitted coordinate bound comes from
the archival parser and is not a natural physical maximum.

The canonical policy describes roundoff with an explicit Float64 bit string
and integral bounds with integers, retaining the data-only metadata contract.

The criterion has independent Decimal and MPFR regression checks. It is not a
universal floating-point proof, a source-provenance certificate, or permission
to use an unverified artifact for fitting. The native artifact reader and outer
source/grant bindings remain required. Comparison results contain diagnostics
and exclusion counts, with false authority, provenance and fit flags.

```@docs
SolarSINDy.ArchiveCatalogCaseComparison
SolarSINDy.ArchiveCatalogCaseComparison.comparison_policy
SolarSINDy.ArchiveCatalogCaseComparison.policy_digest
SolarSINDy.ArchiveCatalogCaseComparison.compare_case_bytes
```

## Optional generated scale check

`examples/archive_case_job_scale_synthetic.jl` exercises eight generated tables:
two stations, each with a full year and three separate 31-day periods. It checks
every scheduled row against closed-form periodic increments and records phase
times and native peak process RSS. It retains all inputs, artifacts and partial
failures in a fresh output directory. It reads no observational data and fits no
forecast model.

```sh
julia --project=. examples/archive_case_job_scale_synthetic.jl /fresh/output primary ground-only
```

Use `independent` for the other parser role and `mixed` for the wind-assisted
layout. The output directory must not already exist. This optional scale check
is separate from the short generated example in `examples/experiments.jl`.
