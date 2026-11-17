# Source-bound point inputs

`research/dbdt_completed_source/point_input_export.jl` exports the exact fixed
split consumed by the point-fit input hash. The numerical exporter is unchanged
from its independently checked prototype. It writes bounded, hashed chunks;
completion appears only after final source and admission checks.

`bundle_reader.py` verifies every exported input value against every scheduled
sealed case record. Its adjacent helpers use only the Python standard library;
they never load a model, open the declared source catalogue, discover data, or
grant permission. Dependency imports resolve relative to the package; exported
data and artifact identities remain externally pinned.

The input stream uses little-endian dimensions, row indices, Julia DateTime
milliseconds, column-major features, targets, transformed targets, current rates,
and scales. The Julia Unix-epoch offset is 62135683200000 milliseconds. Membership
requires accepted features and labels, a split-contained issue, label completion
strictly before the split stop, and completion no later than the stated as-of
time. Native `log1p` values retain their original bits and must independently
agree with a separate calculation within absolute 2e-15 plus relative 2e-15.

Bounds remain 600000 rows, 128 MiB per stream, 8 MiB per chunk, at most 16 chunks,
and the existing bounded canonical metadata and case-artifact limits. Files and
path components must be ordinary; extra, missing, linked, changed, or incorrectly
pinned members fail verification.

Archival export is confined to 2017 and takes reverified actual cases, an
explicit split and a fresh destination. Its `archive_point_input_binding_v2`
records the case-read admission, request and implementation identities. The
independent reader checks those bindings and every input value without requiring
CI certificates or a planning grant. Historical v1 authority fields remain
provenance only. Generated fixtures exercise formats and numerical checks; they
do not establish archival source reconstruction or forecasting performance.

`evaluation_binding.py` independently reconstructs the generated-only metadata
connection from a fixed point/calibration pair to a separately sealed evaluation
case. It checks externally pinned opaque model and recipe bytes, training and
evaluation case identities, station, layout, role, runtime, split order and the
eight research helper files. It never loads a model or opens source paths.
This metadata check does not reconstruct source numbers, validate calibration
values, grant archival access, or establish scientific performance. The native
future-source tests separately recompute features and source completion events.
