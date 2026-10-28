# Source-bound point inputs

`research/dbdt_completed_source/point_input_export.jl` exports the exact fixed
split consumed by the point-fit input hash. The numerical exporter is unchanged
from its independently checked prototype. It writes bounded, hashed chunks;
completion appears only after final source and admission checks.

`bundle_reader.py` verifies every exported input value against every scheduled
sealed case record. Its adjacent helpers use only the Python standard library;
they never load a model, open the declared source catalogue, discover data, or
grant permission. All dependency imports are relative to the package and pinned.

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

Generated export grants no actual-data authority. Archival export is confined to
2017 and requires a distinct fixed-split read-only planning grant, a separately
qualified case reread, the exact whole-package qualification, and the qualified
exporter path and hash. It does not grant fitting, discovery, case generation,
provider-receipt claims, or scientific qualification. Generated positive tests
cannot substitute for actual admission or full-size qualification.

`evaluation_binding.py` independently reconstructs the generated-only metadata
connection from a fixed point/calibration pair to a separately sealed evaluation
case. It checks externally pinned opaque model and recipe bytes, training and
evaluation case identities, station, layout, role, runtime, split order and the
eight research helper files. It never loads a model or opens source paths.
This metadata check does not reconstruct source numbers, validate calibration
values, grant archival access, or establish scientific performance. The native
future-source tests separately recompute features and source completion events.
