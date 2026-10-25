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
