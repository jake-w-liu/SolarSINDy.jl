# Prospective CME collection

`prospective_probability_monitor.jl` contains the collector callback loop and
the launcher for an independently reviewed frozen CME producer. Supply the
reviewed snapshot, existing store and freeze key as command-line arguments,
using the exact runtime and startup options required by that freeze. Set
`CME_COLLECTOR_SHA256` to the independently verified collector-file digest.
The source export receipt and all frozen members are checked before acquisition.

Acquisition retries admit known HTTP connection/request transport causes,
including errors preserved by the HTTP timeout boundary in `CapturedException`.
Wrapper traversal is bounded to 32 layers. Local I/O, interruption, unknown
exceptions, source drift and publication failures stop collection for inspection.
The fixed hourly cadence and producer's collection deadline remain in force.
Operational status records acquisition failures; committed scientific records
belong to the unchanged producer. Collection retrieves no outcomes and writes
no forecast scores.

Run generated offline recovery controls with:

```sh
julia --startup-file=no --pkgimages=no --project=. test/test_cme_prospective_monitor.jl
```

The controls preserve the original collector as an exact-byte before oracle,
exercise the installed HTTP timeout boundary, and verify cadence, fatal paths
and exclusive-claim cleanup. A real collector launch additionally needs its
reviewed external snapshot, store, source hashes and completed software gates.
