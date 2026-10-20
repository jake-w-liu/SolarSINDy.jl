# Captured HTTP transport failures

Observed production collector21920 stopped after successful cycle24 with an
HTTP.ConnectError showing ECONNRESET through HTTP.Connections.try_with_timeout.
The dead process's exception object was not retained. Do not claim its concrete
inner type was inspected after exit.

The installed ConcurrentUtilities timeout boundary wraps thrown exceptions in
CapturedException. An offline execution of that exact installed boundary,
throwing Base.IOError for ECONNRESET and then constructing HTTP.ConnectError,
confirms that the current classifier returns false. The same direct IOError
inside HTTP.ConnectError returns true. The known transport error becomes fatal
solely because its capture wrapper is not inspected. No network or observations
were used by this reproduction.

Fix only this confirmed classification gap. Unwrap CapturedException inside an
already recognized HTTP connection/request boundary. Preserve direct local I/O
errors, integrity/parse/source failures, unknown exceptions and interruptions as
fatal. Bound traversal to32 wrappers and return false beyond that bound. Do not
change HTTP status rules, TLS verification, model/source review, acquisition
cadence, collection deadline, issue publication or any scientific calculation.

Freeze the original3fa61b68 collector as a test oracle. Independently require
the installed timeout wrapper reproduction to fail classification before and
pass after. Test wrapped interruption/unknown failures, direct local I/O,
bounded nesting, acquisition-failure status then recovery at the original
cadence, source verification outside recovery, fatal issue-publication errors
and exclusive claim cleanup. Run both local Julia versions, full unfiltered
package tests, examples and canonical harness, and all six native CI cells.

The canonical maintained collector belongs in nested SolarSINDy/tools. Keep
the reviewed production snapshot402, original collector and all prior records
unchanged. Operational recovery currently runs the unchanged qualified collector
8900. A later corrected collector deployment requires completed software gates,
the exact new collector hash, fresh source/store checks, preserved append-only
records and explicit recording of any collection gap. Do not backfill, backdate,
retrieve outcomes, score forecasts or alter the frozen March2027 deadline.
