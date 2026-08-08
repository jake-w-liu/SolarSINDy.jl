# Historical Operational V2.0 Core

These artifacts preserve the 21-candidate, 10-active-term core that was served
before the V2.1 migration. The library contains the redundant `n*V^2` pressure
proxy. Current forecasts must not load this directory unless the caller
explicitly requests model version `v2.0`.

The unqualified files in the parent `data/` directory are the current V2.1
20-candidate, 11-active-term artifacts.

`live_forecast_log.csv` is the accumulated V2.0 hot log preserved byte-for-byte
before V2.1 accrual began. Its manifest records the row counts and SHA-256 hash.
The two rows without a model-version value predate versioned issuance; all
nonmissing versioned rows are V2.0 (`v2`). They are not V2.1 evidence.
