# Historical Operational SINDy Core

These artifacts preserve the 21-candidate, 10-active-term core that was served
before the inertial-SINDy migration. The library contains the redundant `n*V^2` pressure
proxy. Current forecasts must not load this directory unless the caller
explicitly requests model version `sindy-core`.

The unqualified files in the parent `data/` directory are the current inertial-SINDy
20-candidate, 11-active-term artifacts.

`live_forecast_log.csv` is the accumulated SINDy core hot log preserved byte-for-byte
before inertial-SINDy accrual began. Its manifest records the row counts and SHA-256 hash.
The two rows without a model-version value predate versioned issuance; all
nonmissing versioned rows are SINDy core (`operational`). They are not inertial-SINDy evidence.
