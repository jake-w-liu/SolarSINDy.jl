# Space-Weather Threat Monitor

A 100% open-source dashboard over a live geomagnetic-storm (**Dst**) forecaster. It turns the
forecast log into a calibrated threat view: current storm level, the Dst
forecast **with its calibrated 90% uncertainty band**, a **rolling forecast-vs-observed track**
(each issued forecast plotted against the observation that later arrived), an explicit lead-time
statement, the scored track record, and the Sun → grid warning chain.

This dashboard ships **as part of [`SolarSINDy.jl`](../)** — it is the operational front-end over
the package's V2 forecaster. A Julia REST backend (no web framework — just
`HTTP.jl`) serving a Plotly UI. Single origin, no build step.

> Research tool, not an operational authority. For official alerts use **NOAA SWPC**.

## Quick start

```bash
cd app
./run.sh                       # → http://127.0.0.1:8723  (open in your browser)
./desktop.sh                   # or: launch as a standalone desktop app window
```

First run instantiates the committed Julia 1.12.6 environment. `run.sh` serves the
dashboard at the URL; `desktop.sh` additionally opens it as a standalone app window (Chrome/Edge
`--app`, with a default-browser fallback) while keeping the backend attached to the terminal
until you press Ctrl-C.

Configuration via environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `SWM_HOST` | `127.0.0.1` | bind address (`0.0.0.0` for LAN) |
| `SWM_PORT` | `8723` | port |
| `SWM_JULIA_THREADS` | `2` | Julia threads used by the shell launchers so upstream refreshes cannot block API requests |
| `SOLARSINDY_LOG` | `../var/monitor/live_forecast_log.csv` | path to the live forecast log |
| `SOLARSINDY_OPERATIONAL_OUTPUT_DIR` | `../validation/output/operational` | regenerated replay artifacts; complete artifacts take priority in the UI |
| `SOLARSINDY_OPERATIONAL_EVIDENCE_DIR` | (auto) | explicit replay-evidence override; missing artifacts fail closed |
| `SWM_WEBHOOK_URL` | (none) | if set, POST an alert on every threat-level change (Slack/Discord/generic JSON) |

Offline use: `./vendor-plotly.sh` downloads Plotly locally; otherwise the page falls back to
the Plotly CDN automatically.

Alerting: with `SWM_WEBHOOK_URL` set, the server re-evaluates the combined alert level (Dst
forecast + calibrated-band watch + SWPC upstream + ground dB/dt) every 5 min and POSTs a
JSON payload (`{text, level, reasons, ...}`) **only when the level changes** — escalation or
all-clear, never per-poll spam. With the dashboard open, the browser also raises a desktop
notification on escalation (with permission).

Docker: see the header of [`Dockerfile`](Dockerfile).

## API

All endpoints return JSON; the dashboard is served from the same origin.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | complete-cycle health (`ok`, `no_log`, `stale`, or `incomplete`), log age, and server time |
| `GET /api/status` | Dst threat level, lead time, calibration summary, SWPC upstream snapshot |
| `GET /api/forecast` | latest forecast cycle: per-horizon point + 90% band |
| `GET /api/history?hours=72` | recent scored forecasts (observed vs predicted) |
| `GET /api/swpc` | NOAA SWPC upstream: L1 solar wind, Kp, G/S/R scales, alerts |
| `GET /api/dbdt?station=FRD` | live ground dB/dt nowcast + Pulkkinen tier + exceedances |
| `GET /api/storm_replay` | storm-replay results from regenerated outputs or the bundled snapshot |
| `GET /api/alerts` | active alerts + combined overall alert level/reasons |

## How forecasts are scored

The integrity rules of this project carry into the UI:

- **No bare point forecasts.** Every forecast is shown with its calibrated 90% interval, and the
  threat "watch" flag is driven by the *worst credible* value within that band, not just the point.
- **Lead time is stated against physics.** Forecast steps use ballistically propagated L1 forcing
  when the corresponding upstream window has sufficient coverage, then regime-aware Bz/By
  relaxation beyond the measured L1 window. The genuine upstream lead
  for a *new* disturbance is the L1 advection time (~30–60 min). Multi-day
  confident-severity lead needs CME models not yet in this system.
- **Calibration is computed from the log.** Coverage and RMSE are recomputed from the
  scored rows every load, with the full baseline set (pre-upgrade baseline, SINDy v1,
  persistence, O'Brien) and a per-method breakdown. V2 and every baseline are scored on the same
  observed targets, so the UI never compares methods on mismatched samples.

## Threat scale

Threat level uses the standard **Dst storm-intensity classification**, whose primary division
points are **−50 / −100 / −200 nT**, with an extended minor tier at **−30 to −50 nT**:

| Level | Dst (nT) | Label |
|---|---|---|
| 0 | > −30 | Quiet |
| 1 | −30 to −50 | Minor storm |
| 2 | −50 to −100 | Moderate storm |
| 3 | −100 to −200 | Intense storm |
| 4 | < −200 | Extreme storm |

These thresholds follow the classifications used by
[Gonzalez et al. (1994)](https://doi.org/10.1029/93JA02867) and
[Loewe and Prölss (1997)](https://doi.org/10.1029/96JA04020).

## Data & provenance

- **Dst forecast**: the project's **V2** nowcaster: interpretable discovered sparse equation,
  causal correction, online adaptive-conformal intervals, ballistically propagated L1 forcing,
  regime-aware Bz/By relaxation, and guarded fallback selection. The pre-upgrade baseline remains in the log
  only for same-row comparison.
- **Solar wind (L1)**: NOAA SWPC real-time products (`rtsw_wind_1m`, `rtsw_mag_1m`) for live
  issuance; the NASA OMNI archive (CDAWeb) is used for offline calibration and historical replay.
  **Dst**: Kyoto WDC (via NOAA SWPC `kyoto-dst`). **Ground dB/dt**: USGS geomagnetic
  observatory data, with calibrated FRD and CMO forecasts.
- Forecasts are **locked when issued and scored only after the target hour is observed** — the log
  is an honest, immutable track record.

## License

MIT — see [`LICENSE`](LICENSE). 100% open source; open data; dependencies are resolved from
the committed `Manifest.toml` by `Pkg.instantiate()` on first run.
