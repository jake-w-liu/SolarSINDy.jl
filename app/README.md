# Space-Weather Threat Monitor

A 100% open-source dashboard over a live geomagnetic-storm (**Dst**) forecaster. It turns the
forecast log into an evidence-bounded threat view: current storm level, the Dst
forecast **with its empirically evaluated 90% target band**, a **rolling forecast-vs-observed track**
(each issued forecast plotted against the observation that later arrived), an explicit lead-time
statement, the scored track record, and the Sun → grid warning chain.

This dashboard ships **as part of [`SolarSINDy.jl`](../)** — it is the operational front-end over
the package's V2.1 forecaster. A Julia REST backend (no web framework — just
`HTTP.jl`) serving a Plotly UI. Single origin, no build step.

> Research tool, not an operational authority. For official alerts use **NOAA SWPC**.

## Quick start

```bash
bin/solarsindy start dashboard # managed background start from the clone root (stop/status/logs)
# or, attached to the terminal:
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
| `SWM_JULIA_THREADS` | `4` | Julia threads used by the shell launchers; one serialized upstream refresh can run while concurrent API requests remain responsive |
| `JULIA_NUM_THREADS` | `4` in Docker | Julia threads for the Docker or direct-Julia launch path; the shell launchers use `SWM_JULIA_THREADS` |
| `SOLARSINDY_LOG` | `../var/monitor/live_forecast_log.csv` | path to the live forecast log |
| `SOLARSINDY_OPERATIONAL_OUTPUT_DIR` | `../validation/output/operational` | regenerated replay artifacts; complete artifacts take priority in the UI |
| `SOLARSINDY_OPERATIONAL_EVIDENCE_DIR` | (auto) | explicit replay-evidence override; missing artifacts fail closed |
| `SWM_WEBHOOK_URL` | (none) | if set, POST an alert on every threat-level change (Slack/Discord/generic JSON) |

Offline use: `./vendor-plotly.sh` downloads Plotly locally; otherwise the page falls back to
the Plotly CDN automatically.

Alerting: with `SWM_WEBHOOK_URL` set, the server re-evaluates the combined alert level (Dst
forecast + target-interval watch + SWPC upstream + ground dB/dt) every 5 min and POSTs a
JSON payload (`{text, level, reasons, ...}`) **only when the level changes** — escalation or
all-clear, never per-poll spam. With the dashboard open, the browser also raises a desktop
notification on escalation (with permission).

The combined integer is an application notification-routing priority, not a physical
cross-calibration between Dst storm classes and ground-dB/dt threshold bands.

Docker: see the header of [`Dockerfile`](Dockerfile).

## API

All endpoints return JSON; the dashboard is served from the same origin.
New clients should read `threat.interval_lower_edge_min_dst_nt`. The value-equivalent
`threat.lower_bound_min_dst_nt` and `threat.worst_credible_dst_nt` keys remain temporarily for
compatibility with older clients.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | complete-cycle health (`ok`, `no_log`, `stale`, or `incomplete`), log age, and server time |
| `GET /api/status` | Dst threat level, lead time, calibration summary, SWPC upstream snapshot |
| `GET /api/forecast` | latest forecast cycle: per-horizon point + 90% target band |
| `GET /api/history?hours=72` | recent scored forecasts (observed vs predicted) |
| `GET /api/swpc` | NOAA SWPC upstream: L1 solar wind, Kp, G/S/R scales, alerts |
| `GET /api/dbdt` | live ground dB/dt nowcast from the provisional USGS adjusted product; selects the first available FRD/CMO feed and reports why the retrospective forecast is disabled |
| `GET /api/dbdt?station=FRD` | exact station-specific dB/dt response, without automatic fallback; unsupported stations or malformed query encodings return HTTP 400 |
| `GET /api/network` | current multi-station USGS dB/dt map |
| `GET /api/storm_replay` | storm-replay results from regenerated outputs or the bundled snapshot |
| `GET /api/alerts` | active alerts + combined overall alert level/reasons |

The live-source endpoints return the last complete cached snapshot immediately and refresh NOAA
or USGS data in the background. A cold cache therefore reports `available=false` until a later poll
rather than holding the dashboard request open on DNS, TLS, or a public-data outage.

## How forecasts are scored

The integrity rules of this project carry into the UI:

- **No bare point forecasts.** Every forecast is shown with its 90% target interval. A
  watch appears when the most negative lower edge among the displayed intervals enters a
  stronger Dst range than the point forecast; it is not a one-sided confidence bound or a
  storm probability. The served-center shift and bounded online update do not retain the
  frozen-center distribution-free guarantee, so coverage is reported empirically.
- **Lead time is stated against physics.** Forecast steps use ballistically propagated L1 forcing
  when the corresponding upstream window has sufficient coverage, then regime-aware Bz/By
  relaxation beyond the measured L1 window. The genuine upstream lead
  for a *new* disturbance is the L1 advection time (~30–60 min). Multi-day
  confident-severity lead needs CME models not yet in this system.
- **Live evaluation is computed from the log.** Coverage and RMSE are recomputed from the
  scored rows every load. The matched RMSE table includes served V2.1, its frozen-tail ablation,
  SINDy v1, persistence, Burton, Burton full, and O'Brien--McPherron on exactly the same
  observed targets. No best method is highlighted before 48 common rows mature, and the
  storm-row count remains visible so quiet-only evidence cannot be mistaken for storm skill.

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

- **Dst forecast**: the project's **V2.1** nowcaster: interpretable discovered sparse equation,
  causal correction, online adaptive-conformal intervals, ballistically propagated L1 forcing,
  regime-aware Bz/By relaxation, and guarded fallback selection. The V2.1
  frozen-tail center remains in the log only for same-row ablation.
- **Solar wind (L1)**: NOAA SWPC real-time products (`rtsw_wind_1m`, `rtsw_mag_1m`) for live
  issuance; the NASA OMNI archive (CDAWeb) is used for offline calibration and historical replay.
  **Dst**: Kyoto WDC (via NOAA SWPC `kyoto-dst`). **Ground dB/dt**: the provisional USGS
  adjusted near-real-time observatory product. The fixed-historical-residual FRD and CMO
  forecasts were trained on archival quasi-definitive ground data and bow-shock-shifted OMNI
  drivers. They are bundled for reproducibility but are not served against the newest unshifted
  L1 values. This fail-closed boundary avoids an unvalidated time-reference and ground-product
  transfer. The retrospective empirical exceedance scores are not per-issue probabilities.
  Archival quality control can revise the live magnetic vectors. Ground dB/dt is a GIC-hazard
  indicator; the displayed 18/42/66/90 nT/min values are the unit-converted
  [Pulkkinen et al. (2013)](https://doi.org/10.1002/swe.20056) threshold magnitudes, not a
  reproduction of that study's nonoverlapping 20-minute protocol or universal grid-risk
  categories. The optional electric-field value uses a generic 1-D reference
  ground and does not estimate GIC or grid impact without a site-specific conductivity model and
  network topology.
- Forecasts are **locked when issued and scored only after the target hour is observed** — the log
  is an honest, immutable track record.

## License

MIT — see [`LICENSE`](LICENSE). 100% open source; open data; dependencies are resolved from
the committed `Manifest.toml` by `Pkg.instantiate()` on first run.
