# Space-Weather Threat Monitor

A 100% open-source dashboard over a live geomagnetic-storm (**Dst**) forecaster. It turns the
locked-live forecast log into an honest, calibrated threat view: current storm level, the Dst
forecast **with its calibrated 90% uncertainty band**, a **rolling forecast-vs-observed track**
(each locked forecast plotted against the observation that later arrived), an explicit lead-time
statement, the verified track record, and the Sun → grid warning chain.

This dashboard ships **as part of [`SolarSINDy.jl`](../)** — it is the operational front-end over
the package's forecaster. A Julia REST backend (no web framework — just `HTTP.jl`) serving a
Plotly UI. Single origin, no build step.

> Research tool, not an operational authority. For official alerts use **NOAA SWPC**.

## Quick start

```bash
cd app
./run.sh                       # → http://127.0.0.1:8723  (open in your browser)
./desktop.sh                   # or: launch as a standalone desktop app window
```

First run instantiates the Julia environment (needs Julia ≥ 1.10). `run.sh` serves the
dashboard at the URL; `desktop.sh` additionally opens it as a standalone app window (Chrome/Edge
`--app`, with a default-browser fallback) and stops the backend on exit.

Configuration via environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `SWM_HOST` | `127.0.0.1` | bind address (`0.0.0.0` for LAN) |
| `SWM_PORT` | `8723` | port |
| `SOLARSINDY_LOG` | auto-discovered | path to the live forecast log; if unset, the server looks for `live_forecasts/live_forecast_log.csv` in the package root and then the parent project |
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
| `GET /api/health` | liveness + log path/age |
| `GET /api/status` | Dst threat level, lead time, calibration summary, SWPC upstream snapshot |
| `GET /api/forecast` | latest forecast cycle: per-horizon point + 90% band |
| `GET /api/history?hours=72` | recent verified forecasts (observed vs predicted) |
| `GET /api/swpc` | NOAA SWPC upstream: L1 solar wind, Kp, G/S/R scales, alerts |
| `GET /api/dbdt?station=FRD` | live ground dB/dt nowcast + Pulkkinen tier + exceedances |
| `GET /api/alerts` | active alerts + combined overall alert level/reasons |

## How it stays honest

The integrity rules of this project carry into the UI:

- **No bare point forecasts.** Every forecast is shown with its calibrated 90% interval, and the
  threat "watch" flag is driven by the *worst credible* value within that band, not just the point.
- **Lead time is stated against physics.** Forecast horizons beyond the last complete hour assume
  solar-wind persistence; the genuine upstream lead for a *new* disturbance is the L1 advection
  time (~30–60 min). Multi-day confident-severity lead needs CME models not yet in this system.
- **Calibration is computed from the log, not asserted.** Coverage and RMSE are recomputed from the
  verified rows every load, with the full baseline set (persistence, O'Brien) and a per-method
  breakdown. The live interval method (adaptive conformal) is reported separately from historical
  coverage when it has few verified rows, so the UI never credits it with a number it has not earned.

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

These thresholds are the widely used scheme across the geomagnetic-storm literature
(e.g., Gonzalez et al. 1994, *What is a geomagnetic storm?*, JGR; Loewe & Prölss 1997).
*Verify exact editions/DOIs before formal citation.*

## Data & provenance

- **Dst forecast**: the project's operational **v2** nowcaster (interpretable discovered sparse
  equation + correction + online adaptive-conformal intervals), written hourly to the locked log.
- **Solar wind (L1)**: NASA OMNI (CDAWeb/HAPI). **Dst**: Kyoto WDC. **Ground dB/dt** (future
  layer): USGS/INTERMAGNET.
- Forecasts are **locked when issued and scored only after the target hour is observed** — the log
  is an honest, immutable track record.

## License

MIT — see [`LICENSE`](LICENSE). 100% open source; open data; dependencies pinned via
`Project.toml` compat bounds and resolved by `Pkg.instantiate()` on first run (commit the
generated `Manifest.toml` if you need byte-exact pinning for a deployment).
