"use strict";
// Space-Weather Threat Monitor — dashboard logic.
// Fetches the backend JSON API and renders an honest, calibrated view: the forecast is
// always shown with its 90% interval, lead time is stated against the physical L1 ceiling,
// and calibration is reported with full baseline + per-method breakdown (no overclaiming).

const WONG = { obs: "#e69f00", fcst: "#0072b2", band: "rgba(0,114,178,0.20)" };
const TIER_COLORS = ["#2e9e6b", "#c9a227", "#d55e00", "#c0392b", "#8e44ad"];
const PULK = [18, 42, 66, 90];   // Pulkkinen 2013 dB/dt thresholds [nT/min]
const THREAT_CLASSES = ["threat-0","threat-1","threat-2","threat-3","threat-4"];
const THRESHOLDS = [ {y:-30,l:"minor"}, {y:-50,l:"moderate"}, {y:-100,l:"intense"}, {y:-200,l:"extreme"} ];
const REFRESH_MS = 60000;

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const fmt = (x, d=0) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toFixed(d);

async function ensurePlotly() {
  for (let i = 0; i < 120; i++) {
    if (window.Plotly) return true;
    if (window.__plotlyFailed) return false;
    await sleep(80);
  }
  return !!window.Plotly;
}

async function fetchJSON(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`${path} → HTTP ${r.status}`);
  return r.json();
}

function relTime(iso) {
  if (!iso) return "—";
  const t = Date.parse(iso.replace(/Z?$/, "Z"));
  if (Number.isNaN(t)) return iso;
  const sec = Math.round((Date.now() - t) / 1000);
  if (sec < 90) return `${sec}s ago`;
  if (sec < 5400) return `${Math.round(sec/60)} min ago`;
  if (sec < 172800) return `${Math.round(sec/3600)} h ago`;
  return `${Math.round(sec/86400)} d ago`;
}

// Re-render every [data-reltime] element once a second so relative timestamps keep counting
// ("updated 5s ago" → "6s ago" …) between the slower data refreshes — a visible live heartbeat.
// This only re-renders text from stored timestamps; the data fetch still runs on REFRESH_MS.
function tickRelTimes() {
  document.querySelectorAll("[data-reltime]").forEach(el => {
    el.textContent = (el.dataset.relprefix || "") + relTime(el.dataset.reltime);
  });
}

function setError(msg) {
  const b = $("banner-error");
  if (!msg) { b.classList.add("hidden"); return; }
  b.textContent = msg; b.classList.remove("hidden");
}

// ---- renderers -------------------------------------------------------------------------
function renderThreat(st) {
  const el = $("threat");
  THREAT_CLASSES.forEach(c => el.classList.remove(c));
  el.classList.remove("threat-unknown");
  if (!st || st.available === false) {
    el.classList.add("threat-unknown");
    $("threat-level").textContent = "—";
    $("threat-label").textContent = st && st.message ? st.message : "awaiting forecast data";
    return;
  }
  const th = st.threat;
  el.classList.add(`threat-${th.level}`);
  $("threat-level").textContent = th.label.toUpperCase();
  $("threat-label").textContent = th.basis;
  $("cur-dst").textContent = fmt(st.latest_observation ? st.latest_observation.dst_nt : null, 0);
  $("worst-dst").textContent = fmt(th.worst_credible_dst_nt, 0);
  const lt = st.lead_time || {};
  $("horizon").textContent = lt.forecast_horizon_hours != null ? `${fmt(lt.forecast_horizon_hours,1)} h` : "—";

  const wf = $("watch-flag");
  if (th.watch) {
    wf.textContent = `WATCH · 90% band reaches ${fmt(th.worst_credible_dst_nt,0)} nT (${th.watch_label})`;
    wf.classList.remove("hidden");
  } else { wf.classList.add("hidden"); }

  $("model-line").textContent =
    `Model ${st.model_version || "v2"} · live interval method: ${(st.calibration||{}).current_interval_source || "—"} · `
    + `status generated ${relTime(st.generated_utc)} · latest solar wind ${relTime(st.latest_solar_wind_utc)}.`;
}

function thresholdShapes(ymin, ymax) {
  const shapes = [], anns = [];
  for (const t of THRESHOLDS) {
    if (t.y > ymin - 18 && t.y < ymax + 4) {
      shapes.push({ type:"line", xref:"paper", x0:0, x1:1, y0:t.y, y1:t.y,
        line:{ color:"rgba(192,57,43,0.35)", width:1, dash:"dot" } });
      anns.push({ xref:"paper", x:1, y:t.y, xanchor:"right", yanchor:"bottom",
        text:`${t.y} nT · ${t.l}`, showarrow:false, font:{size:9, color:"rgba(192,57,43,0.7)"} });
    }
  }
  return { shapes, anns };
}

const PLOT_LAYOUT = () => ({
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#9fb0cc", size: 12 },
  margin: { l: 52, r: 16, t: 10, b: 40 },
  xaxis: { type: "date", gridcolor: "rgba(120,140,180,0.12)", zeroline:false, title:{text:"UTC", font:{size:11}} },
  yaxis: { title: { text: "Dst [nT]" }, gridcolor: "rgba(120,140,180,0.12)", zerolinecolor:"rgba(120,140,180,0.25)" },
  showlegend: true, legend: { orientation:"h", x:0, y:1.12, font:{size:11}, bgcolor:"rgba(0,0,0,0)" },
  hovermode: "x unified",
});

function observedSeries(history) {
  // Dedup verified rows by target time → one observed Dst per timestamp; sort ascending.
  const m = new Map();
  for (const r of (history.rows || [])) {
    if (r.observed_dst_nt != null && r.target_utc) m.set(r.target_utc, r.observed_dst_nt);
  }
  const xs = [...m.keys()].sort();
  return { x: xs, y: xs.map(t => m.get(t)) };
}

function forecastTrack(history, cutoffMs) {
  // The locked v2 forecast that was issued for each PAST target hour, at the shortest
  // available lead (the most-informed estimate for that hour). One point per timestamp —
  // so the chart shows, continuously, what we predicted vs what actually happened.
  const best = new Map();
  for (const r of (history.rows || [])) {
    if (r.pred_dst_nt == null || !r.target_utc) continue;
    if (cutoffMs && Date.parse(r.target_utc) < cutoffMs) continue;
    const lead = Math.abs(r.horizon_hours == null ? 1e9 : r.horizon_hours);
    const cur = best.get(r.target_utc);
    if (!cur || lead < cur.lead) best.set(r.target_utc, { lead, pred: r.pred_dst_nt });
  }
  const xs = [...best.keys()].sort();
  return { x: xs, y: xs.map(t => best.get(t).pred) };
}

async function renderForecast(forecast, history, status) {
  if (!(await ensurePlotly())) { setError("Could not load the Plotly library (offline and no vendored copy)."); return; }
  const cap = $("forecast-caption");
  $("interval-badge").textContent = forecast && forecast.interval_source ? `interval: ${forecast.interval_source}` : "—";
  if (!forecast || !forecast.horizons || forecast.horizons.length === 0) {
    Plotly.purge("forecast-plot"); cap.textContent = "No active forecast cycle in the log yet."; return;
  }
  const H = forecast.horizons;
  const anchorT = forecast.anchor_dst_time_utc, anchorY = forecast.anchor_dst_nt;
  const fx = H.map(h => h.target_utc);
  const pred = H.map(h => h.pred_dst_nt);                       // v2 frozen-driver (reference)
  const served = H.map(h => h.served_dst_nt != null ? h.served_dst_nt : h.pred_dst_nt);   // promoted: v2 + L1 look-ahead
  const lo = H.map(h => h.served_ci05_dst_nt != null ? h.served_ci05_dst_nt : h.ci05_dst_nt);  // served 90% band
  const hi = H.map(h => h.served_ci95_dst_nt != null ? h.served_ci95_dst_nt : h.ci95_dst_nt);

  // observed (past) context, last ~36 h. Merge the verified-row observations with the log's recent latest_dst
  // feed so the line is continuous up to the forecast anchor (verified rows lag by the verification delay).
  const obsMap = new Map();
  for (const r of (history.rows || [])) if (r.observed_dst_nt != null && r.target_utc) obsMap.set(r.target_utc, r.observed_dst_nt);
  for (const r of (forecast.recent_observed || [])) if (r.observed_dst_nt != null && r.target_utc) obsMap.set(r.target_utc, r.observed_dst_nt);
  const obsX = [...obsMap.keys()].sort();
  const obs = { x: obsX, y: obsX.map(t => obsMap.get(t)) };
  const cutoff = Date.now() - 36*3600*1000;
  const oi = obs.x.map((t,i)=>i).filter(i => Date.parse(obs.x[i]) >= cutoff);
  const ox = oi.map(i=>obs.x[i]), oy = oi.map(i=>obs.y[i]);

  // what we FORECAST for those same past hours (locked v2 prediction + its 90% band),
  // so the chart continuously shows prediction vs realized observation
  const track = forecastTrack(history, cutoff);

  // lines connect from the anchor observation
  const px = (anchorT ? [anchorT] : []).concat(fx);
  const py = (anchorT ? [anchorY] : []).concat(pred);          // v2 reference
  const svy = (anchorT ? [anchorY] : []).concat(served);       // promoted served forecast

  // sub-hour model trajectory (display only): served forecast integrated at sub-hour steps
  const traj = (forecast.subhour_trajectory || []).filter(p => p && p.dst_nt != null);
  const trajX = traj.map(p => p.target_utc), trajY = traj.map(p => p.dst_nt);

  const ally = [...oy, ...lo, ...hi, ...py, ...svy, ...trajY, ...track.y]
    .filter(v => v != null && !Number.isNaN(v));
  const ymin = ally.length ? Math.min(...ally) : -50, ymax = ally.length ? Math.max(...ally) : 20;
  const { shapes, anns } = thresholdShapes(ymin, ymax);
  // "now" / issue boundary
  if (forecast.issue_time_utc) {
    shapes.push({ type:"line", x0:forecast.issue_time_utc, x1:forecast.issue_time_utc, yref:"paper", y0:0, y1:1,
      line:{ color:"rgba(78,161,255,0.5)", width:1, dash:"dash" } });
    anns.push({ x:forecast.issue_time_utc, y:1, yref:"paper", yanchor:"bottom", xanchor:"left",
      text:"forecast issued", showarrow:false, font:{size:9, color:"rgba(78,161,255,0.8)"} });
  }

  const traces = [];
  // forward 90% band (current cycle). The past interval is summarized cleanly by the
  // calibration panel (coverage) rather than a per-hour band here, which mixes leads and
  // reads as visual noise; the dotted past-forecast line below carries the prediction.
  traces.push({ x: fx, y: hi, mode:"lines", line:{width:0}, hoverinfo:"skip", showlegend:false });
  traces.push({ x: fx, y: lo, mode:"lines", line:{width:0}, fill:"tonexty", fillcolor:WONG.band,
    name:"90% interval", hoverinfo:"skip" });
  // what we forecast for the past hours (dotted), drawn under the observed reality
  if (track.x.length) traces.push({ x: track.x, y: track.y, mode:"lines",
    name:"Forecast (past, locked)", line:{color:WONG.fcst, width:1.6, dash:"dot"}, opacity:0.9,
    hovertemplate:"predicted %{y:.0f} nT<extra></extra>" });
  // observed reality (on top)
  if (ox.length) traces.push({ x: ox, y: oy, mode:"lines+markers", name:"Observed Dst",
    line:{color:WONG.obs, width:2}, marker:{size:5} });
  // v2 (frozen-driver) reference, thin dashed.
  traces.push({ x: px, y: py, mode:"lines", name:"v2 (frozen-driver)",
    line:{color:"rgba(0,114,178,0.45)", width:1.4, dash:"dash"}, opacity:0.85,
    hovertemplate:"v2 %{y:.0f} nT<extra></extra>" });
  // PROMOTED served forecast = v2 + L1 look-ahead, drawn at 15-min SUB-HOUR resolution when the trajectory is
  // available (falls back to the hourly points). The forecast line ITSELF carries the sub-hour resolution, so
  // zooming the forecast region resolves it into 4 points/hour.
  const fcx = trajX.length ? [anchorT].concat(trajX) : px;
  const fcy = trajX.length ? [anchorY].concat(trajY) : svy;
  traces.push({ x: fcx, y: fcy, mode:"lines", name:"Forecast Dst (v2 + L1 look-ahead, 15-min)",
    line:{color:WONG.fcst, width:2.6}, hovertemplate:"forecast %{y:.1f} nT<extra></extra>" });
  // markers at the issued hourly horizons (the scored targets), drawn on top of the sub-hour line.
  traces.push({ x: px, y: svy, mode:"markers", name:"issued horizons",
    marker:{size:8, color:WONG.fcst, line:{color:"#0b1020", width:1.2}},
    hovertemplate:"issued %{y:.1f} nT<extra></extra>" });

  const layout = Object.assign(PLOT_LAYOUT(), { shapes, annotations: anns });
  // Default the view to the forecast window (recent observed + the forecast/sub-hour), so the 15-min trace is
  // front-and-centre instead of a thin slice inside 36 h of history. Autoscale (home button) restores full range.
  const fEnd = trajX.length ? trajX[trajX.length-1] : (fx.length ? fx[fx.length-1] : null);
  const vStart = anchorT || (fx.length ? fx[0] : null);
  if (fEnd && vStart) {
    layout.xaxis = Object.assign({}, layout.xaxis, {
      range: [new Date(Date.parse(vStart) - 6*3600*1000).toISOString(),
              new Date(Date.parse(fEnd) + 30*60*1000).toISOString()], autorange: false });
  }
  await Plotly.react("forecast-plot", traces, layout, {displayModeBar:true, displaylogo:false, scrollZoom:true, responsive:true});

  const src = forecast.interval_source || "—";
  cap.innerHTML = `Solid blue: the served forecast (v2 + L1 look-ahead) issued <span data-reltime="${forecast.issue_time_utc}">${relTime(forecast.issue_time_utc)}</span> from solar wind through `
    + `<span data-reltime="${forecast.latest_solar_wind_utc}">${relTime(forecast.latest_solar_wind_utc)}</span>. The L1 look-ahead drives the near term with the 1-min upstream wind already measured at L1; beyond it the driver is held constant. `
    + `Shaded: the calibrated 90% interval (${src}); the severity scale uses its lower bound. `
    + `Dashed blue: the v2 frozen-driver reference — the solid line tracks it except where fresh upstream wind sharpens the first hours; severity uses the deeper of the two so the look-ahead never under-warns. `
    + `Dotted blue: forecasts already locked for past hours, plotted against the observed Dst (orange) that has since arrived. `
    + `The vertical dashed line marks the latest issue time; horizontal dotted lines mark Dst storm tiers. Genuine new-disturbance lead is the L1 transit (~30–60 min).`;
}

async function renderHistory(history) {
  if (!(await ensurePlotly())) return;
  const cap = $("history-caption");
  const rows = (history.rows || []).filter(r => r.observed_dst_nt != null && r.pred_dst_nt != null);
  $("cov-badge").textContent = history.coverage_90 != null ? `coverage ${fmt(history.coverage_90,3)}` : "—";
  if (rows.length === 0) { Plotly.purge("history-plot"); cap.textContent = "No verified forecasts in this window yet."; return; }

  const obs = observedSeries(history);
  const inside = rows.filter(r => r.inside_90ci === true);
  const outside = rows.filter(r => r.inside_90ci === false);
  const pt = (rs, color, name) => ({ x: rs.map(r=>r.target_utc), y: rs.map(r=>r.pred_dst_nt),
    mode:"markers", name, marker:{size:5, color, opacity:0.8} });

  const traces = [
    { x: obs.x, y: obs.y, mode:"lines", name:"Observed Dst", line:{color:WONG.obs, width:2} },
    pt(inside, "rgba(46,158,107,0.85)", "forecast · inside 90%"),
    pt(outside, "rgba(192,57,43,0.9)", "forecast · outside 90%"),
  ];
  const ally = [...obs.y, ...rows.map(r=>r.pred_dst_nt)].filter(v=>v!=null);
  const { shapes, anns } = thresholdShapes(Math.min(...ally), Math.max(...ally));
  const layout = Object.assign(PLOT_LAYOUT(), { shapes, annotations: anns });
  layout.margin.t = 10;
  await Plotly.react("history-plot", traces, layout, {displayModeBar:true, displaylogo:false, scrollZoom:true, responsive:true});

  cap.innerHTML = `Last ${fmt(history.hours,0)} h of forecasts, scored after observation. `
    + `Green = observation fell inside the 90% interval, red = outside. `
    + `Empirical coverage ${fmt(history.coverage_90,3)} vs nominal 0.90.`;
}

function renderCalib(status) {
  const c = (status && status.calibration) || {};
  const el = $("calib");
  if (!c || c.n_verified == null || c.n_verified === 0) { el.innerHTML = `<p class="caption">Calibration accrues once forecasts are verified.</p>`; return; }
  const bse = (v) => v == null ? "—" : `${fmt(v,2)}`;
  const rows = [
    ["Operational v2", c.rmse_nt, true],
    ["Persistence", c.rmse_persistence_nt, false],
    ["O'Brien (physics)", c.rmse_obrien_nt, false],
  ];
  // honest highlight: best (lowest) RMSE among the three
  const vals = rows.map(r => r[1]).filter(v => v != null);
  const best = vals.length ? Math.min(...vals) : null;

  let html = `<div class="big">
      <div class="stat"><div class="v">${fmt(c.coverage_90,3)}</div><div class="k">90% coverage</div></div>
      <div class="stat"><div class="v">${bse(c.rmse_nt)}</div><div class="k">v2 RMSE nT</div></div>
      <div class="stat"><div class="v">${c.n_verified}</div><div class="k">verified</div></div>
    </div>
    <table><thead><tr><th>point forecast</th><th>RMSE [nT]</th></tr></thead><tbody>`;
  for (const [name, v, isV2] of rows) {
    const win = (v != null && best != null && Math.abs(v-best) < 1e-9) ? ` class="win"` : "";
    html += `<tr><td>${name}</td><td${win}>${bse(v)}</td></tr>`;
  }
  html += `</tbody></table>`;

  // honest interval-method note
  const liveSrc = c.current_interval_source || "—";
  const nLive = c.n_verified_current_source != null ? c.n_verified_current_source : 0;
  let note = `Live intervals use <strong>${liveSrc}</strong> (online, distribution-free). `;
  if (nLive === 0) note += `Its forecasts are still pending verification (0 scored so far); the coverage above is over all `
    + `${c.n_verified} verified forecasts, mostly the prior interval method. `;
  if (c.deepest_obs_dst_nt != null) note += `This live period has been geomagnetically quiet — deepest observed Dst `
    + `${fmt(c.deepest_obs_dst_nt,0)} nT, ${c.n_storm_verified} storm hour(s) below −50 nT — so storm-time calibration is not yet stress-tested.`;
  html += `<div class="note">${note}</div>`;

  if (c.by_source && c.by_source.length > 1) {
    html += `<table><thead><tr><th>interval method</th><th>n</th><th>coverage</th></tr></thead><tbody>`;
    for (const b of c.by_source) html += `<tr><td>${b.source}</td><td>${b.n}</td><td>${fmt(b.coverage_90,3)}</td></tr>`;
    html += `</tbody></table>`;
  }
  el.innerHTML = html;
}

function renderUpstream(status) {
  const up = status.upstream, us = status.upstream_status;
  const badge = $("upstream-badge"), stats = $("upstream-stats"), alertsEl = $("swpc-alerts"), cap = $("upstream-caption");
  if (!up || up.available === false) {
    badge.textContent = "unavailable"; badge.style.color = "var(--ink-mute)";
    stats.innerHTML = ""; alertsEl.innerHTML = "";
    cap.textContent = "NOAA SWPC feeds are currently unreachable; the Dst forecast above is unaffected.";
    return;
  }
  const sw = up.solar_wind || {};
  const kp = up.kp ? up.kp.value : null;
  const g = up.scales ? up.scales.G : null;
  const elevated = us && us.elevated;
  badge.textContent = elevated ? "ELEVATED" : "quiet";
  badge.style.color = elevated ? "var(--t2)" : "var(--good)";

  const bz = sw.bz_gsm_nt;
  // Highlight Bz only when strongly southward (< -10 nT, the geoeffective threshold) — mild
  // southward is not alarming, so do not color it like a storm.
  const bzStyle = (bz != null && bz < -10) ? "color:var(--t2)" : "";
  const stat = (v, k, unit = "", style = "") =>
    `<div class="ustat"><span class="v" style="${style}">${v == null ? "—" : v}${unit}</span><span class="k">${k}</span></div>`;
  stats.innerHTML =
    stat(fmt(sw.speed_kms, 0), "L1 wind", " km/s") +
    stat(fmt(bz, 1), "Bz GSM", " nT", bzStyle) +
    stat(fmt(sw.bt_nt, 1), "Bt", " nT") +
    stat(fmt(sw.density_cm3, 1), "density", " cm⁻³") +
    stat(fmt(kp, 1), "Kp", "") +
    stat(g == null ? "—" : "G" + g, "NOAA scale", "");

  const al = up.alerts || [];
  alertsEl.innerHTML = al.length
    ? `<div class="alabel">Recent SWPC alerts</div>` + al.slice(0, 4).map(a =>
        `<div class="arow"><span class="apid">${a.product_id || ""}</span><span class="atime">${relTime(a.issue_utc)}</span><span class="asum">${a.summary || ""}</span></div>`).join("")
    : "";

  let c = `L1 solar wind <span data-reltime="${sw.mag_time_utc}">${relTime(sw.mag_time_utc)}</span> (DSCOVR, ~30–60 min upstream of Earth). `;
  if (elevated) c += `<strong style="color:var(--t2)">Elevated:</strong> ${us.reasons.join("; ")}.`;
  else c += `Quiet by NOAA scales (G${g}, Kp ${fmt(kp, 1)}); strong southward Bz or high wind speed would raise this.`;
  cap.innerHTML = c;
}

async function renderDbdt(dbdt) {
  const badge = $("dbdt-badge"), stats = $("dbdt-stats"), cap = $("dbdt-caption"), stn = $("dbdt-station");
  if (!dbdt || dbdt.available === false) {
    badge.textContent = "unavailable"; badge.style.color = "var(--ink-mute)";
    stats.innerHTML = ""; if (window.Plotly) Plotly.purge("dbdt-plot");
    cap.textContent = "USGS ground-magnetometer feed currently unreachable — it throttles intermittently; this panel fills in automatically on the next refresh once the feed responds. The Dst forecast above is unaffected.";
    return;
  }
  stn.textContent = "· USGS " + dbdt.station;
  const ct = dbdt.current_tier, mt = dbdt.max30_tier;
  const col = (lvl) => lvl == null ? "var(--ink-soft)" : TIER_COLORS[lvl];
  badge.textContent = ct.label; badge.style.color = col(ct.level);
  const stat = (v, k, unit, color) =>
    `<div class="ustat"><span class="v" style="color:${color}">${v == null ? "—" : v}${unit}</span><span class="k">${k}</span></div>`;
  let statsHtml =
    stat(fmt(dbdt.current_dbdt, 1), "current dB/dt", " nT/min", col(ct.level)) +
    stat(fmt(dbdt.max30_dbdt, 1), "30-min max", " nT/min", col(mt.level));
  const ge = dbdt.geoelectric;
  if (ge) {
    const gcol = col(ge.tier.level);
    statsHtml += stat(fmt(ge.max_vkm, 2), "geoelectric (30-min max)", " V/km", gcol) +
                 stat(ge.tier.label, "GIC risk", "", gcol);
  }
  stats.innerHTML = statsHtml;

  // calibrated next-30-min FORECAST (paper3 online conformal), when both feeds are live
  const fcEl = $("dbdt-forecast"), fc = dbdt.forecast;
  if (fc) {
    const exc = fc.exceedance.map(e => {
      const lvl = Math.min(PULK.indexOf(e.threshold) + 1, 4);
      const col = e.prob >= 0.5 ? TIER_COLORS[lvl] : "var(--ink-mute)";
      return `<span class="exc"><b style="color:${col}">${Math.round(e.prob * 100)}%</b> &gt;${e.threshold}</span>`;
    }).join("");
    fcEl.innerHTML = `<span class="fc-label">forecast · next ${fc.horizon_min} min</span>`
      + `<span class="fc-pt">${fmt(fc.point_dbdt, 1)} nT/min</span>`
      + `<span class="fc-ub">90% ≤ ${fmt(fc.ub90_dbdt, 1)}</span>`
      + `<span class="fc-exc">P(exceed): ${exc}</span>`;
  } else { fcEl.innerHTML = ""; }

  if (await ensurePlotly() && dbdt.series && dbdt.series.length) {
    const x = dbdt.series.map(s => s.t), y = dbdt.series.map(s => s.dbdt);
    const ymax = Math.max(22, 1.2 * Math.max(...y));
    const shapes = [], anns = [];
    PULK.forEach((thr, i) => {
      if (thr <= ymax) {
        shapes.push({ type:"line", xref:"paper", x0:0, x1:1, y0:thr, y1:thr, line:{ color:TIER_COLORS[i+1], width:1, dash:"dot" } });
        anns.push({ xref:"paper", x:0, y:thr, xanchor:"left", yanchor:"bottom", text:`${thr} nT/min`, showarrow:false, font:{ size:9, color:TIER_COLORS[i+1] } });
      }
    });
    const layout = Object.assign(PLOT_LAYOUT(), { shapes, annotations: anns, showlegend: false });
    layout.yaxis.title.text = "dB/dt [nT/min]"; layout.yaxis.range = [0, ymax]; layout.margin.t = 8;
    await Plotly.react("dbdt-plot", [{ x, y, mode:"lines", line:{ color:"#4ea1ff", width:2 }, fill:"tozeroy", fillcolor:"rgba(78,161,255,0.12)" }], layout, { displayModeBar:false, responsive:true });
  }
  let capHtml = `Observed horizontal ground d<i>B</i>/d<i>t</i> = √(Δ<i>X</i>²+Δ<i>Y</i>²) per minute at USGS ${dbdt.station} (mid-latitude). `
    + `Dotted lines are the Pulkkinen et al. (2013) GIC alert thresholds (18/42/66/90 nT/min). This is a <strong>nowcast</strong> of the current GIC driver, not a forecast.`;
  if (ge) capHtml += ` Geoelectric field <i>E</i> (the quantity that drives GIC) via the plane-wave method, `
    + `1-D uniform half-space ρ = ${fmt(ge.rho_ohm_m, 0)} Ω·m — a coarse estimate; real GIC also depends on 3-D ground and grid topology.`;
  cap.innerHTML = capHtml;
}

function renderPipeline(status, dbdt) {
  const sw = (status && status.upstream && status.upstream.solar_wind) || {};
  const l1 = sw.available ? `Bz ${fmt(sw.bz_gsm_nt, 1)} nT · ${fmt(sw.speed_kms, 0)} km/s` : "DSCOVR/ACE · OMNI";
  const gic = (dbdt && dbdt.available) ? `${fmt(dbdt.current_dbdt, 1)} nT/min · ${dbdt.current_tier.label}` : "USGS ground mag";
  const stages = [
    { ic:"☉", nm:"Eruption", ds:"Flare / CME launch", tag:"future", tl:"not in this system (T2)" },
    { ic:"🪐", nm:"CME transit", ds:"Heliosphere → arrival", tag:"future", tl:"CME models (T2)" },
    { ic:"🛰", nm:"L1 solar wind", ds:l1, tag:"live", tl:"live (SWPC)" },
    { ic:"🧲", nm:"Dst (ring current)", ds:"This forecaster (v2)", tag:"live", tl:"live nowcast" },
    { ic:"📈", nm:"dB/dt (GIC driver)", ds:gic, tag:"live", tl:"live (USGS)" },
    { ic:"⚡", nm:"GIC / grid", ds:"Transformer stress", tag:"future", tl:"T3/T4" },
  ];
  const tagClass = { live:"tag-live", research:"tag-research", future:"tag-future" };
  $("pipeline").innerHTML = stages.map(s =>
    `<div class="stage"><div class="ic">${s.ic}</div><div class="nm">${s.nm}</div>`
    + `<div class="ds">${s.ds}</div><span class="tag ${tagClass[s.tag]}">${s.tl}</span></div>`).join("");
}

async function renderNetwork(net) {
  const badge = $("network-badge"), cap = $("network-caption");
  if (!net || !net.stations || net.stations.length === 0) {
    badge.textContent = "unavailable"; badge.style.color = "var(--ink-mute)";
    if (window.Plotly) Plotly.purge("network-plot");
    cap.textContent = "USGS network feed currently unreachable — it throttles intermittently and the map fills in automatically on the next refresh once it responds; the single-station panel above is independent.";
    return;
  }
  if (!(await ensurePlotly())) return;
  const S = net.stations;
  const maxLvl = Math.max(...S.map(s => s.tier.level || 0));
  badge.textContent = `${net.n_stations} stations`; badge.style.color = TIER_COLORS[maxLvl];
  const trace = {
    type: "scattergeo", mode: "markers+text",
    lon: S.map(s => s.lon), lat: S.map(s => s.lat),
    text: S.map(s => s.station), textposition: "top center", textfont: { size: 9, color: "#9fb0cc" },
    marker: { size: S.map(s => 10 + Math.min(s.max_dbdt, 60) / 3),
              color: S.map(s => TIER_COLORS[s.tier.level || 0]),
              line: { width: 1, color: "rgba(255,255,255,0.5)" }, opacity: 0.9 },
    hovertext: S.map(s => `${s.name} (${s.station})<br>${s.max_dbdt} nT/min · ${s.tier.label}`),
    hoverinfo: "text",
  };
  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)", margin: { l: 0, r: 0, t: 6, b: 0 }, height: 380,
    geo: { scope: "north america", resolution: 50,
           showland: true, landcolor: "#16203c", showocean: true, oceancolor: "#0b1020",
           showcountries: true, countrycolor: "#2a3550", showsubunits: true, subunitcolor: "#26314e",
           showlakes: false, bgcolor: "rgba(0,0,0,0)", framecolor: "#2a3550",
           lataxis: { range: [14, 73] }, lonaxis: { range: [-172, -58] } },
  };
  await Plotly.react("network-plot", [trace], layout, { displayModeBar: false, responsive: true });
  cap.innerHTML = `30-min max ground d<i>B</i>/d<i>t</i> across USGS observatories (marker size + colour = Pulkkinen tier). `
    + `Auroral-latitude stations (Alaska) respond first and strongest in a storm; elevated mid-latitude stations indicate a severe, low-latitude-reaching event.`;
}

// Browser desktop notification on threat escalation (client-side; complements the server
// webhook). Requests permission once; fires only when the level rises above the last seen.
let _lastNotifiedLevel = null;
function browserNotify(status) {
  if (!("Notification" in window)) return;
  const lvl = (status && status.available) ? status.threat.level : 0;
  if (Notification.permission === "default") Notification.requestPermission().catch(() => {});
  if (Notification.permission === "granted" && _lastNotifiedLevel !== null && lvl > _lastNotifiedLevel) {
    try {
      new Notification("⛬ Space-Weather alert", {
        body: `Threat escalated to ${status.threat.label}. 90% band worst case ${fmt(status.threat.worst_credible_dst_nt, 0)} nT.`,
      });
    } catch (e) { /* notifications best-effort */ }
  }
  _lastNotifiedLevel = lvl;
}

// ---- main refresh loop -----------------------------------------------------------------
async function refresh() {
  // Every endpoint is fetched independently (.catch → null) so a single failed/500 fetch can never reject
  // the whole batch and blank the dashboard. Each panel then renders inside its own guard, so one panel's
  // error does not stop the others. The header banner reflects only whether the core status loaded.
  const [health, status, forecast, history, dbdt, network] = await Promise.all([
    fetchJSON("/api/health").catch(() => null),
    fetchJSON("/api/status").catch(() => null),
    fetchJSON("/api/forecast").catch(() => null),
    fetchJSON("/api/history?hours=72").catch(() => null),
    fetchJSON("/api/dbdt").catch(() => null),
    fetchJSON("/api/network").catch(() => null),
  ]);
  $("health-dot").className = "dot " + (health && health.status === "ok" ? "dot-ok" : "dot-bad");
  if (status) {
    setError(null);
    const upd = $("updated");
    upd.dataset.reltime = status.generated_utc; upd.dataset.relprefix = "updated ";
    upd.textContent = "updated " + relTime(status.generated_utc);
    try { renderThreat(status); browserNotify(status); renderUpstream(status); renderCalib(status); renderPipeline(status, dbdt); }
    catch (e) { console.error(e); }
  } else {
    setError("Backend status unavailable; other panels may still update.");
    $("health-dot").className = "dot dot-bad";
  }
  try { await renderForecast(forecast, history, status); } catch (e) { console.error(e); }
  try { await renderHistory(history); } catch (e) { console.error(e); }
  try { await renderDbdt(dbdt); } catch (e) { console.error(e); }
  try { await renderNetwork(network); } catch (e) { console.error(e); }
}

refresh();
setInterval(refresh, REFRESH_MS);   // fetch fresh data
setInterval(tickRelTimes, 1000);     // keep relative timestamps ticking between fetches
