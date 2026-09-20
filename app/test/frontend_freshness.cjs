// Execute the shipped dashboard against a minimal DOM; no network or test dependencies.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const { test } = require("node:test");
const source = fs.readFileSync(require("node:path").join(__dirname, "../public/app.js"), "utf8");

function dashboard() {
  const elements = {};
  const element = id => elements[id] ||= {
    style: {}, dataset: {}, innerHTML: "", textContent: "", className: "",
    classList: { add() {}, remove() {}, toggle() {} }, addEventListener() {},
  };
  const context = vm.createContext({
    document: { getElementById: element, querySelectorAll: () => [], addEventListener() {} },
    window: { Plotly: { purge() {}, setPlotConfig() {}, async react() {} } },
    Plotly: { purge() {}, setPlotConfig() {}, async react() {} }, console, AbortController,
    fetch: () => new Promise(() => {}),
    setTimeout: () => 0, clearTimeout() {}, setInterval: () => 0,
  });
  vm.runInContext(source, context);
  return { context, element };
}

function upstream() {
  return {
    upstream: {
      available: true,
      solar_wind: {
        speed_kms: 431, density_cm3: 7.3, bz_gsm_nt: -2.8, bt_nt: 6.7,
        mag_time_utc: new Date().toISOString(), plasma_time_utc: new Date().toISOString(),
        mag_source: "SOLAR1", plasma_source: "ACE",
      },
      kp: { value: 8.7 }, scales: { G: "4" }, alerts: [],
    },
    upstream_status: {
      available: true, elevated: false, reasons: [],
      kp_stale: false, scales_stale: false, mag_stale: false, plasma_stale: false,
    },
  };
}

test("upstream readings respect each product's freshness", () => {
  for (const [flag, absent, retained] of [
    ["kp_stale", ["8.7"], ["431", "G4"]],
    ["scales_stale", ["G4"], ["431", "8.7"]],
    ["mag_stale", ["-2.8", "6.7"], ["431", "8.7"]],
    ["plasma_stale", ["431", "7.3"], ["-2.8", "8.7"]],
  ]) {
    const { context, element } = dashboard();
    const payload = upstream();
    payload.upstream_status[flag] = true;
    context.renderUpstream(payload);
    context.renderPipeline(payload, null);
    const stats = element("upstream-stats").innerHTML;
    const caption = element("upstream-caption").innerHTML;
    for (const value of absent) assert.ok(!stats.includes(value), `${flag}: stale ${value} displayed`);
    if (flag === "mag_stale") assert.ok(!element("pipeline").innerHTML.includes("-2.8"));
    if (flag === "plasma_stale") assert.ok(!element("pipeline").innerHTML.includes("431"));
    for (const value of retained) assert.ok(stats.includes(value), `${flag}: fresh ${value} hidden`);
    assert.ok(!caption.includes("Quiet by NOAA scales"));
    assert.match(caption, /stale|unavailable/i);
    assert.equal(element("upstream-badge").textContent, "partial");
  }
});

test("source names come from the feed and are escaped", () => {
  const { context, element } = dashboard();
  const payload = upstream();
  payload.upstream_status.elevated = true;
  payload.upstream_status.reasons = ["Kp storm level"];
  context.renderUpstream(payload);
  const caption = element("upstream-caption").innerHTML;
  assert.ok(caption.includes("SOLAR1") && caption.includes("ACE"));
  assert.ok(!caption.includes("DSCOVR"));
  payload.upstream.solar_wind.mag_source = "<img src=x onerror=BOOM>";
  context.renderUpstream(payload);
  assert.ok(!element("upstream-caption").innerHTML.includes("<img"));
  assert.ok(element("upstream-caption").innerHTML.includes("&lt;img"));
});

test("partial sources cannot imply quiet geomagnetic conditions", () => {
  const { context, element } = dashboard();
  const payload = upstream();
  payload.upstream_status.kp_stale = payload.upstream_status.scales_stale = true;
  context.renderUpstream(payload);
  assert.equal(element("upstream-badge").textContent, "partial");
  assert.ok(!element("upstream-caption").innerHTML.includes("G4"));
  assert.ok(!element("upstream-caption").innerHTML.includes("8.7"));
  payload.upstream_status.elevated = true;
  payload.upstream_status.reasons = ["L1 wind 800 km/s"];
  context.renderUpstream(payload);
  assert.equal(element("upstream-badge").textContent, "ELEVATED");
  payload.upstream_status.available = false;
  context.renderUpstream(payload);
  assert.equal(element("upstream-badge").textContent, "stale");
  assert.equal(element("upstream-stats").innerHTML, "");
});

test("request timeout covers the body after response headers arrive", async () => {
  const { context } = dashboard();
  const timers = new Map();
  let nextTimer = 0, bodyStarted;
  const started = new Promise(resolve => { bodyStarted = resolve; });
  context.setTimeout = callback => { timers.set(++nextTimer, callback); return nextTimer; };
  context.clearTimeout = id => timers.delete(id);
  context.fetch = async (_path, { signal }) => ({
    ok: true,
    json: () => new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => reject(new Error("request aborted")), { once: true });
      bodyStarted();
    }),
  });
  const request = context.fetchJSON("/api/status");
  await started;
  assert.equal(timers.size, 1, "timeout cleared while response body is pending");
  [...timers.values()][0]();
  await assert.rejects(request, /request aborted/);
  assert.equal(timers.size, 0);
  context.fetch = async () => ({ ok: true, json: async () => ({ status: "ok" }) });
  assert.deepEqual(await context.fetchJSON("/api/status"), { status: "ok" });
  assert.equal(timers.size, 0);
  context.fetch = async () => ({ ok: true, json: async () => { throw new Error("invalid JSON"); } });
  await assert.rejects(context.fetchJSON("/api/status"), /invalid JSON/);
  assert.equal(timers.size, 0);
  context.fetch = async () => ({ ok: false, status: 503 });
  await assert.rejects(context.fetchJSON("/api/status"), /503/);
  assert.equal(timers.size, 0);
});

test("a failed status poll clears the previous upstream and verification displays", async () => {
  const { context, element } = dashboard();
  context.renderUpstream(upstream());
  element("calib").innerHTML = "previous verification";
  element("pipeline").innerHTML = "previous pipeline";
  context.fetch = async () => ({ ok: true, json: async () => null });
  await context.refresh();
  assert.equal(element("upstream-badge").textContent, "unavailable");
  assert.equal(element("upstream-stats").innerHTML, "");
  assert.ok(!element("calib").innerHTML.includes("previous verification"));
  assert.ok(!element("pipeline").innerHTML.includes("previous pipeline"));
});

test("an unavailable ground panel clears its previous station identity", async () => {
  const { context, element } = dashboard();
  element("dbdt-station").textContent = "USGS FRD";
  await context.renderDbdt({available:false});
  assert.equal(element("dbdt-station").textContent, "");
});

test("variation nowcasts disclose the same-station uncorrected product", async () => {
  const {context, element} = dashboard();
  const payload = {available:true, station:"FRD", data_type:"variation",
    current_dbdt:5, max30_dbdt:5, current_tier:{level:0,label:"Below 18 nT/min"},
    max30_tier:{level:0,label:"Below 18 nT/min"}, series:[], forecast:null};
  await context.renderDbdt(payload);
  assert.match(element("dbdt-station").textContent,/FRD variation \(uncorrected\)/);
  assert.match(element("dbdt-caption").innerHTML,/same station's uncorrected variation/);
  assert.match(element("dbdt-caption").innerHTML,/No geoelectric estimate or calibrated forecast/);
  assert.doesNotMatch(element("dbdt-caption").innerHTML,/adjusted near-real-time product is provisional/);
  payload.data_type = "adjusted";
  await context.renderDbdt(payload);
  assert.match(element("dbdt-station").textContent,/FRD adjusted \(provisional\)/);
  assert.doesNotMatch(element("dbdt-caption").innerHTML,/same station's uncorrected variation/);
});

test("claim display retains historical failures instead of suggesting only more waiting", () => {
  const { context, element } = dashboard();
  const status = {calibration:{n_verified:1, calibration_shadow:{current_status:"ok"}},
    claim_audit:{available:true, assessment:{generated_utc:"2026-09-19T12:00:00Z",
      integrity:{violations:Array(48).fill("historical failure")},
      marginal:{claim_ready:false, gates:{coverage:false, integrity:false}}}}};
  context.renderCalib(status);
  assert.match(element("calib").innerHTML, /48 historical integrity findings/);
  assert.match(element("calib").innerHTML, /Collecting more days does not remove/);
  assert.match(element("calib").innerHTML, /Unmet checks: coverage, integrity/);
  status.claim_audit.available = false;
  context.renderCalib(status);
  assert.match(element("calib").innerHTML, /assessment is unavailable or stale/);
  assert.doesNotMatch(element("calib").innerHTML, /48 historical/);
});

test("wrapped chart legends stay above the data and date labels reserve space", () => {
  const { context } = dashboard();
  const layout = vm.runInContext("PLOT_LAYOUT()", context);
  assert.equal(layout.legend.yanchor, "bottom");
  assert.ok(layout.legend.y > 1);
  assert.equal(layout.xaxis.automargin, true);
  assert.equal(layout.xaxis.tickangle, 0);
  // Full date labels must also fit the narrow layout; five ticks overlapped.
  assert.equal(layout.xaxis.nticks, 3);
  const styles = fs.readFileSync(require("node:path").join(__dirname, "../public/style.css"), "utf8");
  for (const selector of ["calib", "footer"]) {
    assert.match(styles, new RegExp(`\\.${selector}\\s*\\{[^}]*overflow-wrap:\\s*anywhere`));
  }
  assert.match(styles, /\.card-head\s*\{[^}]*flex-wrap:\s*wrap/);
});

test("unavailable sources cannot carry live pipeline badges", () => {
  const {context, element} = dashboard();
  context.renderPipeline({available:false, upstream_status:{available:false}}, {available:false});
  assert.doesNotMatch(element("pipeline").innerHTML, /tag-live|live nowcast|live \(USGS\)|live \(SWPC\)/i);
  context.renderPipeline({available:true,served_product:"V2.4e",upstream_status:{available:true,mag_stale:false}},
    {available:true,current_dbdt:1,current_tier:{label:"Below 18 nT/min"}});
  assert.equal((element("pipeline").innerHTML.match(/tag-live/g)||[]).length, 3);
});

test("ground and network expose observation timestamps independently", async () => {
  const {context,element} = dashboard();
  const time = "2026-09-20T04:29:00Z";
  await context.renderDbdt({available:true,station:"FRD",data_type:"variation",current_time_utc:time,
    current_dbdt:1,max30_dbdt:2,current_tier:{level:0,label:"quiet"},max30_tier:{level:0},series:[]});
  assert.ok(element("dbdt-caption").innerHTML.includes(time));
  assert.match(element("dbdt-caption").innerHTML,/data-reltime/);
  let plotted;
  context.Plotly.react = async (...args) => { plotted = args; };
  await context.renderNetwork({n_stations:1,stations:[{station:"CMO",name:"College",lon:-148,lat:65,
    max_dbdt:20,tier:{level:1,label:"18 nT/min"},data_type:"adjusted",time_utc:time}]});
  assert.ok(plotted[1][0].hovertext[0].includes(time));
  assert.ok(element("network-caption").innerHTML.includes(time));
});

test("pending ground refresh distinguishes waiting from completed unavailability", async () => {
  const {context,element}=dashboard();
  await context.renderDbdt({available:false,refresh_in_progress:true});
  assert.equal(element("dbdt-badge").textContent,"refreshing");
  assert.match(element("dbdt-caption").textContent,/refresh is in progress/);
  await context.renderDbdt({available:false,stale:true,refresh_in_progress:true});
  assert.equal(element("dbdt-badge").textContent,"stale · refreshing");
  context.renderPipeline({available:true,stale:true,served_product:"V2.4e"},null);
  assert.doesNotMatch(element("pipeline").innerHTML,/live nowcast/);
});

test("notification escalation is deduplicated and permission-bound", () => {
  const {context}=dashboard();
  const notices=[];
  function Notification(title, options) { notices.push({title,...options}); }
  Notification.permission="granted";
  context.Notification=context.window.Notification=Notification;
  const quiet={available:true,threat:{level:0,label:"Quiet"}};
  const watch={available:true,threat:{level:0,label:"Quiet",watch:true,watch_level:2,
    watch_label:"Moderate storm",interval_lower_edge_min_dst_nt:-65}};
  context.browserNotify(quiet);
  context.browserNotify(watch);
  context.browserNotify(watch);
  assert.equal(notices.length,1);
  assert.match(notices[0].body,/-65 nT/);
  Notification.permission="denied";
  context.browserNotify({available:true,threat:{level:4,label:"Extreme storm"}});
  assert.equal(notices.length,1);
});
