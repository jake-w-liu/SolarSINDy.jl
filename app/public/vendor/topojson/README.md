# Vendored base map

`north-america_50m.json` is the topojson the plotting library draws the network panel's coastlines,
country borders and subunit outlines from. The library fetches it at draw time and its built-in
source is the plotting CDN; the dashboard's response policy admits this origin only, and an offline
deployment cannot reach that CDN in any case, so the file is served from here and `app.js` sets the
library's `topojsonURL` to `/vendor/topojson/` before the first plot is drawn.

| | |
|---|---|
| file | `north-america_50m.json` |
| retrieved from | `https://cdn.plot.ly/north-america_50m.json` |
| retrieved on | 2026-08-20 |
| bytes | 1003051 |
| sha256 | `1b6a7bcf364f56504d619267489cac105441914b8dc6ce6ead89f04b70b2339a` |
| matching bundle | plotly.js v2.35.2 (`../plotly.min.js`) |

The name is not a choice: the library derives it from the requested geo `scope` and `resolution`
(`"north america"` and `50` in `renderNetwork`) as `scope`-with-hyphens + `_` + `resolution` + `m`.
A panel that changes either value needs the matching file placed here. The app test suite derives
the name from the shipped `app.js` and the shipped bundle and asserts this file exists, so the
mismatch fails a test rather than blanking the map at runtime.
