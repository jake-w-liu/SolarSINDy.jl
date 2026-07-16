# Operational validation evidence

`storm_replay_scored.csv` and `storm_replay_report.md` are the frozen storm-window
replay served by the monitor's `/api/storm_replay` endpoint. Regenerate them with:

```sh
julia --project=. validation/operational/storm_replay.jl
```

Set `SOLARSINDY_OPERATIONAL_OUTPUT_DIR` to choose a different output directory.
The mutable live forecast ledger is stored separately under `var/monitor/`.
