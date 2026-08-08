# Operational validation evidence

`storm_replay_scored.csv` and `storm_replay_report.md` are the frozen storm-window
replay served by the monitor's `/api/storm_replay` endpoint. Regenerate them with:

```sh
julia --project=. validation/operational/storm_replay.jl
```

Set `SOLARSINDY_OPERATIONAL_OUTPUT_DIR` to choose a different output directory.
The mutable live forecast ledger is stored separately under `var/monitor/`.

The `v2_1_served_holdout_*` files are the frozen evidence snapshot for
complete-hour causal replay of the served Operational V2.1 stack on the
chronological holdout. The scored
forecasts include the 20-candidate/11-active-term core and every served L1 tail
component. Static conformal offsets are shifted from the frozen-tail center to
the complete-hour served center without residual updates from the holdout. This
center shift is evaluated empirically and does not inherit the frozen-center
finite-sample guarantee automatically. The replay does not reconstruct the
fractional subhourly upstream windows available during live issuance. Regenerate
the snapshot with:

```sh
julia --project=. validation/operational/v2_1_served_holdout.jl
```

The summary reports pooled, lead-specific, and activity-regime results. The
audit records split boundaries, artifact hashes, model identity, interval
policy, and continuity checks. These files are immutable package evidence;
fresh runs write to `validation/output/operational/` unless
`SOLARSINDY_OPERATIONAL_OUTPUT_DIR` selects another directory.
