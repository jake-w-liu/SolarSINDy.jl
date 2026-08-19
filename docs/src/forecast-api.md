# Forecasting and alarms

Forecast issuance and rollout, the alarm ladder, conformal intervals, the assimilation filter, and
the real-time feed readers. The library, data and metric layers these are built on are documented in
the [core API](api.md); the versioned operational stages that publish the served forecast are in the
[operational API](operational-api.md).

## Forecasting

```@autodocs
Modules = [SolarSINDy]
Pages = ["forecast.jl"]
Private = false
```

## Alarms

```@autodocs
Modules = [SolarSINDy]
Pages = ["alarm.jl"]
Private = false
```

The cooldown state an alarm evaluation returns is internal but is part of that return value, so it is
documented here.

```@docs
SolarSINDy.AlarmCooldownState
```

## Conformal Intervals

```@autodocs
Modules = [SolarSINDy]
Pages = ["conformal.jl"]
Private = false
```

## Assimilation

```@autodocs
Modules = [SolarSINDy]
Pages = ["assimilation.jl"]
Private = false
```

## Monitoring And Real-Time Feeds

```@autodocs
Modules = [SolarSINDy]
Pages = ["monitor.jl", "realtime.jl"]
Private = false
```
