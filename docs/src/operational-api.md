# Operational API — the served path

The stages that stand behind the published forecast: the versioned operational cores and their
calibration artifacts, the static V2.2 regime stack, and the V2.3 analog serving layer. The stage
that serves the current center is documented in the
[V2.4e super-learner](operational-v24-api.md).

Discovery, data, baseline, metric and utility functions are documented in the
[core API](api.md), forecast issuance and alarms in
[forecasting and alarms](forecast-api.md). The offline research chain behind the V2.2 stack is
documented in the [V2.2 research chain](operational-v22-research-api.md), and the V2.3 study
machinery in the [V2.3 study API](operational-v23-api.md).

## Versioned Cores And Calibration Artifacts

```@autodocs
Modules = [SolarSINDy]
Pages = ["operational_artifacts.jl"]
Private = false
```

## Static V2.2 Stack Serving

```@autodocs
Modules = [SolarSINDy]
Pages = ["operational_v22_serving.jl"]
Private = false
```

## V2.3 Analog Serving

```@autodocs
Modules = [SolarSINDy]
Pages = ["operational_v23_serving.jl"]
Private = false
```
