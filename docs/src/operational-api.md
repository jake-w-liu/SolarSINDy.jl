# Operational API — the served path

The stages that stand behind the published forecast: the versioned operational cores and their
calibration artifacts, the static stack regime stack, and the analog ensemble serving layer. The stage
that serves the current center is documented in the
[super-learner](superlearner-api.md).

Discovery, data, baseline, metric and utility functions are documented in the
[core API](api.md), forecast issuance and alarms in
[forecasting and alarms](forecast-api.md). The offline research chain behind the static stack is
documented in the [static stack research chain](static-stack-research-api.md), and the analog ensemble study
machinery in the [analog ensemble study API](analog-api.md).

## Versioned Cores And Calibration Artifacts

```@autodocs
Modules = [SolarSINDy]
Pages = ["operational_artifacts.jl"]
Private = false
```

## Static Stack Serving

```@autodocs
Modules = [SolarSINDy]
Pages = ["static_stack_serving.jl"]
Private = false
```

## Analog Serving

```@autodocs
Modules = [SolarSINDy]
Pages = ["analog_serving.jl"]
Private = false
```
