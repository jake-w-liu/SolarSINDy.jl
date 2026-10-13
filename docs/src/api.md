# Core API

Discovery, data preparation, baselines, metrics and utilities: the layer the forecasts are built
from. Forecast issuance, alarms, intervals and assimilation are documented in
[forecasting and alarms](forecast-api.md). The served operational path is documented in the
[operational API](operational-api.md), the static stack research chain in the
[static stack research chain](static-stack-research-api.md), and the analog ensemble study machinery in the
[analog ensemble study API](analog-api.md).

Each section renders every exported binding of the source file it names, so a new export appears here
without an edit and cannot go missing from the manual.

## Module

```@docs
SolarSINDy
```

## Utilities

```@autodocs
Modules = [SolarSINDy]
Pages = ["utils.jl"]
Private = false
```

## Candidate Libraries

```@autodocs
Modules = [SolarSINDy]
Pages = ["library.jl"]
Private = false
```

## Sparse Identification

```@autodocs
Modules = [SolarSINDy]
Pages = ["sindy.jl"]
Private = false
```

## Data Loading And Preparation

```@autodocs
Modules = [SolarSINDy]
Pages = ["data.jl", "data_pipeline.jl", "data_cleaning.jl"]
Private = false
```

`parse_omni2` is the exported name of the CSV parser; the docstring lives on the aliased function, so
it is listed explicitly rather than through the section above.

```@docs
parse_omni2
```

## Storm Selection

```@autodocs
Modules = [SolarSINDy]
Pages = ["storm_selection.jl"]
Private = false
```

## Baselines

```@autodocs
Modules = [SolarSINDy]
Pages = ["baselines.jl"]
Private = false
```

## Metrics And Performance Statistics

```@autodocs
Modules = [SolarSINDy]
Pages = ["metrics.jl", "performance_statistics.jl"]
Private = false
```
