# Operational analog ensemble study API

The analog-ensemble study: feature construction, the analog archive and its retrieval, the
gradient-boosted increment comparator, and the statistics the study reported. The analog ensemble candidate was
a shadow forecast whose confirmatory decision was NO_GO; its serving layer survives as an expert of
the super-learner and is documented in the [operational API](operational-api.md).

## Features

```@autodocs
Modules = [SolarSINDy]
Pages = ["analog_features.jl"]
Private = false
```

## Analog Archive And Retrieval

```@autodocs
Modules = [SolarSINDy]
Pages = ["analog.jl"]
Private = false
```

## Gradient-Boosted Increment Comparator

```@autodocs
Modules = [SolarSINDy]
Pages = ["analog_gbm.jl"]
Private = false
```

## Study Statistics

```@autodocs
Modules = [SolarSINDy]
Pages = ["analog_stats.jl"]
Private = false
```
