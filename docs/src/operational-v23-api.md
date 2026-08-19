# Operational V2.3 study API

The analog-ensemble study: feature construction, the analog archive and its retrieval, the
gradient-boosted increment comparator, and the statistics the study reported. The V2.3 candidate was
a shadow forecast whose confirmatory decision was NO_GO; its serving layer survives as an expert of
the V2.4e super-learner and is documented in the [operational API](operational-api.md).

## Features

```@autodocs
Modules = [SolarSINDy]
Pages = ["operational_v23_features.jl"]
Private = false
```

## Analog Archive And Retrieval

```@autodocs
Modules = [SolarSINDy]
Pages = ["operational_v23_analog.jl"]
Private = false
```

## Gradient-Boosted Increment Comparator

```@autodocs
Modules = [SolarSINDy]
Pages = ["operational_v23_gbm.jl"]
Private = false
```

## Study Statistics

```@autodocs
Modules = [SolarSINDy]
Pages = ["operational_v23_stats.jl"]
Private = false
```
