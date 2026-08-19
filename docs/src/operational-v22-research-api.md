# Operational V2.2 research chain

The offline chain the static V2.2 stack was fitted through: driver reconstruction and arrival
geometry, the boosted and ridge error models, the innovation history, the residual and core-path
layers, and the shadow chain that composed them. None of these stages is on the served path — the
deployed product uses the fitted stack itself, documented in the
[operational API](operational-api.md) — but they are what the stack's provenance rests on.

## Regime Stack And Cell Structure

```@autodocs
Modules = [SolarSINDy]
Pages = ["operational_v22.jl"]
Private = false
```

## Arrival Geometry

```@autodocs
Modules = [SolarSINDy]
Pages = ["operational_v22_arrival.jl"]
Private = false
```

## Driver Reconstruction

```@autodocs
Modules = [SolarSINDy]
Pages = ["operational_v22_driver.jl"]
Private = false
```

## Core Path

```@autodocs
Modules = [SolarSINDy]
Pages = ["operational_v22_core_path.jl"]
Private = false
```

## Boosted Error Model

```@autodocs
Modules = [SolarSINDy]
Pages = ["operational_v22_boost.jl"]
Private = false
```

## Exogenous Error Model

```@autodocs
Modules = [SolarSINDy]
Pages = ["operational_v22_error_exogenous.jl"]
Private = false
```

## Error State Model

```@autodocs
Modules = [SolarSINDy]
Pages = ["operational_v22_error_state.jl"]
Private = false
```

## Innovation History

```@autodocs
Modules = [SolarSINDy]
Pages = ["operational_v22_history.jl"]
Private = false
```

## Residual Layer

```@autodocs
Modules = [SolarSINDy]
Pages = ["operational_v22_residual.jl"]
Private = false
```

## Shadow Chain

```@autodocs
Modules = [SolarSINDy]
Pages = ["operational_v22_shadow_chain.jl"]
Private = false
```
