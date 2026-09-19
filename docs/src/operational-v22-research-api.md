# Operational V2.2 driver and arrival models

Offline regime-stack fitting, driver reconstruction, arrival geometry, and
frozen-core trajectories. See [error and shadow models](operational-v22-errors-api.md)
for residual corrections and their composition, and the
[operational API](operational-api.md) for static-stack inference.

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
