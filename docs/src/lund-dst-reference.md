# Published Lund Dst reference

`LundDst2002` implements the fixed four-state recurrence in Appendix A of
[Lundstedt, Gleisner and Wintoft (2002)](https://doi.org/10.1029/2002GL016151).
It consumes hourly Bz in nT, proton density in cm^-3 and speed in km/s.
The normalization is `(Bz/30, n/80-1, V/400-1.5)` and the published initial
state is `(0.39, 0.22, -0.83, 0.14)`. Each update computes all four hidden
coordinates from the previous state, then computes Dst from the new state.
No observed Dst or fitting enters this recurrence.

The scalar call returns an immutable state tuple and a prediction:

```julia
state = LundDst2002.initial_state()
point = LundDst2002.predict_next(state, -5.0, 8.0, 450.0)
continued = LundDst2002.predict_next(point.state, 0.0, 5.0, 400.0)
```

`rollout` consumes aligned vectors and returns one next-label prediction per
input row, together with the final state. Passing that state to the next chunk
preserves the recurrence. Inputs must form a contiguous hourly sequence;
the kernel does not interpolate, repair gaps or infer source availability.

The published model is a one-hour reference. A four- or six-hour forecast needs
an explicitly declared future-driver policy and matched evaluation. Providing
realized future wind is a simulation, not evidence of forecast lead. The
default operational monitor does not call this module. Its example verifies
arithmetic and continuation on deterministic drivers, without an observed-data
skill or novelty claim.

The output-weight triangle inequality gives the envelope
`[-283.015, 104.375]` nT because the four hidden coordinates lie in `[-1,1]`.
This does not establish attainability of either endpoint. It also limits the
fixed model's extrapolation to extreme storms. Source-equation hashes and the
independent-oracle contract are recorded in `docs/LUND_DST_2002_CONTRACT.md`.

```@docs
SolarSINDy.LundDst2002
SolarSINDy.LundDst2002.initial_state
SolarSINDy.LundDst2002.predict_next
SolarSINDy.LundDst2002.rollout
```
