# Published Lund Dst reference

`LundDst2002` implements Appendix A of Lundstedt, Gleisner and Wintoft (2002),
*Operational forecasts of the geomagnetic Dst index*, DOI
[10.1029/2002GL016151](https://doi.org/10.1029/2002GL016151).
The publisher's normalization, update and coefficient images were inspected
before implementation. Their SHA-256 values are, respectively,
`0b4ee4f12db218a827acb6fd29e78e83be01e87db9bcddd10cc43c0701885ad0`,
`f0500acfb62d4e425fdf04c92d6d08db25f0a0b2687d51e8f71852a8512a2007`,
and `2ceb5f38763a3884b0c997871a735e2e9b1c257641ada0816a8a383483955ba9`.

The low-resolution first input glyph was checked against Lundstedt's
[2002 lecture hosted by ESA](https://swe.ssa.esa.int/TECEES/spweather/Alpbach2002/Lundstedt-space%20weather%20forecasting.pdf),
page 6 (PDF SHA-256
`d6ba3c77ee1bab2dc4ff51592e2c3218c78e0ede6c60e70fbb94e4626849ef0b`).
It explicitly maps signed Bz in [-30,+30] nT to [-1,+1]. No half-wave
rectification is added. Density and speed use the explicit Appendix A
coefficients; the slide's prose ranges are not substituted for that equation.
The original author software has not been retrieved. Coordinate conversion is
not implemented or inferred by this arithmetic kernel.

The input order is Bz in nT, proton density in cm^-3, and speed in km/s.
Normalization is `(Bz/30, n/80-1, V/400-1.5)`. Each call computes all four new
hidden coordinates from the old coordinates, then evaluates the output with
the new state and converts it by `150*y-100`. The default initial mean is
`(0.39,0.22,-0.83,0.14)`. Published four-decimal coefficients are used exactly;
no undocumented higher-precision training state is reconstructed.

The scalar API returns an immutable state tuple and prediction. The vector API
preserves input order and returns one next-label output per driver row and the
final state. It rejects nonfinite drivers, negative density/speed, and invalid
latent states. Empty rollout inputs are defined, and chunk continuation is exact.
No hidden state, observed Dst, training, imputation, interpolation, pressure
correction, output clipping, or future-driver continuation is introduced.

This is an hourly arithmetic reference. Source completion, publication/receipt,
contiguous cadence and gap/reset policy belong to a separately verified adapter.
The paper's one-hour model is not a published four-/six-hour forecast. Repeated
steps with assumed drivers require a named extension and matched evaluation.
The output-weight triangle inequality bounds its output by -283.015 and
104.375 nT; this is an envelope, not a claim that every value is attainable.
The bounded output limits extrapolation to extreme storms.

Before the Julia implementation, a scalar Python standard-library oracle using
`math.fsum` and `tanh` fixed 38 state/output transitions from the published
coefficients. The frozen fixture retains its source image hashes. Tests compare
those values, continuation, future-prefix independence, state/input immutability,
empty/error paths and the output envelope. The experiment entry runs a separate
60-hour deterministic driver scenario without reporting observational skill.
No novelty, operational availability or superiority claim follows from these
computational checks.
