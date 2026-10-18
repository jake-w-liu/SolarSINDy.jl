# Namespace-only adapter for the byte-frozen prototype oracles. No assertion,
# expected value, tolerance, numerical implementation or exception is changed.
module DiskAffineGainFit
using SolarSINDy
const T = SolarSINDy.IntervalMeanForecaster.DstIntervalTraining
const export_fit_input = T.export_fit_input
const solve_fit_input = T._solve_fit_input
end
