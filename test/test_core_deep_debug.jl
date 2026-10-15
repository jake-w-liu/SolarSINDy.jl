using Test

@testset "Fresh core boundary and numerical regressions" begin
    include("test_conformal_precision.jl")
    include("test_finite_interpolation.jl")
    include("test_parameter_narrowing.jl")
    include("test_assimilation_covariance.jl")
    include("test_metric_precision.jl")
    include("test_forecast_decay_oracle.jl")
    include("test_point_kernel_oracle.jl")
end
