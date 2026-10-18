module IndependentGaussianOracle

using LinearAlgebra

export affine_flow, interval_joint, timeline_joint, gaussian_condition

"""Independent affine ODE/covariance ODE exponential, column-major vec.

No scalar OU moments or candidate filtering formulas occur in this oracle.
The covariance operator has only sums of nonpositive drift eigenvalues in
our fixtures, avoiding the growing block used by the Van Loan construction.
"""
function affine_flow(A::Matrix{Float64}, b::Vector{Float64}, D::Matrix{Float64},
                     delta::Float64, mean::Vector{Float64}, P::Matrix{Float64})
    n = length(mean)
    augmented_mean = zeros(n + 1, n + 1)
    augmented_mean[1:n, 1:n] .= A
    augmented_mean[1:n, n + 1] .= b
    evolved_mean = exp(delta * augmented_mean) * [mean; 1.0]
    covariance_operator = kron(Matrix{Float64}(I, n, n), A) +
                          kron(A, Matrix{Float64}(I, n, n))
    augmented_covariance = zeros(n^2 + 1, n^2 + 1)
    augmented_covariance[1:n^2, 1:n^2] .= covariance_operator
    augmented_covariance[1:n^2, n^2 + 1] .= vec(D)
    evolved_covariance = exp(delta * augmented_covariance) * [vec(P); 1.0]
    return (mean=evolved_mean[1:n], covariance=reshape(evolved_covariance[1:n^2], n, n))
end

"""Joint endpoint and per-mode interval average by augmented state integration."""
function interval_joint(mean, P, lambdas, sigmas, forcing, delta)
    m, covariance = Float64.(mean), Matrix{Float64}(P)
    rates, noise, f = Float64.(lambdas), Float64.(sigmas), Float64.(forcing)
    n, duration = length(m), Float64(delta)
    A = zeros(2n, 2n)
    A[1:n, 1:n] .= -Diagonal(rates)
    A[n+1:2n, 1:n] .= Matrix{Float64}(I, n, n)
    D = zeros(2n, 2n)
    D[1:n, 1:n] .= Diagonal(noise .^ 2)
    prior = zeros(2n, 2n)
    prior[1:n, 1:n] .= covariance
    result = affine_flow(A, [f; zeros(n)], D, duration, [m; zeros(n)], prior)
    scale = Diagonal([ones(n); fill(inv(duration), n)])
    return (mean=scale * result.mean, covariance=scale * result.covariance * scale')
end

"""Dense joint law of final endpoint and each exact past interval average.

All interval integrals remain coordinates in a single augmented Gaussian state.
They freeze at their own stop boundary. No observation is assimilated here.
"""
function timeline_joint(mean, P, lambdas, sigmas, forcing_intervals, durations)
    m, covariance = Float64.(mean), Matrix{Float64}(P)
    n, count = length(m), length(durations)
    state_mean = [m; zeros(count)]
    state_covariance = zeros(n + count, n + count)
    state_covariance[1:n, 1:n] .= covariance
    for interval in 1:count
        A = zeros(n + count, n + count)
        A[1:n, 1:n] .= -Diagonal(Float64.(lambdas))
        A[n + interval, 1:n] .= 1.0
        D = zeros(n + count, n + count)
        D[1:n, 1:n] .= Diagonal(Float64.(sigmas) .^ 2)
        b = [Float64.(forcing_intervals[interval]); zeros(count)]
        next = affine_flow(A, b, D, Float64(durations[interval]), state_mean, state_covariance)
        state_mean, state_covariance = next.mean, next.covariance
    end
    scale = Diagonal([ones(n); 1.0 ./ Float64.(durations)])
    return (mean=scale * state_mean, covariance=scale * state_covariance * scale')
end

"""One dense block Gaussian solve, never a candidate sequential-update formula."""
function gaussian_condition(joint, output_rows, observation_rows, values, R)
    x, y = collect(output_rows), collect(observation_rows)
    isempty(y) && return (mean=joint.mean[x], covariance=joint.covariance[x, x])
    cross = joint.covariance[x, y]
    innovation = joint.covariance[y, y] + Matrix{Float64}(R)
    factor = cholesky(Symmetric(innovation))
    corrected_mean = joint.mean[x] + cross * (factor \ (Float64.(values) - joint.mean[y]))
    corrected_covariance = joint.covariance[x, x] - cross * (factor \ cross')
    return (mean=corrected_mean, covariance=corrected_covariance)
end

end
