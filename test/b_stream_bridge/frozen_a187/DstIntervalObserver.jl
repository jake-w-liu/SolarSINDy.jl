module DstIntervalObserver

using Dates, LinearAlgebra, SHA

export interval_kernel, interval_moments, condition_interval

# Prototype numerical scope: serial caller ownership. MPFR precision is restored
# even on failure; callers must not concurrently change BigFloat precision.
# Wider intermediates avoid a silently incorrect small-lambda cancellation or
# an overflow/underflow that can disappear after multiplication by another input.
const _NUMERIC_LOCK = ReentrantLock()
const _INTERNAL_PRECISION = 256

# Exact unrounded coefficients are private to this loaded module. Public calls
# retain the original validation and numeric lock. Misses commit only after the
# entire callback succeeds; nested callbacks remain part of the outer transaction.
const _KernelKey = NTuple{3,UInt64}
const _KernelValue = NTuple{7,BigFloat}
const _KERNEL_CACHE = Dict{_KernelKey,_KernelValue}()
const _KERNEL_FIFO = _KernelKey[]
struct _KernelStage
    owner::Task
    pending::Vector{Pair{_KernelKey,_KernelValue}}
end
const _KERNEL_STAGE = Ref{Union{Nothing,_KernelStage}}(nothing)
_cache_context() = try
    precision(BigFloat)==_INTERNAL_PRECISION && rounding(BigFloat)===RoundNearest
catch
    false
end

function _commit_kernel_stage(stage, parent)
    for entry in stage.pending
        key, value = entry
        haskey(_KERNEL_CACHE,key) && continue
        if parent!==nothing
            any(p -> first(p)==key,parent.pending) || push!(parent.pending,entry)
        else
            length(_KERNEL_FIFO)==64 && delete!(_KERNEL_CACHE,popfirst!(_KERNEL_FIFO))
            _KERNEL_CACHE[key]=value
            push!(_KERNEL_FIFO,key)
        end
    end
end

function _number(x::Real, label::AbstractString)
    x isa Bool && throw(ArgumentError("$label must be a real number, not Bool"))
    isfinite(x) || throw(ArgumentError("$label must be finite"))
    y = Float64(x)
    isfinite(y) || throw(ArgumentError("$label exceeds Float64 range"))
    x != 0 && y == 0 && throw(ArgumentError("$label is below Float64 range"))
    return y
end

function _result(x, label)
    y = Float64(x)
    isfinite(y) || throw(ArgumentError("$label exceeds the supported finite Float64 result range"))
    return y
end

function _precise(f::Function)
    lock(_NUMERIC_LOCK) do
        setprecision(BigFloat, _INTERNAL_PRECISION) do
            previous = _KERNEL_STAGE[]
            stage = _cache_context() ? _KernelStage(current_task(),Pair{_KernelKey,_KernelValue}[]) : nothing
            _KERNEL_STAGE[] = stage
            try
                result = f()
                stage===nothing || _commit_kernel_stage(stage,previous)
                return result
            finally
                _KERNEL_STAGE[] = previous
            end
        end
    end
end

function _kernel_big(lambda::Float64, sigma::Float64, delta::Float64)
    stage = _KERNEL_STAGE[]
    if stage===nothing || stage.owner!==current_task() || !_cache_context()
        return _kernel_big_uncached(lambda,sigma,delta)
    end
    key = (reinterpret(UInt64,lambda),reinterpret(UInt64,sigma),reinterpret(UInt64,delta))
    cached = get(_KERNEL_CACHE,key,nothing)
    cached===nothing || return cached
    for entry in stage.pending
        first(entry)==key && return last(entry)
    end
    value = _kernel_big_uncached(lambda,sigma,delta)
    push!(stage.pending,key=>value)
    return value
end

# Returns phi, average-state coefficient, forcing endpoint/average coefficients,
# endpoint process variance, endpoint/average covariance, average process variance.
function _kernel_big_uncached(lambda::Float64, sigma::Float64, delta::Float64)
    l, s, d = BigFloat(lambda), BigFloat(sigma), BigFloat(delta)
    x = l * d
    if iszero(x)
        return (one(x), one(x), d, d / 2, s^2*d, s^2*d/2, s^2*d/3)
    end
    phi = exp(-x)
    a = -expm1(-x) / x
    a2 = -expm1(-2x) / (2x)
    if x < BigFloat("0.01")
        # b=(1-a)/x and c=(1-2a+a2)/x^2. Polynomial truncation
        # error at x<=.01 is far below a Float64 ulp; small x remains stable.
        b = zero(x)
        c = zero(x)
        for k in 18:-1:0
            b = b*x + BigFloat((-1)^k) / BigFloat(factorial(big(k+2)))
        end
        for k in 18:-1:0
            n = k + 2
            coefficient = BigFloat((-1)^n * (big(2)^n - 2)) /
                          (BigFloat(factorial(big(n))) * (n+1))
            c = c*x + coefficient
        end
    else
        b = (1-a)/x
        c = (1-2a+a2)/(x*x)
    end
    scale = s*s*d
    return (phi, a, d*a, d*b, scale*a2, scale*a*a/2, scale*c)
end

"""
    interval_kernel(lambda, sigma, delta)

Exact scalar moments for dx=(-lambda*x+f)dt+sigma*dW on a constant-forcing
interval of positive duration delta (hours). Output concerns the average of x
over that interval, not its terminal value. lambda and sigma may be zero.
Inputs narrow explicitly to finite Float64; outputs are rounded finite Float64.
Call serially; no fitting, data access or pressure conversion occurs.
"""
function interval_kernel(lambda::Real, sigma::Real, delta::Real)
    l, s, d = _number(lambda, "lambda"), _number(sigma, "sigma"), _number(delta, "delta")
    l >= 0 && s >= 0 && d > 0 || throw(ArgumentError("lambda/sigma must be nonnegative and delta positive"))
    return _precise() do
        phi, a, u, v, qe, qem, qm = _kernel_big(l,s,d)
        (; delta=d, phi=_result(phi,"phi"), mean_coefficient=_result(a,"mean coefficient"),
           forcing_endpoint=_result(u,"endpoint forcing coefficient"),
           forcing_mean=_result(v,"mean forcing coefficient"),
           endpoint_process_variance=_result(qe,"endpoint process variance"),
           endpoint_mean_process_covariance=_result(qem,"endpoint/mean process covariance"),
           mean_process_variance=_result(qm,"mean process variance"))
    end
end

function _vector(x::AbstractVector{<:Real}, label)
    isempty(x) && throw(ArgumentError("$label must be nonempty"))
    length(x) <= 16 || throw(ArgumentError("prototype supports at most16 modes"))
    return [_number(v,label) for v in x]
end

function _covariance(x::AbstractMatrix{<:Real}, n::Int, label)
    size(x) == (n,n) || throw(DimensionMismatch("$label shape differs from state"))
    p = map(v -> _number(v,label), x)
    all(>=(0),diag(p)) || throw(ArgumentError("$label has a negative variance"))
    scale = maximum(abs,p)
    iszero(scale) && return p
    normalized = p ./ scale
    tolerance = 64*eps(Float64)*n
    maximum(abs,normalized-normalized') <= tolerance || throw(ArgumentError("$label must be symmetric"))
    # Symmetrization only removes permitted floating-point asymmetry. No
    # eigenvalue or variance is clipped; materially non-PSD inputs fail closed.
    p = p + (p'-p)/2
    minimum_eigenvalue = eigmin(Symmetric(p ./ scale))
    isfinite(minimum_eigenvalue) && minimum_eigenvalue >= -tolerance ||
        throw(ArgumentError("$label must be positive semidefinite"))
    return p
end

"""
    interval_moments(mean, covariance, lambdas, sigmas, forcing, delta)

Independent mode noise with a possibly correlated prior. `forcing[j]` is the
constant additive f_j in dx_j=(-lambda_j*x_j+f_j)dt+sigma_j*dW_j. The scalar
observation is the interval-average sum(x), already pressure corrected.
Returns new endpoint/mean joint moments without modifying any input.
"""
function interval_moments(mean::AbstractVector{<:Real}, covariance::AbstractMatrix{<:Real},
        lambdas::AbstractVector{<:Real}, sigmas::AbstractVector{<:Real},
        forcing::AbstractVector{<:Real}, delta::Real)
    m, l, s, f = _vector(mean,"state"), _vector(lambdas,"lambda"),
                 _vector(sigmas,"sigma"), _vector(forcing,"forcing")
    n = length(m)
    all(v -> length(v)==n,(l,s,f)) || throw(DimensionMismatch("mode vectors differ"))
    all(>=(0),l) && all(>=(0),s) || throw(ArgumentError("lambda/sigma must be nonnegative"))
    d = _number(delta,"delta")
    d > 0 || throw(ArgumentError("delta must be positive"))
    p = _covariance(covariance,n,"prior covariance")
    return _precise() do
        k = [_kernel_big(l[j],s[j],d) for j in 1:n]
        mb, pb, fb = BigFloat.(m), BigFloat.(p), BigFloat.(f)
        phi = [k[j][1] for j in 1:n]
        c = [k[j][2] for j in 1:n]
        end_mean = [phi[j]*mb[j]+k[j][3]*fb[j] for j in 1:n]
        bar_mean = sum(c[j]*mb[j]+k[j][4]*fb[j] for j in 1:n)
        end_cov = [phi[i]*pb[i,j]*phi[j] + (i==j ? k[i][5] : zero(BigFloat)) for i in 1:n, j in 1:n]
        cross = [phi[i]*sum(pb[i,j]*c[j] for j in 1:n)+k[i][6] for i in 1:n]
        bar_var = sum(c[i]*pb[i,j]*c[j] for i in 1:n,j in 1:n) + sum(k[j][7] for j in 1:n)
        out_m = [_result(v,"endpoint mean") for v in end_mean]
        out_p = map(v -> _result(v,"endpoint covariance"), end_cov)
        out_c = [_result(v,"endpoint/mean covariance") for v in cross]
        out_v = _result(bar_var,"interval variance")
        # Validate the entire joint block; valid tiny roundoff is retained.
        _covariance([out_p out_c; out_c' out_v],n+1,"joint covariance")
        (; endpoint_mean=out_m, endpoint_covariance=out_p,
           interval_mean=_result(bar_mean,"interval mean"), interval_variance=out_v,
           endpoint_interval_covariance=out_c)
    end
end

"""
    condition_interval(joint, observed_mean, observation_variance)

Condition the endpoint on the noisy interval mean using the full correlated
joint distribution. One likelihood is assimilated once. Returns a new state;
the independent-noise endpoint-observation Joseph equation is not used.
"""
function condition_interval(joint, observed_mean::Real, observation_variance::Real)
    m = _vector(joint.endpoint_mean,"endpoint mean")
    n = length(m)
    p = _covariance(joint.endpoint_covariance,n,"endpoint covariance")
    c = _vector(joint.endpoint_interval_covariance,"endpoint/mean covariance")
    length(c)==n || throw(DimensionMismatch("endpoint/mean covariance length differs"))
    v, mu, y, r = _number(joint.interval_variance,"interval variance"),
                 _number(joint.interval_mean,"interval mean"),
                 _number(observed_mean,"observation"), _number(observation_variance,"observation variance")
    r > 0 || throw(ArgumentError("observation variance must be positive"))
    block = _covariance([p c; c' v],n+1,"endpoint/mean joint covariance")
    return _precise() do
        # Joint Gaussian conditioning in wide intermediates. No clipping or
        # absolute value changes a signed covariance cross entry.
        variance = BigFloat(v)+BigFloat(r)
        variance > 0 || throw(ArgumentError("innovation variance must be positive"))
        gain = BigFloat.(c)./variance
        innovation = BigFloat(y)-BigFloat(mu)
        post_m = BigFloat.(m).+gain.*innovation
        # Correlated-noise augmented Joseph construction, B=[I -K].
        b = hcat(Matrix{BigFloat}(I,n,n),-gain)
        post_p = b*BigFloat.(block)*b' + BigFloat(r).*(gain*gain')
        out_m = [_result(z,"posterior mean") for z in post_m]
        out_p = map(z -> _result(z,"posterior covariance"),post_p)
        _covariance(out_p,n,"posterior covariance")
        (; mean=out_m,covariance=out_p,gain=[_result(z,"gain") for z in gain],
           innovation=_result(innovation,"innovation"),innovation_variance=_result(variance,"innovation variance"))
    end
end

end # module
