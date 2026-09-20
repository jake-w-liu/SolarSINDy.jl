module GroundAdaptiveCalibration

using CSV, DataFrames, Dates, JSON3, SHA, Statistics
include(joinpath(@__DIR__, "ground_delay_check.jl"))
const BaseModel = GroundDelayCheck
const MODEL_SHA256 = "c1af43235dfdea8525100227a7649535bfc977b43672be5e994310a14cae6cea"
const PROTOCOL_SHA256 = "cf58b0164174f25eea7eb98f5e2c1f1a918a27e62ac27a02985691cb63af31b1"
const PANEL_HASHES = Dict(
    "CMO_adjusted_2018_delay0.csv" => "5a359da92f087d654a5a8066880ef3b1359f404f22bda2dbecc5090ae82a6a84",
    "CMO_adjusted_2018_delay5.csv" => "2f961a68baa225fb5b79313dea446a3ca7127c11e9559c6c0dfb8b5b84355cec",
    "CMO_adjusted_2018_delay10.csv" => "c206ec4f2423644be336cc3f3531e7a331b5a4b53c7ec4352c5af56010d0e1d7",
    "CMO_adjusted_2021_delay0.csv" => "a1bf96a8348daf56965eb99b104c2b81f8a233e9eb066d3d7d9d24767fd66088",
    "CMO_adjusted_2021_delay5.csv" => "a77565475ba9984f5771aeae69f2d8b24f0d00a1b366655d03c6dcc1dd939ecb",
    "CMO_adjusted_2021_delay10.csv" => "aa6ad811f79dccd1edf0ac1b90a14ab3ec115f2d74eab339216fa3f3a7eb8bd2",
    "CMO_adjusted_2024_delay0.csv" => "366325a3009ba35d0d952505eec109478b64a5c9e0591c2df70b1b8c24ad1833",
    "CMO_adjusted_2024_delay5.csv" => "3c6066b068c9ae294a20dec63e2d9887918a6c58962fb72b6c9c96ee2f78bd15",
    "CMO_adjusted_2024_delay10.csv" => "f785cda3fac35948a4042c267a291072273361dbece959c34ed5e3d53e21309a",
    "CMO_adjusted_2026_delay0.csv" => "7e5168b86fa337eb554f6ce9ced761fbb224edd0e5b75ff0abc6e6d164344e63",
    "CMO_adjusted_2026_delay5.csv" => "8e51eba2bace00c004cd9da876e69fb430ebc9c65bd391bc1e5a9262fb6cbdf7",
    "CMO_adjusted_2026_delay10.csv" => "7e102adca6512166fc7b39bba0ad232ca6ba6b39a1e9614566fb0ea4f309bc32")

function upper_estimate(point, scale, scores, alpha)
    isfinite(alpha) || throw(ArgumentError("alpha must be finite"))
    isfinite(point) && point >= 0 && isfinite(scale) && scale >= 1 ||
        throw(ArgumentError("invalid point or residual scale"))
    !isempty(scores) && all(isfinite, scores) || throw(ArgumentError("invalid residual scores"))
    alpha <= 0 && return Inf
    alpha >= 1 && return -Inf
    rank = clamp(ceil(Int, (1-alpha)*(length(scores)+1)), 1, length(scores))
    return max(0, point + sort(scores)[rank]*scale)
end

function recalibrate(frame, seed, delay)
    delay in BaseModel.DELAYS || throw(ArgumentError("unsupported receipt delay"))
    nrow(frame) > 0 || throw(ArgumentError("empty forecast panel"))
    times = frame.anchor
    issorted(times) && allunique(times) || throw(ArgumentError("unordered or duplicate anchors"))
    frame.target_start == times .+ Minute(delay) &&
        frame.target_end == times .+ Minute(delay+30) || throw(ArgumentError("invalid target windows"))
    for column in (:prediction, :past, :target, :upper)
        all(x -> BaseModel.good(x) && x >= 0, frame[!,column]) ||
            throw(ArgumentError("invalid ground value in $column"))
    end
    length(seed) == 1440 && all(BaseModel.good, seed) || throw(ArgumentError("invalid seed"))
    history = Float64.(seed)
    upper = zeros(nrow(frame)); alpha = similar(upper)
    received = zeros(Int,nrow(frame)); missed = similar(received)
    next = 1; count = 0; misses = 0
    for i in eachindex(times)
        if i > 1 && times[i]-times[i-1] > Day(1)
            history = Float64.(seed); next = i; count = 0; misses = 0
        end
        while next < i && frame.target_end[next]+Minute(delay) < times[i]+Minute(delay)
            count += 1
            misses += frame.target[next] > upper[next]
            push!(history, (frame.target[next]-frame.prediction[next])/(1+frame.past[next]))
            length(history) > 1440 && popfirst!(history)
            next += 1
        end
        alpha[i] = .10 + (.10*count-misses)/6000
        received[i] = count; missed[i] = misses
        upper[i] = upper_estimate(frame.prediction[i], 1+frame.past[i], history, alpha[i])
    end
    result = copy(frame)
    result.original_upper = copy(frame.upper)
    result.upper = upper; result.alpha = alpha
    result.received = received; result.misses_received = missed
    return result
end

pinball(y, upper) = .90max(y-upper, 0) + .10max(upper-y, 0)
function score_panel(frame, model)
    scores = BaseModel.metrics(frame, model)
    return merge(scores, (; finite_nonempty=all(x -> isfinite(x) && x >= 0, frame.upper),
        mean_upper=mean(frame.upper), original_mean_upper=mean(frame.original_upper),
        mean_pinball=mean(pinball.(frame.target,frame.upper)),
        original_mean_pinball=mean(pinball.(frame.target,frame.original_upper)),
        alpha_min=minimum(frame.alpha), alpha_max=maximum(frame.alpha)))
end

function decision(metrics)
    nrow(metrics) > 0 || throw(ArgumentError("empty evaluation"))
    expected = Set((period, delay) for period in (2018,2021,2024,2026) for delay in (0,5,10))
    Set(zip(metrics.period,metrics.delay_minutes)) == expected && nrow(metrics) == 12 ||
        throw(ArgumentError("incomplete or duplicate development panel"))
    finite = all(metrics.finite_nonempty)
    accuracy = all(metrics.native_pass .& metrics.log_pass)
    coverage = all(metrics.coverage_pass)
    mean_loss = sum(metrics.n .* metrics.mean_pinball)/sum(metrics.n)
    old_loss = sum(metrics.n .* metrics.original_mean_pinball)/sum(metrics.n)
    pinball_pass = isfinite(mean_loss) && mean_loss <= old_loss
    return (; finite, accuracy, coverage, pinball_pass, mean_pinball=mean_loss,
        original_mean_pinball=old_loss, development_pass=finite && accuracy && coverage && pinball_pass,
        retrospective_only=true, prospective_qualification=false, serving_enabled=false)
end

function run_experiment(input, output)
    ispath(output) && error("Refusing to overwrite adaptive calibration results: $output")
    model_bytes = read(BaseModel.MODEL_PATH)
    bytes2hex(sha256(model_bytes)) == MODEL_SHA256 || error("frozen model changed")
    protocol = read(joinpath(@__DIR__,"ground_adaptive_protocol.md"))
    bytes2hex(sha256(protocol)) == PROTOCOL_SHA256 || error("frozen protocol changed")
    model = JSON3.read(model_bytes)
    hashes = Dict(name => bytes2hex(sha256(read(joinpath(input,name)))) for name in keys(PANEL_HASHES))
    hashes == PANEL_HASHES || error("frozen development panels changed")
    mkpath(output); results = NamedTuple[]; daily = NamedTuple[]
    for period in (2018,2021,2024,2026), delay in (0,5,10)
        name = "CMO_adjusted_$(period)_delay$(delay).csv"
        bytes = read(joinpath(input,name))
        bytes2hex(sha256(bytes)) == hashes[name] || error("development panel changed during evaluation")
        base = CSV.read(IOBuffer(bytes),DataFrame;strict=true)
        frame = recalibrate(base,model.calibration_seed,delay)
        CSV.write(joinpath(output,name),frame)
        push!(results,merge((;period,delay_minutes=delay),score_panel(frame,model)))
        frame.anchor_day = Date.(frame.anchor)
        for group in groupby(frame,:anchor_day;sort=true)
            push!(daily,merge((;period,delay_minutes=delay,anchor_day=first(group.anchor_day)),
                             score_panel(group,model)))
        end
    end
    metrics = DataFrame(results)
    CSV.write(joinpath(output,"metrics.csv"),metrics)
    CSV.write(joinpath(output,"daily.csv"),DataFrame(daily))
    verdict = decision(metrics)
    write(joinpath(output,"decision.json"),JSON3.write(verdict))
    write(joinpath(output,"receipt.json"),JSON3.write((;
        generated_utc=string(now(UTC))*"Z",source_sha256=hashes,
        model_sha256=MODEL_SHA256,protocol_sha256=PROTOCOL_SHA256,
        source_code_sha256=bytes2hex(sha256(read(@__FILE__))))))
    println(JSON3.write(verdict))
    return verdict
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("Usage: ground_adaptive_calibration.jl DELAY_PANELS NEW_OUTPUT")
    run_experiment(ARGS...)
end
end
