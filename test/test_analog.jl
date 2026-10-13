module OperationalAnalogTests

using Test
using Dates
using Random
using SolarSINDy

"""
Independent O(n²) nearest-neighbour reference. It evaluates the weighted squared distance term by
term, which is a different computation from the blockwise ‖a‖² − 2⟨q, a⟩ expansion under test, and
it resolves ties by explicit lexicographic ordering on (distance, archive index).
"""
function _brute_neighbours(Xq, tq, Xa, ta, K, weights, exclusion)
    nq, p = size(Xq)
    na = size(Xa, 1)
    out = Matrix{Int}(undef, nq, K)
    keys = Vector{Tuple{Float64,Int}}(undef, na)
    for i in 1:nq
        for j in 1:na
            distance = 0.0
            for c in 1:p
                distance += weights[c] * (Xq[i, c] - Xa[j, c])^2
            end
            keys[j] = (abs(ta[j] - tq[i]) < exclusion ? Inf : distance, j)
        end
        sorted = sort(keys)
        for m in 1:K
            out[i, m] = sorted[m][2]
        end
    end
    return out
end

function _synthetic_lookup(entries)
    lookup = Dict{DateTime,NamedTuple{(:V, :Bz, :By, :n, :Pdyn),NTuple{5,Float64}}}()
    for (t, V, Bz, By, n) in entries
        lookup[t] = (V = V, Bz = Bz, By = By, n = n, Pdyn = dynamic_pressure(n, V))
    end
    return lookup
end

@testset "analog ensemble retrieval matches a brute-force reference" begin
    rng = MersenneTwister(20260817)
    nq, na, K = 300, 300, 12
    Xq = randn(rng, nq, 18)
    Xa = randn(rng, na, 18)
    # Query and archive hours overlap so the ±168 h exclusion removes a query-dependent slice of
    # the archive rather than being vacuous.
    tq = collect(0.0:8.0:(8.0 * (nq - 1)))
    ta = collect(-400.0:8.0:(-400.0 + 8.0 * (na - 1)))

    for weights in (ANALOG_WEIGHTS.uniform, ANALOG_WEIGHTS.magnetic)
        expected = _brute_neighbours(Xq, tq, Xa, ta, K, weights, 168.0)
        @test analog_knn(Xq, tq, Xa, ta, K; weights = weights) == expected
        # Blocking is an implementation detail and must not touch the answer.
        @test analog_knn(Xq, tq, Xa, ta, K; weights = weights, block = 7) == expected
        @test analog_knn(Xq, tq, Xa, ta, K; weights = weights, block = 4096) == expected
    end

    # A zero exclusion window is a different admissible set, so the reference must move with it.
    open_expected = _brute_neighbours(Xq, tq, Xa, ta, K, ANALOG_WEIGHTS.uniform, 0.0)
    @test analog_knn(Xq, tq, Xa, ta, K; exclusion_hours = 0) == open_expected
    @test open_expected != _brute_neighbours(Xq, tq, Xa, ta, K, ANALOG_WEIGHTS.uniform, 168.0)

    # The archive order is a labelling, not information: permuting it permutes the answer.
    permutation = randperm(rng, na)
    permuted = analog_knn(Xq, tq, Xa[permutation, :], ta[permutation], K)
    inverse = Dict(original => shuffled for (shuffled, original) in enumerate(permutation))
    reference = analog_knn(Xq, tq, Xa, ta, K)
    @test [inverse[reference[i, m]] for i in 1:nq, m in 1:K] == permuted
end

@testset "analog ensemble retrieval with DateTime axes" begin
    rng = MersenneTwister(4242)
    Xq = randn(rng, 40, 18)
    Xa = randn(rng, 120, 18)
    origin = DateTime(2015, 5, 1, 0)
    tq = [origin + Hour(6 * (i - 1)) for i in 1:40]
    ta = [origin - Day(30) + Hour(4 * (j - 1)) for j in 1:120]
    hours(times) = [Dates.value(t - DateTime(2000, 1, 1)) / 3_600_000 for t in times]
    expected = _brute_neighbours(Xq, hours(tq), Xa, hours(ta), 5, ANALOG_WEIGHTS.uniform, 168.0)
    @test analog_knn(Xq, tq, Xa, ta, 5) == expected
    @test analog_knn(Xq, string.(tq), Xa, string.(ta), 5) == expected
end

@testset "analog ensemble exclusion window is strict and symmetric" begin
    X = zeros(1, 18)
    Xa = zeros(4, 18)
    Xa[1, 1] = 3.0                       # farthest in feature space, admissible in time
    Xa[2, 1] = 1.0
    Xa[3, 1] = 2.0
    Xa[4, 1] = 4.0
    # Archive hours: exactly -168 (admissible), -167 (excluded), +167.5 (excluded), +168 (admissible).
    ta = [-168.0, -167.0, 167.5, 168.0]
    selected = analog_knn(X, [0.0], Xa, ta, 2)
    @test selected == [1 4]

    # One hour closer on either side removes that origin from the admissible set.
    @test analog_knn(X, [0.0], Xa, [-168.0, -167.0, 167.5, 168.0], 1) == reshape([1], 1, 1)
    @test analog_knn(X, [0.0], Xa, [-169.0, -167.0, 167.5, 167.9], 1) == reshape([1], 1, 1)
    @test analog_knn(X, [0.0], Xa, [-167.9, -167.0, 167.5, 168.0], 1) == reshape([4], 1, 1)

    # Unsorted archive times take the general scan instead of the binary-search range and must give
    # the same answer.
    order = [3, 1, 4, 2]
    @test analog_knn(X, [0.0], Xa[order, :], ta[order], 2) == [2 3]

    # Too few admissible origins is a hard failure, never a silently short or padded neighbour list.
    @test_throws ArgumentError analog_knn(X, [0.0], Xa, ta, 3)
    @test_throws ArgumentError analog_knn(X, [0.0], Xa, fill(0.0, 4), 1)
end

@testset "analog ensemble retrieval is deterministic under ties" begin
    Xa = zeros(6, 18)
    Xa[5, 1] = -1.0                      # one strictly nearer neighbour, five exact ties
    X = zeros(1, 18)
    X[1, 1] = -1.0
    ta = collect(1000.0:1000.0:6000.0)
    selected = analog_knn(X, [0.0], Xa, ta, 4)
    # The nearest is row 5; the remaining four are exact ties resolved toward the smaller index.
    @test selected == [5 1 2 3]
    @test analog_knn(X, [0.0], Xa, ta, 4; block = 1) == selected
end

@testset "analog ensemble retrieval rejects malformed input" begin
    X = zeros(3, 18)
    Xa = zeros(5, 18)
    ta = collect(1000.0:1000.0:5000.0)
    tq = [0.0, 1.0, 2.0]
    @test_throws DimensionMismatch analog_knn(X, tq, zeros(5, 17), ta, 2)
    @test_throws DimensionMismatch analog_knn(X, tq, Xa, ta, 2; weights = ones(17))
    @test_throws DimensionMismatch analog_knn(X, [0.0], Xa, ta, 2)
    @test_throws DimensionMismatch analog_knn(X, tq, Xa, ta[1:4], 2)
    @test_throws ArgumentError analog_knn(X, tq, Xa, ta, 0)
    @test_throws ArgumentError analog_knn(X, tq, Xa, ta, 6)
    @test_throws ArgumentError analog_knn(X, tq, Xa, ta, 2; block = 0)
    @test_throws ArgumentError analog_knn(X, tq, Xa, ta, 2; weights = zeros(18))
    @test_throws ArgumentError analog_knn(X, tq, Xa, ta, 2; weights = fill(-1.0, 18))
    @test_throws ArgumentError analog_knn(X, tq, Xa, ta, 2; exclusion_hours = -1)
    broken_query = copy(X); broken_query[2, 3] = NaN
    @test_throws ArgumentError analog_knn(broken_query, tq, Xa, ta, 2)
    broken_archive = copy(Xa); broken_archive[4, 1] = Inf
    @test_throws ArgumentError analog_knn(X, tq, broken_archive, ta, 2)
end

@testset "analog retrieval resolves cancellation and rejects invalid time axes" begin
    query = zeros(1,18)
    archive = zeros(2,18)
    archive[1,2] = sqrt(2.)
    archive[2,1] = 1.
    weights = ones(18); weights[1] = 2.
    # The first distance is strictly greater than two in exact arithmetic;
    # pre-scaling by sqrt(2) rounds both transformed distances to the same value.
    exact_distances = [sum(BigFloat(weights[c]) *
        (BigFloat(archive[j,c]) - BigFloat(query[1,c]))^2 for c in 1:18) for j in 1:2]
    @test sortperm(exact_distances) == [2,1]
    @test vec(analog_knn(query,[0.],archive,[1000.,2000.],2;weights)) == [2,1]
    @test analog_knn(query,[0.],archive,[1000.,2000.],1;weights) == reshape([2],1,1)
    # Independently subtract in BigFloat: all three squared distances collapse
    # to the same expanded BLAS score before the cancellation fix.
    for shift in (3.0, 1e8), block in (1, 7)
        query = fill(shift, 1, 18)
        archive = fill(shift, 3, 18)
        archive[:, 1] .+= [3e-8, 1e-8, 2e-8]
        distances = [sum(abs2, BigFloat.(query[1, :]) .-
                              BigFloat.(archive[j, :])) for j in 1:3]
        expected = sortperm(distances)
        @test expected == [2, 3, 1]
        @test vec(analog_knn(query, [0.0], archive, [1000., 2000., 3000.], 3;
                            block)) == expected
        @test analog_knn(query, [0.0], archive, [1000., 2000., 3000.], 1;
                         block) == reshape([2], 1, 1)
    end
    for times in ([NaN], [Inf], [-Inf])
        @test_throws ArgumentError analog_knn(zeros(1,18), times,
                                              zeros(2,18), [1000.,2000.], 1)
        @test_throws ArgumentError analog_knn(zeros(1,18), [0.0],
                                              zeros(1,18), times, 1)
    end
    @test_throws ArgumentError analog_knn(fill(1e200,1,18), [0.0],
                                          fill(1e200,1,18), [1000.], 1)
    @test_throws ArgumentError analog_knn(zeros(1,18), [0.0],
        zeros(2,18), [1000.,2000.], 1; weights=fill(big"1e400",18))
    @test_throws ArgumentError analog_knn(zeros(1,18), [0.0],
        zeros(2,18), [1000.,2000.], 1; weights=fill(big"1e-400",18))
end

@testset "analog ensemble driver continuation" begin
    s = DateTime(2020, 3, 1, 10)
    lookup = _synthetic_lookup([
        (s - Hour(1), 400.0, -3.0, 1.0, 4.0),      # analog issue record
        (s,           450.0, -12.0, 3.0, 8.0),     # step 1
        (s + Hour(1), 470.0, -20.0, -4.0, 2.0),    # step 2
        (s + Hour(2), 50.0, 5.0, 0.0, 0.02),       # step 3: below both physical bounds
        (s + Hour(3), 600.0, -1.0, 2.0, 400.0),    # step 4: above the density bound
    ])
    query = (V = 500.0, Bz = -5.0, By = 1.0, n = 8.0, Pdyn = dynamic_pressure(8.0, 500.0))

    step1 = analog_member_driver(query, lookup, s, 1)
    @test step1.V == 550.0                          # 500 + (450 - 400)
    @test step1.n ≈ 16.0 rtol = 1e-14               # 8 * 8/4, through one log and one exp rounding
    @test step1.Bz == -12.0                         # magnetic components are copied, not anchored
    @test step1.By == 3.0
    @test step1.Pdyn == dynamic_pressure(step1.n, step1.V)
    @test propertynames(step1) == (:V, :Bz, :By, :n, :Pdyn)

    step2 = analog_member_driver(query, lookup, s, 2)
    @test step2.V == 570.0                          # 500 + (470 - 400)
    @test step2.n ≈ 4.0 rtol = 1e-14                # 8 * 2/4
    @test step2.Bz == -20.0
    @test step2.By == -4.0

    # Step 3 pushes both continued quantities outside the admissible physical range.
    step3 = analog_member_driver(query, lookup, s, 3)
    @test step3.V == ANALOG_MEMBER_MIN_V_KMS           # 500 + (50 - 400) = 150 -> floored
    @test step3.n == ANALOG_MEMBER_MIN_N_CM3           # 8 * 0.02/4 = 0.04 -> clamped up
    @test step3.Pdyn == dynamic_pressure(ANALOG_MEMBER_MIN_N_CM3, ANALOG_MEMBER_MIN_V_KMS)

    step4 = analog_member_driver(query, lookup, s, 4)
    @test step4.V == 700.0                          # 500 + (600 - 400)
    @test step4.n == ANALOG_MEMBER_MAX_N_CM3           # 8 * 400/4 = 800 -> clamped down

    # The T1a ablation copies speed and density instead of anchoring them.
    direct1 = analog_member_driver(query, lookup, s, 1; direct = true)
    @test direct1.V == 450.0
    @test direct1.n == 8.0
    @test direct1.Bz == step1.Bz
    @test direct1.Pdyn == dynamic_pressure(8.0, 450.0)
    @test analog_member_driver(query, lookup, s, 3; direct = true).V == ANALOG_MEMBER_MIN_V_KMS
    @test analog_member_driver(query, lookup, s, 4; direct = true).n == ANALOG_MEMBER_MAX_N_CM3

    # The continuation depends on the query issue driver only through speed and density.
    other = (V = 300.0, Bz = 9.0, By = -9.0, n = 2.0, Pdyn = dynamic_pressure(2.0, 300.0))
    other1 = analog_member_driver(other, lookup, s, 1)
    @test other1.V == 350.0
    @test other1.n ≈ 4.0 rtol = 1e-14
    @test other1.Bz == step1.Bz && other1.By == step1.By
    @test analog_member_driver(other, lookup, s, 1; direct = true).V == direct1.V

    @test_throws ArgumentError analog_member_driver(query, lookup, s, 0)
    @test_throws ArgumentError analog_member_driver(query, lookup, s, 6)
    orphan = _synthetic_lookup([(s, 450.0, -12.0, 3.0, 8.0)])
    @test_throws ArgumentError analog_member_driver(query, orphan, s, 1)
    # The ablation needs no origin record, so it survives where the anchored form cannot.
    @test analog_member_driver(query, orphan, s, 1; direct = true).V == 450.0
end

@testset "analog ensemble member wraps the continuation for the kernel" begin
    s = DateTime(2021, 7, 4, 6)
    lookup = _synthetic_lookup([
        (s - Hour(1), 380.0, -2.0, 0.5, 5.0),
        (s,           420.0, -9.0, 1.5, 7.0),
        (s + Hour(1), 460.0, -14.0, 2.5, 9.0),
    ])
    query = (V = 520.0, Bz = -6.0, By = 0.0, n = 10.0, Pdyn = dynamic_pressure(10.0, 520.0))
    member = analog_member(query, lookup, s)
    frozen = (V = 999.0, Bz = 99.0, By = 99.0, n = 99.0, Pdyn = 99.0)
    # The member ignores the frozen last-known driver and the transit index: those govern only the
    # L1-admitted steps, which never reach a member.
    @test member(2, frozen, 0) == analog_member_driver(query, lookup, s, 2)
    @test member(2, query, 1) == analog_member_driver(query, lookup, s, 2)
    @test analog_member(query, lookup, s; direct = true)(1, frozen, 0) ==
          analog_member_driver(query, lookup, s, 1; direct = true)
end

@testset "analog ensemble set is invariant to post-issue records" begin
    # End-to-end causality: build the retrieval key from an hourly frame, retrieve, then rewrite
    # every frame record from the issue hour onward. The retrieved analog set must not move, because
    # nothing at or after the issue hour is available to the forecaster except the issue-time Dst.
    rng = MersenneTwister(31337)
    start = DateTime(2016, 1, 1, 0)
    hours = 0:399
    speed = 350.0 .+ 120.0 .* randn(rng, length(hours))
    bz = 3.0 .* randn(rng, length(hours))
    by = 3.0 .* randn(rng, length(hours))
    density = 5.0 .+ abs.(randn(rng, length(hours)))
    dst = -30.0 .+ 15.0 .* randn(rng, length(hours))
    frame = (
        time_utc = [start + Hour(h) for h in hours],
        V = speed, Bz = bz, By = by, n = density,
        Pdyn = [dynamic_pressure(density[i], speed[i]) for i in eachindex(speed)],
        Dst = dst,
    )
    issue = start + Hour(360)
    archive_times = [start + Hour(h) for h in 10:150]      # all more than 168 h before the issue
    Xq, ok_q = analog_feature_matrix(frame, [issue])
    Xa, ok_a = analog_feature_matrix(frame, archive_times)
    @test ok_q[1]
    @test all(ok_a)
    stats = analog_feature_stats(Xa)
    reference = analog_knn(analog_standardize(Xq, stats.mean, stats.sd), [issue],
                        analog_standardize(Xa, stats.mean, stats.sd), archive_times, 20;
                        weights = ANALOG_WEIGHTS.magnetic)

    future_rows = frame.time_utc .>= issue
    mutated_speed = copy(speed); mutated_speed[future_rows] .= 900.0
    mutated_density = copy(density); mutated_density[future_rows] .= 40.0
    mutated = (
        time_utc = frame.time_utc, V = mutated_speed,
        Bz = (b = copy(bz); b[future_rows] .= -35.0; b),
        By = (b = copy(by); b[future_rows] .= 22.0; b),
        n = mutated_density,
        Pdyn = [dynamic_pressure(mutated_density[i], mutated_speed[i]) for i in eachindex(speed)],
        Dst = (d = copy(dst); d[frame.time_utc .> issue] .= -300.0; d),
    )
    Xq_mutated, _ = analog_feature_matrix(mutated, [issue])
    Xa_mutated, _ = analog_feature_matrix(mutated, archive_times)
    @test Xq_mutated == Xq
    @test Xa_mutated == Xa
    @test analog_knn(analog_standardize(Xq_mutated, stats.mean, stats.sd), [issue],
                  analog_standardize(Xa_mutated, stats.mean, stats.sd), archive_times, 20;
                  weights = ANALOG_WEIGHTS.magnetic) == reference

    # Sensitivity: a change inside the issue's own history window does move the retrieved set.
    perturbed_bz = copy(bz)
    perturbed_bz[frame.time_utc .== issue - Hour(1)] .-= 18.0
    perturbed = (time_utc = frame.time_utc, V = speed, Bz = perturbed_bz, By = by,
                 n = density, Pdyn = frame.Pdyn, Dst = dst)
    Xq_perturbed, _ = analog_feature_matrix(perturbed, [issue])
    @test Xq_perturbed != Xq
    @test analog_knn(analog_standardize(Xq_perturbed, stats.mean, stats.sd), [issue],
                  analog_standardize(Xa, stats.mean, stats.sd), archive_times, 20;
                  weights = ANALOG_WEIGHTS.magnetic) != reference
end

@testset "analog ensemble origin eligibility" begin
    s = DateTime(2018, 9, 9, 0)
    complete = _synthetic_lookup([
        (s + Hour(k), 400.0 + k, -1.0 * k, 0.5, 5.0) for k in -1:6
    ])
    @test analog_origin_ok(complete, [s]) == BitVector([true])
    @test analog_origin_ok(complete, [string(s)]) == BitVector([true])
    @test analog_origin_ok(complete, [s], max_step = 8) == BitVector([false])
    @test analog_origin_ok(complete, [s + Hour(1)]) == BitVector([false])

    # Every record from s-1 through s+6 is load bearing at max_step = 7.
    for missing_hour in -1:6
        gapped = _synthetic_lookup([
            (s + Hour(k), 400.0 + k, -1.0 * k, 0.5, 5.0) for k in -1:6 if k != missing_hour
        ])
        @test analog_origin_ok(gapped, [s]) == BitVector([false])
    end
    @test analog_origin_ok(complete, DateTime[]) == BitVector()
    @test_throws ArgumentError analog_origin_ok(complete, [s], max_step = 0)
end

end # module
