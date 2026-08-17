module OperationalV23FeatureTests

using Test
using DataFrames
using Dates
using SolarSINDy

# A 20-hour synthetic hourly frame whose issue-hour features are all hand-computable. Values are
# chosen so that every six-hour statistic has a closed form: the six southward Bz records alternate
# between -6 and -2 (mean -4, population sd exactly 2), the transverse field is non-zero only at the
# freshest record (hypot(8, -6) = 10), and the six densities alternate between 5 and 10.
const FRAME_START = DateTime(2020, 1, 1, 0)
const ISSUE = DateTime(2020, 1, 1, 13)

function _synthetic_frame()
    hours = 0:19
    times = [FRAME_START + Hour(h) for h in hours]
    bz = fill(0.0, length(hours))
    by = fill(0.0, length(hours))
    speed = fill(350.0, length(hours))
    density = fill(7.0, length(hours))
    dst = fill(-20.0, length(hours))

    # Records t-1 … t-6 (hours 12 … 7) carry the six-hour statistics.
    for (hour, value) in ((12, -6.0), (11, -2.0), (10, -6.0), (9, -2.0), (8, -6.0), (7, -2.0))
        bz[hour + 1] = value
    end
    by[13] = 8.0                        # hour 12 -> hypot(8, -6) = 10
    speed[13] = 500.0                   # hour 12  (t-1)
    speed[12] = 400.0                   # hour 11  (t-2)
    speed[11] = 300.0                   # hour 10  (t-3)
    speed[7] = 400.0                    # hour 6   (t-7) -> dv6 = 500 - 400
    for (hour, value) in ((12, 5.0), (11, 10.0), (10, 5.0), (9, 10.0), (8, 5.0), (7, 10.0))
        density[hour + 1] = value
    end
    # The southward run reaches back past the seven-hour history window and stops at hour 5.
    bz[7] = -3.0                        # hour 6
    bz[6] = 1.0                         # hour 5 terminates the run
    dst[14] = -50.0                     # hour 13 (issue time)
    dst[13] = -44.0                     # hour 12

    pdyn = [dynamic_pressure(density[i], speed[i]) for i in eachindex(speed)]
    return DataFrame(
        time_utc = times, V = speed, Bz = bz, By = by, n = density, Pdyn = pdyn, Dst = dst,
    )
end

# Independently hand-computed expectations for ISSUE, in V23_FEATURE_NAMES order.
const EXPECTED = Dict(
    :bz0 => -6.0,
    :bz1 => -2.0,
    :bz2 => -6.0,
    :bz_mean6 => -4.0,
    :bz_sd6 => 2.0,
    :by0 => 8.0,
    :bperp0 => 10.0,
    :bperp_mean6 => (10.0 + 2.0 + 6.0 + 2.0 + 6.0 + 2.0) / 6,
    :v0 => 500.0,
    :dv6 => 100.0,
    :logn0 => log(5.0),
    :logn_mean6 => (3 * log(5.0) + 3 * log(10.0)) / 6,
    :pdyn0 => 1.6726e-6 * 5.0 * 500.0^2,
    :vbs0 => 3.0,
    :vbs_mean3 => (3.0 + 0.8 + 1.8) / 3,
    :south_run => 7.0,
    :dst0 => -50.0,
    :ddst1 => -6.0,
)

@testset "V2.3 feature schema" begin
    @test length(V23_FEATURE_NAMES) == 18
    @test V23_FEATURE_COUNT == 18
    @test allunique(V23_FEATURE_NAMES)
    @test V23_FEATURE_NAMES[1] == :bz0
    @test V23_FEATURE_NAMES[end] == :ddst1
    @test v23_feature_index(:vbs_mean3) == 15
    @test_throws ArgumentError v23_feature_index(:not_a_feature)

    @test V23_WEIGHTS.uniform == ones(18)
    @test count(==(2.0), V23_WEIGHTS.magnetic) == 10
    @test count(==(1.0), V23_WEIGHTS.magnetic) == 8
    for name in (:bz0, :bz1, :bz2, :bz_mean6, :bz_sd6, :bperp0, :bperp_mean6,
                 :vbs0, :vbs_mean3, :south_run)
        @test V23_WEIGHTS.magnetic[v23_feature_index(name)] == 2.0
    end
    for name in (:by0, :v0, :dv6, :logn0, :logn_mean6, :pdyn0, :dst0, :ddst1)
        @test V23_WEIGHTS.magnetic[v23_feature_index(name)] == 1.0
    end
    # `v23_weights` must hand out a private copy: mutating it cannot corrupt the shared constant.
    scratch = v23_weights(:uniform)
    scratch[1] = 99.0
    @test V23_WEIGHTS.uniform[1] == 1.0
    @test v23_weights(:uniform)[1] == 1.0
    @test_throws ArgumentError v23_weights(:heavy)
end

@testset "V2.3 features on a hand-computed frame" begin
    frame = _synthetic_frame()
    X, ok = v23_feature_matrix(frame, [ISSUE])
    @test size(X) == (1, 18)
    @test ok == BitVector([true])
    for (position, name) in pairs(V23_FEATURE_NAMES)
        @test X[1, position] == EXPECTED[name]
    end
    # The pressure feature is the record's own pressure under the package identity, not a
    # recomputation from a different density/speed pair.
    @test X[1, v23_feature_index(:pdyn0)] == dynamic_pressure(5.0, 500.0)

    # A NamedTuple of columns is an equally valid frame.
    columns = (time_utc = frame.time_utc, V = frame.V, Bz = frame.Bz, By = frame.By,
               n = frame.n, Pdyn = frame.Pdyn, Dst = frame.Dst)
    @test v23_feature_matrix(columns, [ISSUE])[1] == X

    # ISO-8601 string timestamps resolve to the same rows.
    string_frame = copy(frame)
    string_frame.time_utc = string.(frame.time_utc)
    @test v23_feature_matrix(string_frame, [string(ISSUE)])[1] == X
end

@testset "V2.3 southward run-length cap" begin
    frame = _synthetic_frame()
    frame.Bz .= -1.0                    # every record southward -> the run would be unbounded
    X, ok = v23_feature_matrix(frame, [ISSUE])
    @test ok[1]
    # The issue hour is 13, so 13 earlier records exist; the feature must stop at the 12 h cap.
    @test X[1, v23_feature_index(:south_run)] == 12.0
    @test V23_SOUTH_RUN_CAP_H == 12

    # A northward record immediately before the issue time ends the run at zero.
    frame.Bz[13] = 0.0                  # hour 12 (t-1), Bz < 0 is strict
    @test v23_feature_matrix(frame, [ISSUE])[1][1, v23_feature_index(:south_run)] == 0.0
end

@testset "V2.3 incomplete history is refused, not imputed" begin
    frame = _synthetic_frame()
    complete, ok = v23_feature_matrix(frame, [ISSUE])
    @test ok[1]

    # Missing driver record t-7 (hour 6): only the six-hour statistics would survive, so the whole
    # row must be rejected.
    gapped = frame[frame.time_utc .!= FRAME_START + Hour(6), :]
    X, ok_gapped = v23_feature_matrix(gapped, [ISSUE])
    @test ok_gapped == BitVector([false])
    @test all(isnan, X)

    # Missing record t-1.
    gapped1 = frame[frame.time_utc .!= FRAME_START + Hour(12), :]
    @test v23_feature_matrix(gapped1, [ISSUE])[2] == BitVector([false])

    # Non-finite driver inside the window, and a non-positive density, are equally incomplete.
    for (column, hour, value) in ((:Bz, 9, NaN), (:V, 7, NaN), (:n, 10, 0.0), (:Pdyn, 8, NaN))
        broken = _synthetic_frame()
        getproperty(broken, column)[hour + 1] = value
        @test v23_feature_matrix(broken, [ISSUE])[2] == BitVector([false])
    end

    # Missing issue-time Dst, and missing Dst(t-1), both break the anchor pair.
    for hour in (13, 12)
        broken = _synthetic_frame()
        broken.Dst[hour + 1] = NaN
        @test v23_feature_matrix(broken, [ISSUE])[2] == BitVector([false])
    end

    # An issue time absent from the frame is incomplete, and a complete row in the same call is
    # unaffected by an incomplete neighbour.
    X_mixed, ok_mixed = v23_feature_matrix(frame, [DateTime(2019, 6, 1, 0), ISSUE])
    @test ok_mixed == BitVector([false, true])
    @test all(isnan, view(X_mixed, 1, :))
    @test view(X_mixed, 2, :) == view(complete, 1, :)

    @test_throws ArgumentError v23_feature_matrix(select(frame, Not(:Pdyn)), [ISSUE])
    duplicated = vcat(frame, frame[frame.time_utc .== ISSUE, :])
    @test_throws ArgumentError v23_feature_matrix(duplicated, [ISSUE])
end

@testset "V2.3 features are causal in the driver records" begin
    frame = _synthetic_frame()
    reference, _ = v23_feature_matrix(frame, [ISSUE])

    # Every driver record tagged at or after the issue hour is future information at issue time and
    # must not reach the features. Dst(t) is deliberately excluded from this sweep: the V2.1 anchor
    # convention makes the issue-hour Dst known at issue time, and it is the dst0 feature.
    mutated = _synthetic_frame()
    future_rows = mutated.time_utc .>= ISSUE
    mutated.V[future_rows] .= 950.0
    mutated.Bz[future_rows] .= -45.0
    mutated.By[future_rows] .= 30.0
    mutated.n[future_rows] .= 60.0
    mutated.Pdyn[future_rows] .= dynamic_pressure(60.0, 950.0)
    mutated.Dst[mutated.time_utc .> ISSUE] .= -400.0
    @test v23_feature_matrix(mutated, [ISSUE])[1] == reference

    # Mutation sensitivity: the features must react to every record they claim to use.
    for (column, hour) in ((:Bz, 12), (:Bz, 11), (:Bz, 10), (:Bz, 7), (:By, 12),
                           (:V, 12), (:V, 11), (:V, 6), (:n, 12), (:n, 7), (:Pdyn, 12))
        perturbed = _synthetic_frame()
        getproperty(perturbed, column)[hour + 1] += 1.5
        @test v23_feature_matrix(perturbed, [ISSUE])[1] != reference
    end
    for hour in (13, 12)
        perturbed = _synthetic_frame()
        perturbed.Dst[hour + 1] -= 3.0
        @test v23_feature_matrix(perturbed, [ISSUE])[1] != reference
    end
end

@testset "V2.3 feature standardisation" begin
    X = fill(NaN, 4, 18)
    X[1, :] .= 1.0
    X[2, :] .= 3.0
    X[4, :] .= 5.0                      # row 3 stays incomplete and must be ignored
    X[1, 2] = 0.0
    X[2, 2] = 0.0
    X[4, 2] = 0.0                       # a constant column exercises the sd floor
    stats = v23_feature_stats(X)
    @test stats.mean[1] == 3.0          # mean of 1, 3, 5
    @test stats.sd[1] == 2.0            # corrected sd of 1, 3, 5
    @test stats.mean[2] == 0.0
    @test stats.sd[2] == 1e-9

    Z = v23_standardize(X, stats.mean, stats.sd)
    @test Z[1, 1] == -1.0
    @test Z[2, 1] == 0.0
    @test Z[4, 1] == 1.0
    @test all(isnan, view(Z, 3, :))     # incomplete rows stay recognisable

    @test v23_feature_stats(X; sd_floor = 0.5).sd[2] == 0.5
    @test_throws ArgumentError v23_feature_stats(X; sd_floor = 0.0)
    @test_throws ArgumentError v23_feature_stats(X[1:1, :])
    @test_throws DimensionMismatch v23_feature_stats(X[:, 1:17])
    @test_throws DimensionMismatch v23_standardize(X, stats.mean[1:17], stats.sd[1:17])
    @test_throws ArgumentError v23_standardize(X, stats.mean, zeros(18))
end

end # module
