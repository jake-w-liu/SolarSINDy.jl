using Test
include(joinpath(@__DIR__,"..","..","tools","initial_reference_audit","IndependentScalarTreeReplay.jl"))
using .IndependentScalarTreeReplay

edges() = [Float64[-1, 0, 2] for _ in 1:29]
bias(x) = (feature=[0], condition=[0], prediction=[Float64(x)], split=[false])
branch(feature=1) = (feature=[feature, 0, 0], condition=[2, 0, 0],
                    prediction=[0.0, -1.5, 2.75], split=[true, false, false])
leaf(x) = (feature=zeros(Int, 3), condition=zeros(Int, 3),
           prediction=[Float64(x), 0.0, 0.0], split=falses(3))

@testset "Pre-code literal tree arithmetic expectations" begin
    model = Model(edges(), [bias(.25), branch()]; max_depth=2, nrounds=1)
    values = zeros(29)
    @test predict_delta(model, values) == -1.25
    values[1] = nextfloat(0.0)
    @test predict_delta(model, values) == 3.0
    values[1] = -2
    @test predict_delta(model, values) == -1.25
    values[1] = 3
    @test predict_delta(model, values) == 3.0

    last_feature = merge(branch(29), (condition=[1, 0, 0],))
    model29 = Model(edges(), [bias(.25), last_feature]; max_depth=2, nrounds=1)
    values .= 0
    for (x, expected) in ((-2.0, -1.25), (-1.0, -1.25), (0.0, 3.0))
        values[29] = x
        @test predict_delta(model29, values) == expected
    end

    cancellation = Model(edges(), [bias(16777216), leaf(1), leaf(-16777216)];
                         max_depth=2, nrounds=2)
    @test predict_delta(cancellation, zeros(29)) == 0.0
    deep = (feature=[1, 2, 0, 0, 0, 0, 0], condition=[2, 1, 0, 0, 0, 0, 0],
            prediction=[0., 0., -4., 7., 11., 0., 0.],
            split=[true, true, false, false, false, false, false])
    deep_model = Model(edges(), [bias(0), deep]; max_depth=3, nrounds=1)
    for ((a, b), expected) in (((0., -1.), 7.), ((0., 0.), 11.), ((1., -1.), -4.))
        values .= 0
        values[1], values[2] = a, b
        @test predict_delta(deep_model, values) == expected
    end
end

@testset "Invalid data and external ownership" begin
    e, trees = edges(), [bias(.25), branch()]
    model = Model(e, trees; max_depth=2, nrounds=1)
    e[1][1] = -99
    trees[2].prediction[2] = 100
    @test predict_delta(model, zeros(29)) == -1.25
    @test_throws ArgumentError predict_delta(model, zeros(28))
    @test_throws ArgumentError predict_delta(model, fill(NaN, 29))
    @test_throws ArgumentError Model(edges()[1:28], [bias(0)]; max_depth=2, nrounds=0)
    for corrupted in ([0., -1., 2.], [-1., NaN, 2.], [-1., Inf, 2.], [-1., 0.])
        bad = edges(); bad[2] = corrupted
        @test_throws ArgumentError Model(bad, [bias(0)]; max_depth=2, nrounds=0)
    end
    @test_throws ArgumentError Model(edges(), [bias(0)]; max_depth=true, nrounds=0)
    @test_throws ArgumentError Model(edges(), [bias(0)]; max_depth=7, nrounds=0)
    @test_throws ArgumentError Model(edges(), [bias(0)]; max_depth=2, nrounds=true)
    @test_throws ArgumentError Model(edges(), [bias(0)]; max_depth=2, nrounds=401)
    @test_throws ArgumentError Model(edges(), [bias(0)]; max_depth=2, nrounds=1)
    @test_throws ArgumentError Model(edges(), [merge(bias(0), (split=[true],))]; max_depth=2, nrounds=0)
    for bad in (merge(branch(), (feature=[0, 0, 0],)),
                merge(branch(), (feature=[true, false, false],)),
                merge(branch(), (condition=[0, 0, 0],)),
                merge(branch(), (condition=[4, 0, 0],)),
                merge(branch(), (prediction=[0., .1, 2.75],)),
                merge(branch(), (prediction=[0., Inf, 2.75],)),
                merge(branch(), (split=[true, true, false],)),
                merge(branch(), (split=[false, true, false],)),
                merge(branch(), (feature=[1.0, 0.0, 0.0],)),
                merge(branch(), (split=[1, 0, 0],)),
                merge(branch(), (extra=1,)))
        @test_throws ArgumentError Model(edges(), [bias(0), bad]; max_depth=2, nrounds=1)
    end
end

@testset "Saved Float64 scaling is model state at bin boundaries" begin
    model = Model(edges(), [bias(.25), branch()]; max_depth=2, nrounds=1)
    values=zeros(29);values[1]=nextfloat(0.0)
    saved_center=zeros(29);saved_scale=ones(29)
    @test predict_delta(model,(values.-saved_center)./saved_scale)==3.0
    recomputed_center=copy(saved_center);recomputed_center[1]=nextfloat(0.0)
    @test abs(recomputed_center[1]-saved_center[1])<=5e-12
    @test predict_delta(model,(values.-recomputed_center)./saved_scale)==-1.25
end
