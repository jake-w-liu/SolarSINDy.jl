@testset "alert restoration waits for complete persisted state" begin
    mktempdir() do directory
        path = joinpath(directory, "alert-state.json")
        write(path, JSON3.write((level=3, stale=false, seen_data=true)))
        reset_notify!()
        entered = Channel{Nothing}(1)
        release = Channel{Nothing}(1)
        reads = Ref(0)
        delayed_read = function (file, kind)
            reads[] += 1
            put!(entered, nothing)
            take!(release)
            return read(file, kind)
        end
        restoring = @async _load_alert_state!(path; read_fn=delayed_read)
        take!(entered)
        @test !_ALERT_STATE_LOADED[]
        evaluating = @async begin
            _load_alert_state!(path)
            state = compute_alert_state((available=false,), (available=false,), nothing)
            delivered = JSON3.Object[]
            result = maybe_notify!(state; state_path=path,
                url="https://example.invalid/test-only",
                post_fn=(url, headers, body) -> push!(delivered, JSON3.read(body)))
            return (; state, result, delivered)
        end
        try
            yield()
            @test !istaskdone(evaluating)
            put!(release, nothing)
            wait(restoring)
            output = fetch(evaluating)
            @test reads[] == 1
            @test _ALERT_STATE_LOADED[] && _SEEN_DATA[]
            @test output.state.stale
            @test output.result.fired && output.result.kind == "stale_onset"
            @test only(output.delivered).previous_level == 3
            @test !occursin("all clear", lowercase(only(output.delivered).text))
            @test JSON3.read(read(path, String)).seen_data
            _load_alert_state!(path; read_fn=(args...) -> error("must be idempotent"))
            @test reads[] == 1
        finally
            isready(release) || istaskdone(restoring) || put!(release, nothing)
            wait(restoring)
            reset_notify!()
        end
    end
end

@testset "interrupted restoration stays retryable" begin
    mktempdir() do directory
        path = joinpath(directory, "alert-state.json")
        write(path, JSON3.write((level=2, stale=false, seen_data=true)))
        reset_notify!()
        try
            @test_throws InterruptException _load_alert_state!(path;
                read_fn=(args...) -> throw(InterruptException()))
            @test !_ALERT_STATE_LOADED[]
            _load_alert_state!(path)
            @test _ALERT_STATE_LOADED[] && _SEEN_DATA[]
            @test _LAST_ALERT_LEVEL[] == 2
        finally
            reset_notify!()
        end
    end
end
