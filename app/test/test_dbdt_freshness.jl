@testset "Cached ground estimates retain truthful observation ages" begin
    reference = DateTime(2026, 9, 20, 12)
    observed = reference - Minute(2)
    geoe = (current_vkm=0.1, max_vkm=0.2, rho_ohm_m=1000.0,
        current_time_utc=jdt(observed), age_minutes=2.0,
        note="1-D uniform half-space estimate")
    source = (station="CMO", available=true, current_time_utc=jdt(observed),
        age_minutes=2.0, stale=false, geoelectric=geoe)
    current = _current_dbdt_result(source; reference=reference+Minute(3), cached=true)
    @test current.available && current.cached
    @test current.age_minutes == 5.0
    @test current.geoelectric.age_minutes == 5.0
    @test current.geoelectric.current_time_utc == geoe.current_time_utc
    @test current.geoelectric.current_vkm == 0.1
    @test source.geoelectric.age_minutes == 2.0

    boundary = _current_dbdt_result(source; reference=reference+Minute(8))
    @test boundary.available && boundary.geoelectric.age_minutes == 10.0
    stale = _current_dbdt_result(source; reference=reference+Minute(8)+Millisecond(1))
    @test !stale.available && stale.stale
    @test stale.geoelectric === nothing

    for stamp in (jdt(reference-Minute(11)), jdt(reference+Minute(3)), "malformed", nothing)
        invalid = merge(source, (geoelectric=merge(geoe, (current_time_utc=stamp,)),))
        result = _current_dbdt_result(invalid; reference)
        @test result.available
        @test result.geoelectric === nothing
    end
    unavailable = merge(source, (available=false,))
    @test _current_dbdt_result(unavailable; reference).geoelectric === nothing
    expired_source = (station="CMO", available=false, stale=true,
        current_time_utc=jdt(reference-Minute(20)), age_minutes=20.0)
    expired = _current_dbdt_result(expired_source; reference=reference+Minute(3), cached=true)
    @test expired.age_minutes == 23.0
    @test !expired.available && expired.stale
    @test get(expired, :cached, false)
    # A rejected payload cannot become available merely because its timestamp is recent.
    rejected = _current_dbdt_result(unavailable; reference=reference+Minute(3))
    @test !rejected.available && rejected.age_minutes == 5.0
    absent = merge(source, (geoelectric=nothing,))
    @test _current_dbdt_result(absent; reference).geoelectric === nothing

    key = ("CMO", 120)
    saved = lock(_DBDT_LOCK) do
        old = get(_DBDT_CACHE, key, nothing)
        _DBDT_CACHE[key] = (time(), source)
        old
    end
    try
        result = usgs_dbdt(; station="CMO", reference=reference+Minute(3),
            compute_fn=(_, _) -> error("a fresh cache must not refetch"))
        @test result.geoelectric.age_minutes == 5.0
        @test _DBDT_CACHE[key][2].geoelectric.age_minutes == 2.0
    finally
        lock(_DBDT_LOCK) do
            saved === nothing ? delete!(_DBDT_CACHE, key) : (_DBDT_CACHE[key] = saved)
        end
    end
end

@testset "Ground responses distinguish pending refresh from completed unavailability" begin
    key = ("CMO", 120)
    gate = Channel{Nothing}(1)
    calls = Threads.Atomic{Int}(0)
    reference = DateTime(2026, 9, 20, 12)
    work = (_, _) -> begin
        Threads.atomic_add!(calls, 1)
        take!(gate)
        (station="CMO", available=false)
    end
    lock(_DBDT_LOCK) do
        @test !haskey(_DBDT_REFRESH_TASKS, key)
        delete!(_DBDT_CACHE, key)
    end
    task = nothing
    try
        initial = usgs_dbdt(; station="CMO", compute_fn=work, reference, wait_timeout=0.0)
        @test !initial.available
        @test get(initial, :refresh_in_progress, nothing) === true
        task = lock(() -> get(_DBDT_REFRESH_TASKS, key, nothing), _DBDT_LOCK)
        @test task isa Task
        repeated = usgs_dbdt(; station="CMO", compute_fn=work, reference, wait_timeout=0.0)
        @test get(repeated, :refresh_in_progress, nothing) === true
        @test lock(() -> get(_DBDT_REFRESH_TASKS, key, nothing), _DBDT_LOCK) === task
        put!(gate, nothing)
        @test Base.timedwait(() -> istaskdone(task), 20.0) === :ok
        fetch(task)
        completed = usgs_dbdt(; station="CMO", reference,
            compute_fn=(_, _) -> error("negative cache must prevent another fetch"))
        @test !completed.available
        @test get(completed, :refresh_in_progress, nothing) === false
        @test calls[] == 1
        @test !haskey(_DBDT_REFRESH_TASKS, key)
    finally
        if task !== nothing && !istaskdone(task)
            isready(gate) || put!(gate, nothing)
            Base.timedwait(() -> istaskdone(task), 20.0) === :ok || error("refresh did not finish")
            fetch(task)
        end
        lock(_DBDT_LOCK) do
            delete!(_DBDT_CACHE, key)
        end
    end
end
