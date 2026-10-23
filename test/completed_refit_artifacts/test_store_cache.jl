@testset "bundle cache preserves requested kind on repeated reads" begin
    directory=joinpath(CODEC_TEST_DIRECTORY,"cached-structural-native")
    mkpath(dirname(directory))
    protocol,bundle,pins=structural_native_bundle(directory)
    # This fixture qualifies only a structural snapshot: its deliberately empty
    # native receipt is not a genuine process credential or audited source.
    @test bundle.kind=="native_job"
    @test_throws ArgumentError R._native_receipt_check(bundle,pins)
    @test_throws ArgumentError R._read_bundle(directory,protocol,pins;expected_kind="schedule")
    graph=R._Graph()
    @test R._read_bundle(directory,protocol,pins;expected_kind="native_job",graph).kind=="native_job"
    @test_throws ArgumentError R._read_bundle(directory,protocol,pins;expected_kind="schedule",graph)
    original=read(joinpath(directory,"COMMIT.json"))
    changed=R.decode_canonical(original);changed["kind"]="schedule"
    changedbytes=R.canonical_bytes(changed)
    write(joinpath(directory,"COMMIT.json"),changedbytes)
    pins[R.digest(changedbytes)]=directory
    @test_throws ArgumentError R._read_bundle(directory,protocol,pins;graph)
    write(joinpath(directory,"COMMIT.json"),original)
    @test R._read_bundle(directory,protocol,pins;graph)===graph.cache[bundle.manifest_sha256]
end
