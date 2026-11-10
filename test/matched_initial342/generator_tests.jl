include(joinpath(pkgdir(SolarSINDy),"tools","actual_b_chain","fixtures","FullShapeFixture342.jl"))
function generator_checks(root)
    G=FullShapeFixture342
    @test G.SOURCE_ROWS==500000&&G.CHANNELS==17
    @test [G.value_at(15,t) for t in (0,1,255,256,257)]==[-80.,-79.75,-16.25,-80.,-79.75]
    @test [G.value_at(6,t) for t in (0,15,16,127,128,143,144)]==[-4.,-4.,-3.,3.,4.,4.,-4.]
    @test [G.value_at(8,t) for t in (0,7,8,247,248)]==[350.,350.,354.,470.,350.]
    @test [G.value_at(12,t) for t in (0,3,4,63,64)]==[1.,1.,1.125,2.875,1.]
    @test_throws ArgumentError G.value_at(true,0)
    @test_throws ArgumentError G.value_at(1,-1)
    @test_throws ArgumentError G.write_raw(joinpath(root,"bad.bin");rows=true)
    @test !ispath(joinpath(root,"bad.bin"))
    saved=G.write_raw(joinpath(root,"source17.bin");rows=260)
    @test saved.bytes==35360
    @test saved.sha256==b2_file(saved.path)
    # Independent LE emission and byte order, including the Dst wrap boundary.
    expected=IOBuffer()
    for t in 0:259,j in 1:17
        value=j==6 ? Float64(mod(div(t,16),9)-4) :
              j==8 ? Float64(350+4*mod(div(t,8),31)) :
              j==12 ? 1.0+Float64(mod(div(t,4),16))/8.0 :
              j==15 ? -80.0+Float64(mod(t,256))/4.0 : Float64(mod(13*t+7*j,16))/4.0
        write(expected,htol(reinterpret(UInt64,value)))
    end
    @test read(saved.path)==take!(expected)
    @test_throws ArgumentError G.write_raw(saved.path;rows=260)
    @test saved.sha256==b2_file(saved.path)
    specpath=joinpath(pkgdir(SolarSINDy),"tools","actual_b_chain","fixtures","full_shape342_spec_v1.json")
    spec=b2_json(specpath)
    @test spec["schema"]=="matched342_initial_full_shape_fixture_spec_v1"
    @test spec["source_rows"]==500000
    @test parse(Int,spec["prefix_start_ms"])==Dates.value(DateTime(1976,11,1))
    @test parse(Int,spec["training_boundary_ms"])==Dates.value(DateTime(2020,1,1))
    @test parse(Int,spec["validation_start_ms"])==Dates.value(DateTime(2020,1,1))
    @test parse(Int,spec["validation_end_ms"])==Dates.value(DateTime(2023,1,1))
    @test spec["generator"]["member"]=="tools/actual_b_chain/fixtures/FullShapeFixture342.jl"
    @test spec["generator"]["sha256"]==b2_file(joinpath(pkgdir(SolarSINDy),spec["generator"]["member"]))
    identifier=pop!(spec,"spec_id")
    @test identifier==b2_id(spec)
end
