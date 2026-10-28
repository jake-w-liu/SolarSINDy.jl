module PublisherUPValueTests
using Test,SHA,SolarSINDy
include(joinpath(@__DIR__,"..","..","research","dbdt_completed_source","publisher_up_values.jl"))
const P=PublisherUPValues
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
digest(x)=bytes2hex(sha256(x))
@testset "Publisher initialization, strict ties and owned states" begin
    initial=P.State(;nominal=.25)
    @test P.radius(initial)==4/3
    tied=P.advance(initial,P.radius(initial))
    @test tied.miss===false && tied.state.resolved==1 && tied.state.misses==0
    @test initial.resolved==initial.misses==0 && initial.log_wealth==0.
    @test P.radius(P.State(;nominal=.5))==0.
    @test P.advance(P.State(;nominal=.5),0.).miss===false
    @test P.advance(P.State(;nominal=.5),nextfloat(0.)).miss===true
    @test_throws MethodError P.State(.1,0,0,0.)
end
@testset "Independent rational wealth paths" begin
    for alpha in (1//10,1//4,1//2,3//4),n in 0:8,mask in 0:(2^n-1)
        s=P.State(;nominal=Float64(alpha));wealth=big(1)//big(1);misses=0
        for i in 1:n
            weight=(2misses+1)//(2i)
            bound=max(big(0)//big(1),wealth*(weight-alpha)/(alpha*(1-alpha)))
            # Scores have a full unit margin from each rational threshold.
            # A zero score is a strict-tie hit when the radius is zero.
            wanted=((mask>>(i-1))&1)==1
            score=wanted ? Float64(bound+1) : 0.
            @test isapprox(P.radius(s),Float64(bound);rtol=4e-14,atol=1e-14)
            before=P.encode_state(s);next=P.advance(s,score)
            @test P.encode_state(s)==before
            @test next.miss===wanted
            wealth*=wanted ? weight/alpha : (1-weight)/(1-alpha)
            misses+=wanted;s=next.state
            @test s.resolved==i && s.misses==misses
            @test isapprox(s.log_wealth,Float64(log(BigFloat(wealth)));rtol=0,atol=2e-13)
        end
        bytes=P.encode_state(s);decoded=P.decode_state(bytes;expected_sha256=digest(bytes))
        @test P.encode_state(decoded)==bytes
    end
end
@testset "Numeric boundaries and rehashed malformed values" begin
    for alpha in (true,false,0.,1.,NaN,Inf,big"1e-1000",big"1e1000")
        @test_throws ArgumentError P.State(;nominal=alpha)
    end
    s=P.State()
    for score in (true,false,-1.,NaN,Inf,big"1e1000",nothing,"1")
        @test_throws ArgumentError P.advance(s,score)
    end
    bytes=P.encode_state(s)
    @test_throws ArgumentError P.decode_state(bytes;expected_sha256="0"^64)
    @test_throws ArgumentError P.decode_state(vcat(bytes,UInt8('\n'));expected_sha256=digest(vcat(bytes,UInt8('\n'))))
    object=R.decode_canonical(bytes;limit=4096)
    for (field,bad) in (("schema","wrong"),("algorithm","alpha_correction"),("publisher_reference_sha256","0"^64),
        ("resolved",true),("resolved",P.MAX_UPDATES+1),("misses",1),("log_wealth","7ff0000000000000"),
        ("log_wealth","3ff0000000000000"),("numeric_values_only",false),("actual_source_authority_granted",true),("actual_fits",true))
        altered=deepcopy(object);altered[field]=bad;raw=R.canonical_bytes(altered;limit=4096)
        @test_throws ArgumentError P.decode_state(raw;expected_sha256=digest(raw))
    end
    mutable=copy(bytes);mutable[1]=UInt8('[')
    @test P.encode_state(s)==bytes
    @test_throws ArgumentError P.decode_state(mutable;expected_sha256=digest(mutable))
    full=P.State(.1,P.MAX_UPDATES,0,0.,P._Token())
    @test_throws ArgumentError P.advance(full,0.)
end
@test Sys.maxrss()<=8*1024^3
println("PUBLISHER_UP_NUMERIC_VALUES_PASSED;ACTUAL_SOURCE_ACCESS=false;ACTUAL_FITS=0;PEAK_RSS_BYTES=",Sys.maxrss())
end
