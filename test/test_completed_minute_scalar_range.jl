@testset "Completed-source scalar range preserves physical domains" begin
    oracle=JSON3.read(read(joinpath(@__DIR__,"completed_minute_scalar_oracle.json")),Dict{String,Any})
    tiny=BigFloat(10)^(-10000)
    values=Dict("zero"=>BigFloat(0),"negative_zero"=>-0.0,
        "min_subnormal"=>BigFloat(2)^(-1074),"negative_min_subnormal"=>-BigFloat(2)^(-1074),
        "half_min_subnormal"=>BigFloat(2)^(-1075),"negative_half_min_subnormal"=>-BigFloat(2)^(-1075),
        "tiny"=>tiny,"negative_tiny"=>-tiny,"float32_subnormal"=>nextfloat(0f0),
        "beyond_max_float"=>BigFloat(2)^1024,"nan"=>NaN,"infinity"=>Inf,"boolean"=>false)
    for row in oracle["scalar_cases"]
        @test SolarSINDy.CompletedMinuteReplay.finite_scalar(values[row["input"]])==row["valid"]
    end
    @test tiny>0 && Float64(tiny)==0.0
    @test Float64(values["min_subnormal"])==nextfloat(0.0)
    @test Float64(values["negative_min_subnormal"])==-nextfloat(0.0)
    issue=DateTime(2027,1,1)
    mag=Dict(issue+Minute(k)=>(0.0,0.0) for k in -40:31)
    wind=Dict(issue-Minute(1)=>(400.0,-5.0,1.0,8.0))
    for bad in ((-tiny,-5.,1.,8.),(400.,-5.,1.,-tiny),(400.,tiny,1.,8.),
                (400.,-tiny,1.,8.),(400.,-5.,tiny,8.))
        answer=admit_features(mag,Dict(issue-Minute(1)=>bad),issue)
        @test !answer.accepted && answer.reason===:invalid_omni_history
    end
    for zero in (BigFloat(0),-0.0), slot in 1:4
        ordinary=Real[400.,-5.,1.,8.];ordinary[slot]=zero
        @test admit_features(mag,Dict(issue-Minute(1)=>Tuple(ordinary)),issue).accepted
    end
    for value in (tiny,-tiny)
        broken=Dict{DateTime,Tuple{Real,Real}}(k=>(v[1],v[2]) for (k,v) in mag)
        broken[issue-Minute(15)]=(value,0.)
        ground=admit_ground_features(broken,issue)
        @test !ground.accepted && ground.reason===:invalid_ground_history
        @test admit_features(broken,wind,issue).reason===:invalid_ground_history
        target=Dict{DateTime,Tuple{Real,Real}}(k=>(v[1],v[2]) for (k,v) in mag)
        target[issue+Minute(1)]=(value,0.)
        answer=admit_target(target,issue,issue+Minute(31))
        @test !answer.accepted && answer.reason===:invalid_ground_target
    end
    @test admit_ground_features(mag,issue).accepted
    @test admit_features(mag,wind,issue).accepted
    @test admit_target(mag,issue,issue+Minute(31)).value.target_max_dbdt==0.0
end
