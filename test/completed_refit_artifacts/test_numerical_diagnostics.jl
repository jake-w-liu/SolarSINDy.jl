@testset "closed directed diagnostic codec" begin
    for bits in (256,512)
        record=setprecision(BigFloat,bits) do
            R._interval_record(R._Iv(BigFloat(-3)*BigFloat(2)^(-2003),BigFloat(5)*BigFloat(2)^1020),bits)
        end
        decoded=R._read_interval_record(record)
        @test R._dyadic(decoded.lo)==record["lower_dyadic"]
        @test R._dyadic(decoded.hi)==record["upper_dyadic"]
    end
    for token in ("0001p0","2p0","1p-0","-0p0","1p+0","1p16385",repeat("f",129)*"p0")
        @test_throws ArgumentError R._read_dyadic(token,512)
    end
    base=Dict{String,Any}("codec"=>"dyadic_interval_v1","lower_dyadic"=>"0p0","upper_dyadic"=>"1p0","precision_bits"=>512)
    malformed=deepcopy(base);malformed["precision_bits"]=true
    @test_throws ArgumentError R._read_interval_record(malformed)
    malformed=deepcopy(base);malformed["codec"]="unknown"
    @test_throws ArgumentError R._read_interval_record(malformed)
    malformed=deepcopy(base);malformed["lower_dyadic"]="3p0"
    @test_throws ArgumentError R._read_interval_record(malformed)
    malformed=deepcopy(base);malformed["extra"]=1
    @test_throws ArgumentError R._read_interval_record(malformed)
    interval(lo,hi)=R._interval_record(R._Iv(BigFloat(lo;precision=512),BigFloat(hi;precision=512)),512)
    payload=Dict{String,Any}("precision_bits"=>512,"backward_eta"=>interval(0,0),"condition_estimate"=>interval(0,2),
        "forward_enclosure"=>interval(0,0),"normal_residual"=>interval(0,0))
    @test R._fit_diagnostic_sanity(payload,10,8)===nothing
    malformed=deepcopy(payload);malformed["forward_enclosure"]=interval(0,big"0.00001")
    @test_throws ArgumentError R._fit_diagnostic_sanity(malformed,10,8)
    malformed=deepcopy(payload);malformed["backward_eta"]=interval(0,big"0.000000000001")
    @test_throws ArgumentError R._fit_diagnostic_sanity(malformed,10,8)
end
