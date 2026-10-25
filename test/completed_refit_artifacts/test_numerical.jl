@testset "independent frozen ridge primitive" begin
    X=[0.0 5.0;1.0 5.0;2.0 5.0];y=[1.0,2.0,3.0]
    for (mixed,expected) in ((false,[2.0,2/3,0.0]),(true,[1.5,0.6666666668888889069052675683111637,0.0]))
        recipe=Dict{String,Any}("penalty"=>R.float_bits(1.0),"intercept_penalty"=>mixed,
            "std_rule"=>(mixed ? "sample_std_plus_1e_minus9" : "sample_std_zero_nonfinite_to_one"))
        pre,Z=R._ridge_preprocess(X,recipe);A,b,w=R._normal_statistics(Z,y,recipe);beta=A\b
        @test beta≈expected atol=1e-13 rtol=0
        @test pre["constant_columns"]==[2]
        @test pre["scale"][2]==(mixed ? "3e112e0be826d695" : "3ff0000000000000")
        @test R._independent_design(X,recipe,pre)==Z
        qr=R.qr(vcat(Z,Matrix(R.Diagonal(sqrt.(w)))))\vcat(y,zeros(3))
        lo=R._ridge_enclosure(Z,y,w,beta,qr,256);hi=R._ridge_enclosure(Z,y,w,beta,qr,512)
        @test hi.eta.hi<=8*(9*2.0^-53/(1-9*2.0^-53))
        @test hi.forward.hi<=big"0.000001"
        @test R._inside(hi.condition,lo.condition)
        wrong=copy(beta);wrong[1]+=0.1
        @test_throws ArgumentError R._ridge_enclosure(Z,y,w,wrong,qr,256)
    end
end

@testset "native statistic dimensional grammar and immutable copy" begin
    for (layout,m) in (("mixed8",9),("ground11",12))
        A=Matrix{Float64}(R.I,m,m);b=collect(Float64,1:m)
        bytes=R._native_statistics_bytes(A,b,layout);oldpin=R.digest(bytes)
        av,bv=R._read_native_statistics(bytes,layout)
        @test length(av)==m*m && length(bv)==m
        @test av[1]==1.0 && bv[end]==Float64(m)
        A[1,1]=2.0;b[end]=-1.0
        @test R.digest(bytes)==oldpin
        @test R.digest(R._native_statistics_bytes(A,b,layout))!=oldpin
        @test_throws ArgumentError R._native_statistics_bytes(A[1:end-1,:],b,layout)
        @test_throws ArgumentError R._read_native_statistics("[[],[]]\n",layout)
        data=R.decode_canonical(bytes);data[1][1]="7ff0000000000000"
        @test_throws ArgumentError R._read_native_statistics(R._jsontext(data),layout)
        @test_throws ArgumentError R._read_native_statistics(bytes,layout=="mixed8" ? "ground11" : "mixed8")
    end
end

@testset "exact-real portable prediction includes normalization underflow" begin
    x=[nextfloat(0.0)];mu=[0.0];scale=[2.0];beta=[0.0,floatmax(Float64)];z=[0.0]
    certificate=R._check_prediction_bound(x,mu,scale,beta,z,0.0)
    exact=big(2.0)^(-51)-big(2.0)^(-104)
    @test certificate.reference.lo<=exact<=certificate.reference.hi
    @test certificate.normal_loss.lo<=exact<=certificate.normal_loss.hi
    @test certificate.bound.hi>=exact
    @test certificate.ED.hi<exact
    @test_throws ArgumentError R._check_prediction_bound([1.0],[0.0],[1.0],[0.0,1.0],[1.0],2.0)
end
