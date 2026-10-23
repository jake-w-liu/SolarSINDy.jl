function nonzero_history_checks()
    oracle=JSON3.read(read(joinpath(@__DIR__,"nonzero_history_oracles_v1.json")))
    literal=oracle.literal;D=T.D
    fraction(x)=begin a,b=split(String(x),'/');parse(BigInt,a)//parse(BigInt,b) end
    cp=D.R.Checkpoint(DateTime(1900),Float64.(literal.checkpoint_mean),[Float64(literal.checkpoint_covariance[j][k]) for j in 1:2,k in 1:2])
    schedule=D._mean_base_schedule(cp,(.5,1/12),(1.,1.),0.,0.,6)
    basis=[Float64(literal.initial_basis[j][k]) for j in 1:2,k in 1:2]
    numeric_checks=0
    for beta in ("5","20")
        qmeans=Tuple(fraction.(oracle.qmeans[beta]))
        actual=D._mean_affine_coefficients(schedule,D._basis(basis),qmeans)
        transposed=D._mean_affine_coefficients(schedule,D._basis(permutedims(basis)),qmeans)
        for (h,expected) in enumerate(literal.beta_results[beta])
            @test expected.horizon==h
            for k in 1:2
                @test isapprox(actual.coefficients[h][k],parse(Float64,expected.coefficients[k]);atol=literal.atol,rtol=literal.rtol)
                numeric_checks+=1
            end
            @test isapprox(schedule.offsets[h],parse(Float64,expected.offset);atol=literal.atol,rtol=literal.rtol)
            numeric_checks+=1
            for j in 1:2,k in 1:2
                @test isapprox(actual.endpoint_bases[h][j][k],parse(Float64,expected.endpoint_basis[j][k]);atol=literal.atol,rtol=literal.rtol)
                numeric_checks+=1
            end
            prediction=schedule.offsets[h]+sum(actual.coefficients[h][k]*literal.gains[k] for k in 1:2)
            @test isapprox(prediction,parse(Float64,expected.prediction);atol=literal.atol,rtol=literal.rtol)
            numeric_checks+=1
            if h in (4,6)
                wrong_transpose=schedule.offsets[h]+sum(transposed.coefficients[h][k]*literal.gains[k] for k in 1:2)
                wrong_endpoint=-11+sum(schedule.means[h])+sum(actual.endpoint_bases[h][j][k]*literal.gains[k] for j in 1:2,k in 1:2)
                @test abs(wrong_transpose-parse(Float64,expected.prediction))>1e-3
                @test abs(wrong_endpoint-parse(Float64,expected.prediction))>1e-3
            end
        end
    end
    @test numeric_checks==oracle.expected_scalar_checks==96
    true
end
