# Literal expectations recorded before the conditional-variance implementation.
using Test
@testset "Stored mean defines finite-precision corrected variance" begin
    @test InitialReferenceAudit.sample_scale_about(fill(1.6,304),1.6)==1.0
    @test InitialReferenceAudit.sample_scale_about(fill(1.6,304),1.6000000000000008)==6.672321400300542e-16
    @test InitialReferenceAudit.sample_scale_about([1.,3.],2.)==sqrt(2.)
    @test InitialReferenceAudit.sample_scale_about(zeros(2),0.)==1.0
    for values in (Float64[],[1.],[NaN,1.],[Inf,1.],[true,false])
        @test_throws ArgumentError InitialReferenceAudit.sample_scale_about(values,0.)
    end
    for center in (true,NaN,Inf)
        @test_throws ArgumentError InitialReferenceAudit.sample_scale_about([1.,3.],center)
    end
    @test_throws ArgumentError InitialReferenceAudit.sample_scale_about([1e308,-1e308],0.)
    @test_throws ArgumentError InitialReferenceAudit.sample_scale_about(zeros(1_000_001),0.)
    @test_throws ArgumentError InitialReferenceAudit.sample_scale_about(Any[0.,true],0.)
end
