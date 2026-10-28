const SB=S.B
@testset "Locked static quantiles, independent ridge oracles and boundary accounting" begin
    # The numeric capsule accepts encoded String bytes and owns mutable input bytes.
    literal=String(UInt8[0x00,0x01,0x02])
    capsule=SB.StaticCalibration("context",literal,"receipt",SB._Token())
    @test collect(codeunits(capsule.numeric))==UInt8[0x00,0x01,0x02]
    mutable_bytes=UInt8[0x00,0x01,0x02]
    owned=SB.StaticCalibration("context",mutable_bytes,"receipt",SB._Token())
    mutable_bytes[1]=0xff
    @test collect(codeunits(owned.numeric))==UInt8[0x00,0x01,0x02]
    @test SB._quantile(collect(1.:9.),.1)==9.
    @test SB._quantile(collect(1.:9.),.09)==Inf
    @test SB._normal_quantile(.5)==0.
    # Independent standard-normal0.9 quantile;retained Acklam approximation
    # has a predeclared5e-9 absolute tolerance against this reference value.
    @test isapprox(SB._normal_quantile(.9),1.2815515655446004;atol=5e-9,rtol=0)
    @test SB._normal_quantile(0.)==-Inf && SB._normal_quantile(1.)==Inf
    @test_throws ArgumentError SB._normal_quantile(NaN)
    @test_throws ArgumentError SB._normal_quantile(-.1)
    train=zeros(4,2);X=zeros(10,2);residuals=collect(-4.:5.);scores=residuals./2;peaks=zeros(10)
    numeric=SB._fit_numeric(train,X,residuals,scores,peaks,.1)
    full_design=hcat(ones(10),zeros(10,2))
    full_qr=SB.qr(vcat(full_design,Matrix{Float64}(SB.I,3,3)))\vcat(abs.(residuals),zeros(3))
    println("STATIC_ZERO_COLUMN_NATIVE_DIAGNOSTIC;FULL_QR=",repr(full_qr),";REDUCED_QR=",repr(numeric.weights),";EXACT_INTERCEPT=25/11;EXACT_INACTIVE_COEFFICIENTS=0,0")
    @test numeric.means==[0.,0.] && numeric.scales==[1e-9,1e-9]
    # With zero centered features,only the penalized intercept is fitted.
    @test isapprox(numeric.weights[1],25/11;atol=0,rtol=8eps(Float64)) && numeric.weights[2:3]==[0.,0.]
    @test numeric.raw==5. && numeric.normalized==2.5
    @test numeric.counts==[0,0,0,10] && numeric.edges==zeros(5) && numeric.quantiles==fill(5.,4)
    projected=SB._predict(numeric,zeros(2,2),zeros(2),fill(2.,2),zeros(2))
    @test keys(projected.bounds)==SB.METHODS
    @test projected.bounds.raw_split==projected.bounds.normalized_split==projected.bounds.regime_split==fill(expm1(5.),2)
    @test isapprox(projected.bounds.het_gaussian[1],expm1(numeric.normal*sqrt(pi/2)*(25/11));atol=2e-13,rtol=2e-15)
    @test all(c->c.issued==2 && c.full==c.empty==c.nonfinite==0,values(projected.counts))
    shifted_inactive=SB._predict(numeric,ones(1,2),zeros(1),fill(2.,1),zeros(1))
    @test shifted_inactive.bounds.het_gaussian==projected.bounds.het_gaussian[1:1]
    decoded=SB._decode(SB._encode(numeric))
    @test decoded.weights==numeric.weights && decoded.edges==numeric.edges && decoded.quantiles==numeric.quantiles
    @test decoded.raw==5. && decoded.normalized==2.5 && decoded.alpha==.1
    encoded=SB._encode(numeric);encoded[1]⊻=0x01
    @test_throws ArgumentError SB._decode(encoded)
    @test_throws ArgumentError SB._decode(vcat(SB._encode(numeric),UInt8[0]))
    # Fixed-edge bins:below min,duplicate last edges and an empty third bin.
    regime=SB._fit_numeric(train,X,residuals,scores,[0.,0.,0.,0.,1.,1.,1.,1.,2.,2.],.5)
    @test regime.counts==[4,4,0,2]
    @test regime.raw==1. && regime.quantiles==[-2.,2.,1.,5.]
    @test [SB.binof(regime.edges,x) for x in (-1.,0.,1.,1.5,2.,3.)]==[1,1,2,3,4,4]
    rp=SB._predict(regime,zeros(4,2),zeros(4),fill(2.,4),[0.,1.,1.5,2.])
    @test rp.bounds.regime_split==expm1.([-2.,2.,1.,5.])
    @test rp.counts.regime_split.empty==1 && !rp.all_bounds_finite_nonempty
    # Orthogonal intercept and single feature give an independent2x2 solve.
    orthogonal=SB._fit_numeric(reshape([-1.,1.],2,1),reshape([-1.,0.,1.],3,1),[1.,2.,3.],[1.,2.,3.],[0.,1.,2.],.5)
    sigma=sqrt(2.)+1e-9
    @test isapprox(orthogonal.weights[1],1.5;atol=0,rtol=8eps(Float64))
    @test isapprox(orthogonal.weights[2],(2/sigma)/(1+2/sigma^2);atol=2e-15,rtol=2e-15)
    # A unit shift of two constant training features produces identical~1e9
    # design columns. Float64 normal equations lose the unit regularizer.
    shifted=SB._fit_numeric(zeros(4,2),ones(2,2),[1.,2.],[1.,2.],[0.,0.],.5)
    @test all(isfinite,shifted.weights)
    design=[1.,1/1e-9,1/1e-9]
    exact_prediction=(3*(1+2*(1/1e-9)^2))/(3+4*(1/1e-9)^2)
    @test isapprox(sum(design.*shifted.weights),exact_prediction;atol=1e-12,rtol=0)
    full=SB._fit_numeric(train,zeros(2,2),[0.,0.],[0.,0.],[0.,0.],.1)
    fp=SB._predict(full,zeros(1,2),[0.],[1.],[0.])
    @test fp.bounds.raw_split==fp.bounds.normalized_split==fp.bounds.regime_split==[Inf]
    @test fp.counts.raw_split.full==fp.counts.raw_split.nonfinite==1 && !fp.all_bounds_finite_nonempty
    empty=SB._fit_numeric(train,X,fill(-2.,10),fill(-1.,10),peaks,.5)
    ep=SB._predict(empty,zeros(2,2),zeros(2),fill(2.,2),zeros(2))
    @test ep.bounds.raw_split==fill(expm1(-2.),2) && ep.counts.raw_split.empty==2
    @test ep.bounds.het_gaussian==[0.,0.] && !ep.all_bounds_finite_nonempty
    @test_throws ArgumentError SB._fit_numeric(zeros(1,2),X,residuals,scores,peaks,.1)
    @test_throws ArgumentError SB._fit_numeric(train,X,residuals,scores,fill(-1.,10),.1)
    @test_throws ArgumentError SB._fit_numeric(train,X,residuals,scores,peaks,0.)
    @test_throws ArgumentError SB._predict(numeric,zeros(2,2),zeros(2),[1.,.9],zeros(2))
    @test_throws ArgumentError SB._predict(numeric,zeros(2,2),[0.,NaN],ones(2),zeros(2))
    @test_throws ArgumentError SB._predict(numeric,zeros(2,3),zeros(2),ones(2),zeros(2))
end
