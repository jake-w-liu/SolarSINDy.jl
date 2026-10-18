# Frozen before edits: cache behavior oracle; implementation state names are planned contract.
using Test, SHA, LinearAlgebra
length(ARGS)==1 || error("usage: future_cache_oracle.jl C:/.../candidate/DstIntervalObserver.jl")
const SOURCE=abspath(ARGS[1])
const BASELINE=raw"C:/Temp/da-optimize-20261016/baseline/DstIntervalObserver.jl"
bytes2hex(sha256(read(BASELINE))) == "657cf432ffdcc09f3058a57e79e327c159292fe6e98318d4f0dc6bc917b4a04c" || error("baseline drift")
function load_fresh(path)
    wrapper=Module(gensym(:PeerFrozenOracle))
    Base.include(wrapper,path)
    getfield(wrapper,:DstIntervalObserver)
end
const B=load_fresh(BASELINE)
key(l,s,d)=(reinterpret(UInt64,Float64(l)),reinterpret(UInt64,Float64(s)),reinterpret(UInt64,Float64(d)))
ordered_keys(N)=copy(getfield(N,:_KERNEL_FIFO))
cache(N)=getfield(N,:_KERNEL_CACHE)
stage(N)=getfield(N,:_KERNEL_STAGE)[]
function value_snapshot(N)
    [(k,Tuple((precision(v),repr(v)) for v in cache(N)[k])) for k in ordered_keys(N)]
end
function stable(N)
    @test length(ordered_keys(N)) <= 64
    @test length(cache(N)) == length(ordered_keys(N))
    @test length(unique(ordered_keys(N))) == length(ordered_keys(N))
    @test Set(ordered_keys(N)) == Set(Base.keys(cache(N)))
    @test stage(N) === nothing
end
function confirm_coefficients(N,l,s,d)
    expected=B._precise(() -> B._kernel_big(Float64(l),Float64(s),Float64(d)))
    actual=cache(N)[key(l,s,d)]
    @test length(actual)==7
    for i in 1:7
        @test isequal(actual[i],expected[i])
        @test precision(actual[i])==256
    end
end
setprecision(BigFloat,128) do
setrounding(BigFloat,RoundNearest) do
@testset "FIFO64 exact-bit cache and whole-callback transaction" begin
    @testset "outside scope bypass, exact keys, values and no public alias" begin
        N=load_fresh(SOURCE)
        stable(N); @test isempty(ordered_keys(N))
        N._kernel_big(.25,3.,1.)
        setprecision(BigFloat,256) do
            N._kernel_big(.25,3.,1.)
        end
        @test isempty(ordered_keys(N)); stable(N)
        tuples=((0.,0.,1.),(-0.,0.,1.),(0.,-0.,1.),(-0.,-0.,1.),(.25,3.,1.),(nextfloat(.25),3.,1.),(.25,3.,nextfloat(1.)))
        for t in tuples
            N.interval_kernel(t...)
            confirm_coefficients(N,t...)
        end
        @test ordered_keys(N)==[key(t...) for t in tuples]
        # A warm private call outside _precise still obeys caller context, never cached precision.
        private_actual=N._kernel_big(.25,3.,1.)
        private_expected=B._kernel_big(.25,3.,1.)
        @test isequal(private_actual,private_expected)
        @test precision.(private_actual)==precision.(private_expected)
        snap=value_snapshot(N)
        m=[1.,-2.]; p=[2. -.7;-.7 1.]; l=[.25,1/24]; s=[3.,.5]; f=[.2,-.3]
        expected=B.interval_moments(m,p,l,s,f,.125)
        actual=N.interval_moments(m,p,l,s,f,.125)
        @test isequal(actual,expected)
        actual.endpoint_mean .= 11.; actual.endpoint_covariance .= 22.; actual.endpoint_interval_covariance .= 33.
        @test isequal(N.interval_moments(m,p,l,s,f,.125),expected)
        @test value_snapshot(N)[1:length(snap)]==snap
        @test m==[1.,-2.] && p==[2. -.7;-.7 1.] && l==[.25,1/24] && s==[3.,.5] && f==[.2,-.3]
        stable(N)
    end
    @testset "FIFO64, warm hit cannot refresh age" begin
        N=load_fresh(SOURCE)
        triples=[(.125+i/1024,.5,1.) for i in 1:65]
        for t in triples[1:64]; N.interval_kernel(t...); end
        @test ordered_keys(N)==[key(t...) for t in triples[1:64]]
        snapshot=value_snapshot(N)
        N.interval_kernel(triples[1]...)
        @test value_snapshot(N)==snapshot
        N.interval_kernel(triples[65]...)
        @test ordered_keys(N)==[key(t...) for t in triples[2:65]]
        @test !haskey(cache(N),key(triples[1]...))
        # Reinserted oldest is newest; oldest remaining must be evicted.
        N.interval_kernel(triples[1]...)
        @test ordered_keys(N)==[key(t...) for t in vcat(triples[3:65],triples[1:1])]
        stable(N)
    end
    @testset "errors discard all misses and warm validation remains first" begin
        N=load_fresh(SOURCE)
        @test_throws ArgumentError N.interval_kernel(0.,floatmax(Float64),1.)
        @test isempty(ordered_keys(N)); stable(N)
        @test_throws ArgumentError N.interval_moments([0.,0.],zeros(2,2),[.2,0.],[1.,floatmax(Float64)],[0.,0.],1.)
        @test isempty(ordered_keys(N)); stable(N)
        @test_throws ArgumentError N.interval_moments([floatmax(Float64)],zeros(1,1),[0.],[0.],[floatmax(Float64)],1.)
        @test isempty(ordered_keys(N)); stable(N)
        N.interval_kernel(0.,3.,1.); N.interval_kernel(.25,3.,1.)
        snapshot=value_snapshot(N)
        for args in ((false,3.,1.),(.25,true,1.),(.25,3.,true),(big"1e-1000",3.,1.),(.25,3.,big"1e-1000"),(-1.,3.,1.),(.25,3.,0.))
            @test_throws ArgumentError N.interval_kernel(args...)
            @test value_snapshot(N)==snapshot
        end
        @test_throws ArgumentError N.interval_moments([1.,2.],[1. 2.;2. 1.],[.25,.5],[3.,3.],[0.,0.],1.)
        @test value_snapshot(N)==snapshot
        @test_throws ArgumentError N.interval_kernel(0.,floatmax(Float64),1.)
        @test value_snapshot(N)==snapshot
        N.interval_kernel(.2,1.,1.)
        @test ordered_keys(N)==vcat([v[1] for v in snapshot],[key(.2,1.,1.)]); stable(N)
    end
    @testset "nonNearest bypass and context restoration at caller precision" begin
        N=load_fresh(SOURCE); N.interval_kernel(.25,3.,1.)
        snapshot=value_snapshot(N)
        for p in (128,512), r in (RoundDown,RoundUp,RoundToZero)
            setprecision(BigFloat,p) do
                setrounding(BigFloat,r) do
                    before=(precision(BigFloat),rounding(BigFloat))
                    @test isequal(N.interval_kernel(.25,3.,1.),B.interval_kernel(.25,3.,1.))
                    @test isequal(N.interval_kernel(.3,1.,.5),B.interval_kernel(.3,1.,.5))
                    function outcome(M)
                        try; (:ok,M.interval_kernel(0.,floatmax(Float64),1.))
                        catch e; (:error,typeof(e),sprint(showerror,e)); end
                    end
                    @test isequal(outcome(N),outcome(B))
                    @test_throws ArgumentError N.interval_kernel(true,3.,1.)
                    @test value_snapshot(N)==snapshot
                    @test (precision(BigFloat),rounding(BigFloat))==before
                end
            end
        end
        stable(N)
        @test precision(BigFloat)==128 && rounding(BigFloat)==RoundNearest
    end
    @testset "nested callback success and failure preserve transaction and stage" begin
        N=load_fresh(SOURCE)
        @test_throws ErrorException N._precise() do
            N._kernel_big(.2,1.,1.)
            outer=stage(N)
            N._precise() do
                N._kernel_big(.3,1.,1.)
            end
            @test stage(N)===outer
            @test isempty(ordered_keys(N))
            error("outer failure")
        end
        @test isempty(ordered_keys(N)); stable(N)
        N._precise() do
            N._kernel_big(.2,1.,1.)
            outer=stage(N)
            @test_throws ErrorException N._precise() do
                N._kernel_big(.3,1.,1.)
                error("inner failure")
            end
            @test stage(N)===outer
            @test isempty(ordered_keys(N))
            N._precise() do
                N._kernel_big(.4,1.,1.)
            end
            @test stage(N)===outer
            @test isempty(ordered_keys(N))
            N._kernel_big(.5,1.,1.)
        end
        @test ordered_keys(N)==[key(l,1.,1.) for l in (.2,.4,.5)]; stable(N)
        # A rounding change inside a supported callback may not read or write cache.
        snapshot=value_snapshot(N)
        N._precise() do
            outer=stage(N)
            setrounding(BigFloat,RoundDown) do
                @test isequal(N._kernel_big(.2,1.,1.),B._kernel_big(.2,1.,1.))
                @test isequal(N._kernel_big(.6,1.,1.),B._kernel_big(.6,1.,1.))
            end
            setprecision(BigFloat,128) do
                got=N._kernel_big(.7,1.,1.); expected=B._kernel_big(.7,1.,1.)
                @test isequal(got,expected)
                @test precision.(got)==precision.(expected)
            end
            @test stage(N)===outer
        end
        @test value_snapshot(N)==snapshot; stable(N)
    end
    @testset "task owner isolation and 16-mode public bound" begin
        N=load_fresh(SOURCE)
        N._precise() do
            outer=stage(N)
            task=@async N._kernel_big(.9,1.,1.)
            wait(task)
            @test stage(N)===outer
            @test isempty(ordered_keys(N))
        end
        @test isempty(ordered_keys(N)); stable(N)
        ls=collect(1.:16.)./100
        @test isequal(N.interval_moments(ones(16),Matrix{Float64}(I,16,16),ls,ones(16),zeros(16),1.),B.interval_moments(ones(16),Matrix{Float64}(I,16,16),ls,ones(16),zeros(16),1.))
        @test ordered_keys(N)==[key(l,1.,1.) for l in ls]; stable(N)
        for l in ls; confirm_coefficients(N,l,1.,1.); end
        M=load_fresh(SOURCE)
        @test isempty(ordered_keys(M))
        @test cache(M) !== cache(N) && getfield(M,:_KERNEL_FIFO) !== getfield(N,:_KERNEL_FIFO)
    end
end
end
end
println("CACHE_ORACLE_PASS source_sha256=",bytes2hex(sha256(read(SOURCE))))



