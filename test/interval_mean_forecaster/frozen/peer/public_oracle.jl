# Independent baseline public-output oracle. Frozen before candidate edits.
using Test, SHA
const BASELINE = raw"C:/Temp/da-optimize-20261016/baseline/DstIntervalObserver.jl"
const EXPECTED_BASELINE_SHA = "657cf432ffdcc09f3058a57e79e327c159292fe6e98318d4f0dc6bc917b4a04c"
const OUTDIR = @__DIR__
const SOURCE = length(ARGS) >= 1 ? abspath(ARGS[1]) : BASELINE
const MODE = length(ARGS) >= 2 ? ARGS[2] : "verify"
MODE in ("record", "verify") || error("mode must be record or verify")
MODE == "record" && bytes2hex(sha256(read(SOURCE))) != EXPECTED_BASELINE_SHA && error("recording requires frozen baseline")
include(SOURCE)
const N = DstIntervalObserver
bits(x::Float64) = string(reinterpret(UInt64,x),base=16,pad=16)
encode(x::Float64) = "f:"*bits(x)
encode(x::Bool) = string(x)
encode(x::NamedTuple) = "("*join((string(k)*"="*encode(v) for (k,v) in pairs(x)),";")*")"
encode(x::AbstractArray) = "A"*string(size(x))*"["*join(encode.(vec(x)),",")*"]"
encode(x::Tuple) = "("*join(encode.(x),";")*")"
encode(x) = repr(x)
function observe(f)
    try
        "OK|"*encode(f())
    catch err
        "ERR|"*string(typeof(err))*"|"*replace(sprint(showerror,err),'\n'=>"\\n",'\r'=>"\\r")
    end
end
function kernel_cases()
    cases = Pair{String,Tuple}[]
    lambdas = (0.0,-0.0,nextfloat(0.0),1e-300,1e-14,prevfloat(.01),.01,nextfloat(.01),1/24,.25,1.,1e4,floatmax(Float64))
    for (i,l) in enumerate(lambdas), (j,s) in enumerate((0.,-0.,3.,1e-150)), (k,d) in enumerate((nextfloat(0.),.125,1.,24.,1e5))
        push!(cases,"grid-$(i)-$(j)-$(k)"=>(l,s,d))
    end
    for (i,t) in enumerate(((0.,floatmax(Float64),1.),(.25,1e154,1.),(1e-300,1e-300,1e300),
            (1e300,1e150,1e-300),(0.,1.,floatmax(Float64)),(true,3.,1.),(.25,true,1.),(.25,3.,true),
            (-1.,3.,1.),(.25,-1.,1.),(.25,3.,0.),(.25,3.,-0.),(NaN,3.,1.),(.25,Inf,1.),(.25,3.,Inf),
            (big"1e-1000",3.,1.),(.25,big"1e-1000",1.),(.25,3.,big"1e-1000"),
            (big"1e1000",3.,1.),(.25,big"1e1000",1.),(.25,3.,big"1e1000")))
        push!(cases,"edge-$(i)"=>t)
    end
    cases
end
function public_rows()
    rows=String[]
    for p in (128,512), (rname,r) in (("Nearest",RoundNearest),("Down",RoundDown),("Up",RoundUp),("ToZero",RoundToZero))
        setprecision(BigFloat,p) do
            setrounding(BigFloat,r) do
                prefix="p$(p)-$(rname)"
                for (id,args) in kernel_cases()
                    before=(precision(BigFloat),rounding(BigFloat))
                    value=observe(() -> N.interval_kernel(args...))
                    after=(precision(BigFloat),rounding(BigFloat))
                    push!(rows,"$(prefix)|kernel|$(id)|$(value)|restored=$(before==after)")
                    # Exact same public call again exercises a future warm key.
                    push!(rows,"$(prefix)|kernel-repeat|$(id)|"*observe(() -> N.interval_kernel(args...)))
                end
                # Correlated PSD and rank-deficient cases; nontrivial off-diagonal signs.
                for (id,args) in (("correlated",([1.,-2.],[2. -.7;-.7 1.],[.25,1/24],[3.,.5],[.2,-.3],.125)),
                    ("rank-one",([0.,1.],[1. -1.;-1. 1.],[0.,.25],[0.,3.],[0.,.1],24.)),
                    ("zero-noise",([1.,-2.],[0. 0.;0. 0.],[0.,-0.],[0.,-0.],[.2,-.3],1.)),
                    ("max-modes",(ones(16),Matrix{Float64}(I,16,16),collect(1.:16.)./100,ones(16),zeros(16),1.)),
                    ("range-failure",([0.],zeros(1,1),[0.],[floatmax(Float64)],[0.],1.)),
                    ("bad-covariance",([1.,2.],[1. 2.;2. 1.],[.25,.5],[1.,1.],[0.,0.],1.)),
                    ("negative-dynamics",([1.],ones(1,1),[-.25],[1.],[0.],1.)),
                    ("dimension",([1.,2.],ones(1,1),[.25,.5],[1.,1.],[0.,0.],1.)))
                    push!(rows,"$(prefix)|moment|$(id)|"*observe(() -> N.interval_moments(args...)))
                end
                m=[1.,-2.]; P=[2. -.7;-.7 1.]; l=[.25,1/24]; s=[3.,.5]; f=[.2,-.3]
                inputs=encode((m,P,l,s,f))
                joint=N.interval_moments(m,P,l,s,f,.125)
                golden=encode(joint)
                conditioned=N.condition_interval(joint,.3,2.)
                push!(rows,"$(prefix)|condition|valid|OK|"*encode(conditioned))
                push!(rows,"$(prefix)|condition|variance-bool|"*observe(() -> N.condition_interval(joint,.3,true)))
                push!(rows,"$(prefix)|condition|variance-zero|"*observe(() -> N.condition_interval(joint,.3,0.)))
                # Mutate every public array; next result must be independent and inputs unchanged.
                joint.endpoint_mean .= 123.; joint.endpoint_covariance .= 456.; joint.endpoint_interval_covariance .= 789.
                conditioned.mean .= 55.; conditioned.covariance .= 66.; conditioned.gain .= 77.
                repeat=N.interval_moments(m,P,l,s,f,.125)
                push!(rows,"$(prefix)|alias|moment-fresh=$(encode(repeat)==golden)|inputs-unchanged=$(encode((m,P,l,s,f))==inputs)")
                push!(rows,"$(prefix)|alias|condition-repeat|OK|"*encode(N.condition_interval(repeat,.3,2.)))
                push!(rows,"$(prefix)|context-end|precision=$(precision(BigFloat))|rounding=$(rounding(BigFloat))")
            end
        end
    end
    rows
end
using LinearAlgebra
before=(precision(BigFloat),rounding(BigFloat))
actual=public_rows()
@test (precision(BigFloat),rounding(BigFloat)) == before
expected_path=joinpath(OUTDIR,"public_expected.tsv")
if MODE == "record"
    isfile(expected_path) && error("expected oracle already exists; refusing overwrite")
    write(expected_path,join(actual,"\n")*"\n")
    println("RECORDED rows=",length(actual)," sha256=",bytes2hex(sha256(read(expected_path))))
else
    expected=readlines(expected_path)
    @test length(actual)==length(expected)
    @testset "Frozen independent public bits, errors, contexts and aliasing" begin
        for i in eachindex(expected)
            @test actual[i] == expected[i]
        end
    end
    println("VERIFIED rows=",length(actual)," source_sha256=",bytes2hex(sha256(read(SOURCE))))
end
