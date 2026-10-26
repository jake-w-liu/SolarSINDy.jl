"""Generated Float64 features versus independent Decimal and MPFR references."""
module GeneratedCatalogFeatureRoundingOracles
using Dates,JSON3,SHA,Test,SolarSINDy
const C=SolarSINDy.CompletedSourceDBDT
const FIXTURE=joinpath(@__DIR__,"ORACLES.json")
open(io->bytes2hex(sha256(io)),FIXTURE)=="195ef89f0433c641034aa6ac0c6d9053387ded7ea13682ad3215a46c5dd85dcd" || error("frozen generated rounding fixture differs")
frombits(value)=reinterpret(Float64,parse(UInt64,value;base=16))
flatten(value)=vcat(value["raw_summaries"],value["ground11"],value["mixed8"],[value["scale"]])
function mpfr(points,issue)
    setprecision(256) do
        rates=[sqrt((BigFloat(after[1])-BigFloat(before[1]))^2+(BigFloat(after[2])-BigFloat(before[2]))^2)
               for (before,after) in zip(points,points[2:end])]
        n=length(rates);now=last(rates);average=sum(rates)/n;peak=maximum(rates)
        deviation=sqrt(sum((rates[i]-rates[j])^2 for i in 1:n for j in i+1:n)/(n*(n-1)))
        daily=2*BigFloat(pi)*(60hour(issue)+minute(issue))/1440
        seasonal=2*BigFloat(pi)*(dayofyear(issue)-1)/daysinyear(issue)
        ground=[log1p(now),log1p(average),log1p(peak),log1p(deviation),
            now/(1+peak),average/(1+peak),deviation/(1+average),sin(daily),cos(daily),sin(seasonal),cos(seasonal)]
        mixed=[now,average,peak,deviation,BigFloat("400.25"),BigFloat("-2.125"),BigFloat("2.125"),BigFloat("850.53125")]
        vcat([now,average,peak,deviation],ground,mixed,[1+ground[3]])
    end
end
fixture=JSON3.read(read(FIXTURE),Any)
fixture["schema"]=="generated_feature_rounding_oracles_v1" && length(fixture["cases"])==42 || error("rounding fixture schema/count")
maximum_error_fraction=0.0
for value in fixture["cases"]
    @testset "Independent rounding reference $(value["name"])" begin
        issue=DateTime(value["issue"][1:end-1])
        points=[(frombits(row[1]),frombits(row[2])) for row in value["points_bits"]]
        mag=Dict(issue-Minute(32-index)=>point for (index,point) in enumerate(points))
        wind=Dict(issue-Minute(1)=>(400.25,-2.125,3.25,4.5))
        original_mag=copy(mag);original_wind=copy(wind)
        ground=C.prepare_issue(mag,issue;mode=:ground11)
        mixed=C.prepare_issue(mag,issue;mode=:mixed8,omni=wind)
        @test ground.accepted
        @test mixed.accepted
        @test length(ground.features)==11
        @test length(mixed.features)==8
        @test isequal(ground.ground11,mixed.ground11)
        row=ground.source_row
        actual=vcat([row.dbdt_now,row.dbdt_mean30,row.dbdt_max30,row.dbdt_std30],ground.features,mixed.features,[1+ground.ground11[3]])
        gold=flatten(value["reference_decimal"]);bounds=flatten(value["absolute_bounds"])
        length(actual)==length(gold)==length(bounds)==24 || error("rounding oracle width")
        independent=mpfr(points,issue)
        for index in 1:24
            expected=parse(Float64,gold[index]);bound=Float64(bounds[index]);error=abs(actual[index]-expected)
            @test isfinite(actual[index]) && error<=bound
            if bound>0;global maximum_error_fraction=max(maximum_error_fraction,error/bound);end
            precise=parse(BigFloat,gold[index])
            allowance=BigFloat(10)^(-65)*(iszero(precise) ? BigFloat(1) : abs(precise))
            @test abs(independent[index]-precise)<=allowance
        end
        @test isequal(mag,original_mag)
        @test isequal(wind,original_wind)
    end
end
println(JSON3.write(Dict("complete"=>true,"generated_cases"=>42,"assertions"=>2310,
    "maximum_generated_error_fraction_of_predeclared_bound"=>maximum_error_fraction,
    "native_runtime"=>string(VERSION),"fixture_sha256"=>bytes2hex(sha256(read(FIXTURE))),
    "actual_source_access"=>false,"actual_fits"=>0,"full_floating_policy_complete"=>false)))
end
