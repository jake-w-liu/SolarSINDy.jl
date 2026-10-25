module ArchiveFormatDuplicateTests
using Test, Dates, SHA, JSON3, Random, SolarSINDy
const P=SolarSINDy.ArchiveFormatParsers
const Q=JSON3.read(read(joinpath(@__DIR__,"fixtures/hidden_duplicates_v1.json"),String),Any)
records=Any[]
function run(c)
    m=c["metadata"];f=c["kind"]=="ground" ? P.parse_ground : P.parse_wind
    f(c["body"];station=m["station"],product=m["product"],start=DateTime(m["start"]),stop=DateTime(m["stop"]))
end
for c in filter(c->occursin("positive",c["name"]),Q["cases"])
    parsed=run(c)
    @assert length(parsed)==1&&P.complete_values(parsed,1)===nothing
    @assert P.row(parsed,1).valid_mask==(c["kind"]=="ground" ? 1 : 7)
    push!(records,(;name=c["name"],positive=true,accepted=true))
end
for c in filter(c->!occursin("positive",c["name"]),Q["cases"])
    accepted=false;error_text=""
    try;run(c);accepted=true
    catch error;error isa ArgumentError||rethrow();error_text=sprint(showerror,error)
    end
    push!(records,(;name=c["name"],positive=false,accepted,error_text))
end
bad=count(x->!x.positive&&x.accepted,records)

@test bad==0
end
