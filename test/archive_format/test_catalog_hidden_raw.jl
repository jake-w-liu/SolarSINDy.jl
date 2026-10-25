module ArchiveCatalogHiddenRawTests
using Dates,JSON3,Test,SolarSINDy
const ArchiveFormatAudit=SolarSINDy.ArchiveFormatAudit
const ArchiveFormatParsers=SolarSINDy.ArchiveFormatParsers
const ArchiveSourceCatalog=SolarSINDy.ArchiveSourceCatalog
const ArchiveCatalogAudit=SolarSINDy.ArchiveCatalogAudit
function run_role(independent)
f=JSON3.read(read(joinpath(@__DIR__,"fixtures","catalog_hidden_raw_v3.json"),String),Any)
reader=independent ? ArchiveFormatAudit.audit_ground : ArchiveFormatParsers.parse_ground
constructor=independent ? ArchiveCatalogAudit.audit_catalog : ArchiveSourceCatalog.build_catalog
readwitness=independent ? ArchiveCatalogAudit.audit_witness : ArchiveSourceCatalog.witness
bodies=Dict(b["name"]=>reader(b["text"];station=b["station"],product=b["product"],start=DateTime(2020)+Minute(b["start_minute"]),stop=DateTime(2020)+Minute(b["stop_minute"])) for b in f["bodies"])
positive=constructor([bodies["C"]]);@assert positive.complete_vectors==1&&length(positive)==1
println("CATALOG_NATIVE_VALID_POSITIVE_CONTROL_PASS");flush(stdout)
e=f["expected"]
@testset "Complete raw witness and original selected bits" begin
    h=constructor([bodies["H"]])
    @test h.raw_rows==2
    @test length(h)==1
    @test h.complete_vectors==0
    @test bytes2hex(codeunits(h.selected_storage))==e["H_selected_hex"]
    @test h.selected_sha256==e["H_selected_sha256"]
    @test h.raw_catalog_sha256==e["H_raw_catalog_sha256"]
    @test readwitness(h,1).valid_mask==1
    @test readwitness(h,1).source_line==8
    @test_throws ArgumentError constructor([bodies["H"],bodies["D"]])
    @test_throws ArgumentError constructor([bodies["D"],bodies["H"]])
end
end
run_role(false)
run_role(true)
end
