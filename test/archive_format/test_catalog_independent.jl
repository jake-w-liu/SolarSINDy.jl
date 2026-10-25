module ArchiveCatalogIndependentTests
using Dates,SHA,JSON3,Test,SolarSINDy
const P=SolarSINDy.ArchiveFormatAudit;const C=SolarSINDy.ArchiveCatalogAudit
f=JSON3.read(read(joinpath(@__DIR__,"fixtures","catalog_v1.json"),String),Any)
parsed=Dict{String,P.AuditBody}()
@testset "Precode independent body identities" begin
    for b in f["bodies"]
        body=P.audit_ground(b["text"];station=b["station"],product=b["product"],start=DateTime(2020)+Minute(b["start_minute"]),stop=DateTime(2020)+Minute(b["stop_minute"]))
        parsed[b["name"]]=body;@test C.audit_identity(body)==b["id"]
    end
end
@testset "Precode cross-body selection and complete raw consistency" begin
    for case in f["cases"],names in case["permutations"]
        bodies=[parsed[name] for name in names];e=case["expected"]
        if !e["accepted"]
            @test_throws ArgumentError C.audit_catalog(bodies);continue
        end
        c=C.audit_catalog(bodies)
        @test collect(c.body_ids)==e["canonical_ids"]
        @test bytes2hex(codeunits(c.selected_storage))==e["selected_hex"]
        @test c.selected_sha256==e["selected_sha256"]
        @test c.raw_catalog_sha256==e["raw_catalog_sha256"]
        @test c.raw_rows==e["raw_count"]
        @test length(c)==e["unique_count"]
        @test c.duplicates==e["duplicate_count"]
        @test c.complete_vectors==e["complete_count"]
        @test c.declared_minutes==e["declared_minutes"]
        @test c.missing_minutes==e["missing_minutes"]
        @test c.uncovered_minutes==e["uncovered_minutes"]
        for i in 1:length(c)
            row=C.audit_witness(c,i)
            @test isequal(C.audit_locate(c,row.tag),row)
            bi=findfirst(==(row.body_id),c.body_ids);@test bi!==nothing
            originals=[P.raw_record(c.bodies[bi],j) for j in 1:c.bodies[bi].record_count]
            @test any(r->r.source_line==row.source_line&&r.valid_mask==row.valid_mask&&isequal(r.values,row.values),originals)
        end
        @test C.audit_locate(c,DateTime(2020)-Minute(1))===nothing
        @test_throws ArgumentError C.audit_witness(c,true)
        @test_throws ArgumentError C.audit_witness(c,0)
        @test_throws ArgumentError C.audit_locate(c,DateTime(2020)+Second(1))
        @test_throws ArgumentError C.audit_catalog([first(bodies),first(bodies)])
    end
    @test_throws ArgumentError C.audit_catalog(P.AuditBody[])
    @test_throws ArgumentError C.audit_catalog(Any[1])
    @test_throws ArgumentError C.audit_catalog(fill(parsed["A"],33))
end
end
