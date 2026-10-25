module ArchiveCatalogWindCoverageTests
using Dates,SHA,JSON3,Test,SolarSINDy
const P=SolarSINDy.ArchiveFormatParsers;const A=SolarSINDy.ArchiveFormatAudit
const C=SolarSINDy.ArchiveSourceCatalog;const I=SolarSINDy.ArchiveCatalogAudit
fixtures=JSON3.read(read(joinpath(@__DIR__,"fixtures","catalog_wind_coverage_v2.json"),String),Any)
primary=Dict{String,P.ParsedBody}();independent=Dict{String,A.AuditBody}()
@testset "Frozen wind and period declarations before catalog construction" begin
    for b in fixtures["bodies"]
        kw=(;station=b["station"],product=b["product"],start=DateTime(2020)+Minute(b["start_minute"]),stop=DateTime(2020)+Minute(b["stop_minute"]))
        p=b["kind"]=="ground" ? P.parse_ground(b["text"];kw...) : P.parse_wind(b["text"];kw...)
        a=b["kind"]=="ground" ? A.audit_ground(b["text"];kw...) : A.audit_wind(b["text"];kw...)
        @test bytes2hex(codeunits(p.raw_storage))==join(b["raw_hex"])
        @test bytes2hex(codeunits(a.raw_records))==join(b["raw_hex"])
        @test C.body_identity(p)==b["id"]
        @test I.audit_identity(a)==b["id"]
        primary[b["name"]]=p;independent[b["name"]]=a
    end
end
@testset "Frozen per-kind endpoint, gap and cross-body catalogs" begin
    for case in fixtures["cases"],names in case["permutations"]
        pb=[primary[n] for n in names];ab=[independent[n] for n in names];e=case["expected"]
        if !e["accepted"]
            @test_throws ArgumentError C.build_catalog(pb)
            @test_throws ArgumentError I.audit_catalog(ab)
            continue
        end
        p=C.build_catalog(pb);a=I.audit_catalog(ab)
        for catalog in (p,a)
            @test collect(catalog.body_ids)==e["canonical_ids"]
            @test bytes2hex(codeunits(catalog.selected_storage))==e["selected_hex"]
            @test catalog.selected_sha256==e["selected_sha256"]
            @test catalog.raw_catalog_sha256==e["raw_catalog_sha256"]
            @test catalog.raw_rows==e["raw_count"]
            @test length(catalog)==e["unique_count"]
            @test catalog.duplicates==e["duplicate_count"]
            @test catalog.complete_vectors==e["complete_count"]
            @test catalog.declared_minutes==e["declared_minutes"]
            @test catalog.missing_minutes==e["missing_minutes"]
            @test catalog.uncovered_minutes==e["uncovered_minutes"]
            @test isimmutable(catalog.bodies)
        end
        @test p.selected_storage==a.selected_storage
        for index in 1:length(p)
            row=C.witness(p,index);other=I.audit_witness(a,index)
            @test isequal(row,other)
            @test isequal(C.locate(p,row.tag),row)
            @test isequal(I.audit_locate(a,row.tag),other)
        end
        if case["name"]=="declared_gap_kept_separate"
            @test C.locate(p,DateTime(2020)+Minute(5))===nothing
            @test I.audit_locate(a,DateTime(2020)+Minute(5))===nothing
        elseif startswith(case["name"],"wind_")
            @test C.locate(p,DateTime(2020)+Minute(3))===nothing
            @test I.audit_locate(a,DateTime(2020)+Minute(3))===nothing
        end
    end
end
end
