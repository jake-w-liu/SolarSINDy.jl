using SolarSINDy, Dates

function run_archive_format_synthetic()
    parser=SolarSINDy.ArchiveFormatParsers
    body="2020-01-01T00:00:00.000Z,1,2,400,5\n" *
         "2020-01-01T00:00:00.000Z,9999.99,2,400,5\n"
    source=parser.parse_wind(body;station="CMO",product="OMNI_HRO_1MIN",
        start=DateTime(2020),stop=DateTime(2020)+Minute(2))
    @assert source.raw_rows==2 && length(source)==1
    @assert parser.complete_values(source,1)==(400.,2.,1.,5.)
    @assert parser.raw_row(source,2).valid_mask==11
    @assert parser.raw_row(source,2).source_line==2
    @assert ncodeunits(source.raw_storage)==96
    independent=SolarSINDy.ArchiveFormatAudit.audit_wind(body;station="CMO",product="OMNI_HRO_1MIN",
        start=DateTime(2020),stop=DateTime(2020)+Minute(2))
    @assert independent.raw_records==source.raw_storage
    @assert independent.selected_records==source.storage
    catalog=SolarSINDy.ArchiveSourceCatalog.build_catalog([source])
    catalog_audit=SolarSINDy.ArchiveCatalogAudit.audit_catalog([independent])
    @assert catalog.raw_rows==2 && length(catalog)==1 && catalog.complete_vectors==1
    @assert catalog.selected_storage==catalog_audit.selected_storage
    @assert catalog.raw_catalog_sha256==catalog_audit.raw_catalog_sha256
    @assert SolarSINDy.ArchiveSourceCatalog.locate(catalog,DateTime(2020)).source_line==1
    println("Generated archive format example passed")
    source
end
