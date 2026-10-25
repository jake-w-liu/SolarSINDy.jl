using SolarSINDy,Dates,JSON3
function run_archive_completed_windows_synthetic()
    fixture=JSON3.read(read(joinpath(@__DIR__,"..","test","archive_format","windows","fixtures","FROZEN_INPUTS_AND_ORACLES_V1.json"),String),Any)
    kw=(;station="CMO",product="quasi-definitive",start=DateTime(2020),stop=DateTime(2020)+Minute(90))
    p=ArchiveSourceCatalog.build_catalog([ArchiveFormatParsers.parse_ground(fixture["ground"];kw...)])
    a=ArchiveCatalogAudit.audit_catalog([ArchiveFormatAudit.audit_ground(fixture["ground"];kw...)])
    issue=DateTime(2020)+Minute(32)
    row=ArchiveCompletedWindows.prepare_catalog_issue(p,issue;wind=:never_consulted)
    other=ArchiveCompletedWindowAudit.audit_catalog_issue(a,issue;wind=:never_consulted)
    @assert row.accepted && isequal(row,other) && length(row.ground_trace)==120 && isempty(row.wind_trace)
    early=ArchiveCompletedWindows.evaluate_catalog_label(p,issue,issue+Minute(30))
    @assert !early.accepted && isempty(early.ground_trace)
    mature=ArchiveCompletedWindows.evaluate_catalog_label(p,issue,issue+Minute(31))
    @assert mature.accepted && mature.target==100.0
    @assert !row.source_pins.source_bound_provenance_verified && !row.source_pins.measured_receipts_available
    println("Generated completed catalog-window example passed")
    nothing
end
