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
    println("Generated archive format example passed")
    source
end
