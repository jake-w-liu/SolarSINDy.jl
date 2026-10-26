using Dates, SolarSINDy
using SolarSINDy.ArchiveCatalogCases

function run_archive_catalog_cases_synthetic()
    start=DateTime(2020)
    header="Format IAGA-2002 |\nSource of Data generated example |\nIAGA CODE CMO |\nReported XYNULNUL |\nData Interval Type 1-minute |\nData Type quasi-definitive |\nDATE TIME DOY CMOX CMOY CMONUL CMONUL |\n"
    io=IOBuffer();write(io,header)
    for k in 0:90
        time=start+Minute(k)
        write(io,Dates.format(time,dateformat"yyyy-mm-dd HH:MM:SS.sss")," 001 ",string(3*k)," ",string(4*k)," 0 0\n")
    end
    body=SolarSINDy.ArchiveFormatParsers.parse_ground(String(take!(io));station="CMO",product="quasi-definitive",start,stop=start+Minute(90))
    catalog=SolarSINDy.ArchiveSourceCatalog.build_catalog([body])
    features=seal_catalog_features(catalog,collect(start:Minute(1):start+Minute(90)))
    labels=audit_catalog_labels(features,catalog;asof=start+Minute(121))
    counts=case_counts(features,labels)
    @assert counts.scheduled==91 && counts.features_accepted==60 && counts.labels_accepted==61 && counts.supervised==30
    arrays=case_arrays(features,labels)
    @assert size(arrays.features)==(30,11) && all(==(5.0),arrays.targets)
    @assert !arrays.actual_fit_performed && !arrays.source_bound_provenance_verified
    println("archive-catalog-cases: generated scheduled=91 matched=30; no fit")
    counts
end
