module ArchiveCatalogCasesPrimaryTests
const ROLE="primary"
include("contract_body.jl")
end
module ArchiveCatalogCasesIndependentTests
const ROLE="audit"
include("contract_body.jl")
end
