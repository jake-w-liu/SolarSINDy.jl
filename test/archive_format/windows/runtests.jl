module ArchiveCompletedWindowPrimaryTests
const ROLE="primary"
include("contract_body.jl")
end
module ArchiveCompletedWindowAuditTests
const ROLE="audit"
include("contract_body.jl")
end
