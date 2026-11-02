"""Run every minute-label oracle and the full551-event generated population."""
module CMEGSMLabelsPackageControls
include(joinpath(@__DIR__,"..","research","cme_gsm_bootstrap","test_cme_gsm_labels.jl"))
run_package_native_fixture()=CMEGSMLabelsTests.run_generated_population_fixture()
end
