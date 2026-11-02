module CMEGSMLabelsSynthetic
using SolarSINDy
include(joinpath(pkgdir(SolarSINDy),"research","cme_gsm_bootstrap","test_cme_gsm_labels.jl"))
run_example() = CMEGSMLabelsTests.run_generated_population_fixture()
end
