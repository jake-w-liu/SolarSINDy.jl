# Runs every reviewed literal/population/native scalar/tamper control on fixtures.
module CMEGSMBootstrapSynthetic
using SolarSINDy
include(joinpath(pkgdir(SolarSINDy),"test","cme_gsm_bootstrap_tests.jl"))
run_example()=CMEGSMBootstrapPackageControls.run_package_native_fixture()
end
