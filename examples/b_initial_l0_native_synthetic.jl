# Generated inputs only; this is neither an actual permit nor source qualification.
module BInitialNativeSyntheticExample
using SolarSINDy
include(joinpath(pkgdir(SolarSINDy),"tools","actual_b_chain","run_initial_l0_native.jl"))

function run_b_initial_l0_native_synthetic()
    mktempdir() do temporary
        generated=BInitialNativeStage.run_generated(realpath(temporary))
        result=generated.result
        result["complete"]===true && result["evidence_mode"]=="synthetic_only" &&
            result["actual_source_access"]===false || error("generated B1 evidence scope differs")
        result["initial_hazard_fit_count"]==2 && result["initial_gain_fit_count"]==36 &&
            result["verified_model_count"]==36 && result["verified_certificate_count"]==36 ||
            error("generated full36 handoff incomplete")
        result["verified_hazard_fit_calls"]==0 && result["verified_gain_fit_calls"]==0 &&
            result["verified_raw_replay_A_streams"]==9 && result["selected_index"]===nothing ||
            error("generated independent readback or selection scope differs")
        result["final_fit_count"]==0 && result["calibration_fit_count"]==0 &&
            result["scientific_verdict"]===false && result["serving_promotion"]===false ||
            error("generated B1 cannot establish later-stage authority")
        println("Generated B1 native-entry handoff: 2 laws, 36 gain fits, all36 independently read back; no actual source or selection authority.")
        true
    end
end
end
BInitialNativeSyntheticExample.run_b_initial_l0_native_synthetic()
