"""Source-bound initial matched342 reference composition; no fitting occurs on import."""
module BMatchedReference342
using Dates, SHA, JSON3, SolarSINDy
include("ActualBChain.jl")
const BI=ActualBChain.BInputs
const T=SolarSINDy.DurationPointTraining
const P=SolarSINDy.DurationPointArtifacts
const M=SolarSINDy.MatchedGBMPrimitives342
const PACKAGE_ROOT=realpath(normpath(joinpath(@__DIR__,"..","..")))
const FEATURE_CONTRACT_MEMBER="tools/actual_b_chain/contracts/matched342_features_v2.md"
const FEATURE_CONTRACT_SHA256="2ae937ee33d610522341b5289dab1321da5fea249fe04a4b49bcc47c9cdaa683"
const LOSS_CONTRACT_MEMBER="tools/actual_b_chain/contracts/matched342_initial_loss_v1.md"
const LOSS_CONTRACT_SHA256="f24c0dd5b9ef4baf58f0595056d5b5a30ab21ff5c278126b5ebefaa9435bc73c"
# A fixed explicit new closure. This does not alter the legacy36 source set.
const COMPONENT_SOURCE_FILES=(
    "Manifest.toml",
    "Project.toml",
    "research/dbdt_completed_source/candidate_native_owner.py",
    "research/dbdt_completed_source/native_stage.py",
    "research/dbdt_completed_source/run_candidate_stage.py",
    "src/SolarSINDy.jl",
    "src/alarm.jl",
    "src/analog.jl",
    "src/analog_features.jl",
    "src/analog_gbm.jl",
    "src/analog_serving.jl",
    "src/analog_stats.jl",
    "src/archive_dst_timing.jl",
    "src/archive_format/ArchiveCatalogAudit.jl",
    "src/archive_format/ArchiveCatalogCaseAuthority.jl",
    "src/archive_format/ArchiveCatalogCaseComparison.jl",
    "src/archive_format/ArchiveCatalogCaseJob.jl",
    "src/archive_format/ArchiveCatalogCaseReader.jl",
    "src/archive_format/ArchiveCatalogCases.jl",
    "src/archive_format/ArchiveCatalogSnapshotFilesystem.jl",
    "src/archive_format/ArchiveCatalogSnapshotValues.jl",
    "src/archive_format/ArchiveCatalogSourceJob.jl",
    "src/archive_format/ArchiveCompletedWindowAudit.jl",
    "src/archive_format/ArchiveCompletedWindows.jl",
    "src/archive_format/ArchiveFormatAudit.jl",
    "src/archive_format/ArchiveFormatParsers.jl",
    "src/archive_format/ArchiveSourceCatalog.jl",
    "src/archive_point_models/ArchiveClassicPointModels.jl",
    "src/archive_point_models/ArchiveLockedMixed8Ridge.jl",
    "src/archive_point_models/ArchivePointNumericTrees.jl",
    "src/archive_point_models/ArchiveSelectedPointBindings.jl",
    "src/archive_point_models/ArchiveSourceBoundPointFit.jl",
    "src/archive_point_models/classic_training.jl",
    "src/archive_point_models/point_fit_authority.jl",
    "src/archive_point_models/point_fit_publication.jl",
    "src/archive_point_models/selected_metadata_publication.jl",
    "src/archive_point_models/selected_point_binding_values.jl",
    "src/assimilation.jl",
    "src/baselines.jl",
    "src/calibration_shadow_trials.jl",
    "src/completed_minute_replay.jl",
    "src/completed_source_dbdt/CompletedSelectionPrimitives.jl",
    "src/completed_source_dbdt/CompletedSourceDBDT.jl",
    "src/completed_source_dbdt/adapter.jl",
    "src/completed_source_dbdt/controllers.jl",
    "src/completed_source_dbdt/label_events/CompletedLabelEvents.jl",
    "src/completed_source_dbdt/label_events/replay.jl",
    "src/completed_source_dbdt/label_events/transition.jl",
    "src/completed_source_dbdt/label_events/types.jl",
    "src/completed_source_dbdt/maximum.jl",
    "src/completed_source_dbdt/reference/dbdt_causal_forecaster.jl",
    "src/completed_source_dbdt/refit_artifacts/CompletedRefitArtifacts.jl",
    "src/completed_source_dbdt/refit_artifacts/calibration.jl",
    "src/completed_source_dbdt/refit_artifacts/codec.jl",
    "src/completed_source_dbdt/refit_artifacts/fields.jl",
    "src/completed_source_dbdt/refit_artifacts/jobs.jl",
    "src/completed_source_dbdt/refit_artifacts/numerical_audit.jl",
    "src/completed_source_dbdt/refit_artifacts/protocol.jl",
    "src/completed_source_dbdt/refit_artifacts/publication.jl",
    "src/completed_source_dbdt/refit_artifacts/ridge.jl",
    "src/completed_source_dbdt/refit_artifacts/runtime.jl",
    "src/completed_source_dbdt/refit_artifacts/source_input.jl",
    "src/completed_source_dbdt/refit_artifacts/store.jl",
    "src/completed_source_dbdt/refit_artifacts/types.jl",
    "src/conformal.jl",
    "src/data.jl",
    "src/data_cleaning.jl",
    "src/data_pipeline.jl",
    "src/duration_conditioned_forcing.jl",
    "src/duration_current_state_stream.jl",
    "src/duration_point_artifacts.jl",
    "src/duration_point_basis.jl",
    "src/duration_point_bundle.jl",
    "src/duration_point_codec.jl",
    "src/duration_point_contexts.jl",
    "src/duration_point_coordinator.jl",
    "src/duration_point_fit.jl",
    "src/duration_point_fit_artifacts.jl",
    "src/duration_point_schema.jl",
    "src/duration_point_training.jl",
    "src/duration_point_values.jl",
    "src/forecast.jl",
    "src/interval_mean_forecaster/DstIntervalAdapter.jl",
    "src/interval_mean_forecaster/DstIntervalArtifacts.jl",
    "src/interval_mean_forecaster/DstIntervalExperiment.jl",
    "src/interval_mean_forecaster/DstIntervalObserver.jl",
    "src/interval_mean_forecaster/DstIntervalReplay.jl",
    "src/interval_mean_forecaster/DstIntervalStream.jl",
    "src/interval_mean_forecaster/DstIntervalTraining.jl",
    "src/interval_mean_forecaster/IntervalMeanForecaster.jl",
    "src/interval_mean_forecaster/sandu_fullchannel_features.jl",
    "src/library.jl",
    "src/lund_dst_2002.jl",
    "src/matched_gbm_primitives/MatchedGBMPrimitives342.jl",
    "src/matched_gbm_primitives/MatchedScalarTreeReplay342.jl",
    "src/matched_gbm_primitives/codec.jl",
    "src/matched_gbm_primitives/encoder.jl",
    "src/matched_gbm_primitives/features.jl",
    "src/matched_gbm_primitives/fitting.jl",
    "src/matched_gbm_primitives/model_codec.jl",
    "src/metrics.jl",
    "src/monitor.jl",
    "src/operational_artifacts.jl",
    "src/performance_statistics.jl",
    "src/realtime.jl",
    "src/serving_depth_safe.jl",
    "src/sindy.jl",
    "src/static_stack.jl",
    "src/static_stack_arrival.jl",
    "src/static_stack_boost.jl",
    "src/static_stack_core_path.jl",
    "src/static_stack_driver.jl",
    "src/static_stack_error_exogenous.jl",
    "src/static_stack_error_state.jl",
    "src/static_stack_history.jl",
    "src/static_stack_residual.jl",
    "src/static_stack_serving.jl",
    "src/static_stack_shadow_chain.jl",
    "src/storm_selection.jl",
    "src/superlearner_serving.jl",
    "src/utils.jl",
    "tools/actual_b_chain/ActualBChain.jl",
    "tools/actual_b_chain/BInitialNativePermitV2.jl",
    "tools/actual_b_chain/BInitialPointFits.jl",
    "tools/actual_b_chain/BInputs.jl",
    "tools/actual_b_chain/BMatched342Core.jl",
    "tools/actual_b_chain/BMatchedInitialProducer.jl",
    "tools/actual_b_chain/BMatchedReference342.jl",
    "tools/actual_b_chain/BMatchedReferenceAudit342.jl",
    "tools/actual_b_chain/BReferenceAdmission.jl",
    "tools/actual_b_chain/BSelection.jl",
    "tools/actual_b_chain/contracts/matched342_features_v2.md",
    "tools/actual_b_chain/contracts/matched342_initial_loss_v1.md",
    "tools/actual_b_chain/run_initial_l0_native.jl",
    "tools/actual_b_chain/run_initial_l0_native.py",
    "tools/run_canonical_harness.py",
)
T._need(realpath(pkgdir(SolarSINDy))==PACKAGE_ROOT,"matched342 package/code checkout differs")
for name in COMPONENT_SOURCE_FILES
    Base.include_dependency(joinpath(PACKAGE_ROOT,name))
end
const LOADED_COMPONENT_SOURCES=Tuple(name=>open(io->bytes2hex(sha256(io)),joinpath(PACKAGE_ROOT,name)) for name in COMPONENT_SOURCE_FILES)
"Recheck every fixed component source byte and the maintained upstream identities."
function source_identity()
    M.source_identity();P.source_identity()
    for (name,digest) in LOADED_COMPONENT_SOURCES
        path=joinpath(PACKAGE_ROOT,name)
        P._ordinary(path)
        T._need(open(io->bytes2hex(sha256(io)),path)==digest,"loaded matched342 component source changed: "*name)
    end
    mapping=Dict(LOADED_COMPONENT_SOURCES)
    T._need(mapping[FEATURE_CONTRACT_MEMBER]==FEATURE_CONTRACT_SHA256&&mapping[LOSS_CONTRACT_MEMBER]==LOSS_CONTRACT_SHA256,
            "matched342 frozen feature/loss contract bytes differ")
    LOADED_COMPONENT_SOURCES
end
include("BMatched342Core.jl")
include("BMatchedReferenceAudit342.jl")
include("BMatchedInitialProducer.jl")
end
