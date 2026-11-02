"""Fixed Python plan and Julia untrusted metadata examples; no archive campaign."""
module CandidateCampaignMetadataSynthetic
using SolarSINDy,JSON3
include(joinpath(@__DIR__,"..","test","completed_refit_artifacts","run_ci_native_chain.jl"))
include(joinpath(@__DIR__,"candidate_descriptor_preview_synthetic.jl"))
integer(value)=value isa Integer && !(value isa Bool)
function run_fixed(mode::Symbol)
    mode in (:tests,:plan) || throw(ArgumentError("fixed campaign package mode required"))
    python=CompletedRefitNativeDriver.choose_python()
    base=haskey(ENV,"RUNNER_TEMP") ? realpath(ENV["RUNNER_TEMP"]) : realpath(tempdir())
    isdir(base)&&!islink(base) || error("campaign control parent must be ordinary")
    parent=realpath(mktempdir(base;prefix="solarsindy-campaign-$(mode)-",cleanup=false))
    output=joinpath(parent,"controls")
    entry=realpath(joinpath(@__DIR__,"..","test","candidate_campaign","run_controls.py"))
    command=addenv(Cmd([python,"-B","-X","utf8",entry,"--mode",String(mode),"--output",output]),"PYTHONDONTWRITEBYTECODE"=>"1")
    CompletedRefitNativeDriver._run_support_command(command,parent)
    terminal=JSON3.read(read(joinpath(output,"TERMINAL.json"),String))
    terminal.schema=="fixed_campaign_package_controls_v1" && terminal.complete===true && terminal.mode==String(mode) || error("fixed campaign package controls failed")
    integer(terminal.native.native_exit) && terminal.native.native_exit==0 && terminal.native.timed_out===false &&
        terminal.native.leader_reaped===true && terminal.native.ownership_boundary_closed===true || error("campaign package native wait incomplete")
    terminal.Julia_executed===false && terminal.actual_data_access===false && integer(terminal.actual_fits) && terminal.actual_fits==0 &&
        terminal.source_qualified===false && terminal.scientific_qualified===false &&
        isempty(terminal.changed_controls) && isempty(terminal.control_read_errors) || error("campaign package evidence mode or controls differ")
    length(terminal.runner_guard_native_probes)==2 && all(probe->integer(probe.native.native_exit) &&
        probe.native.native_exit==probe.expected_native_exit && probe.native.timed_out===false && probe.output_created===false,
        terminal.runner_guard_native_probes) || error("campaign runner guard controls incomplete")
    if mode===:tests
        integer(terminal.methods) && terminal.methods==17 && terminal.skipped==0 || error("all17 campaign methods required")
    else
        integer(terminal.stage_count) && terminal.stage_count==142 && integer(terminal.fit_operations) && terminal.fit_operations==38 &&
            terminal.verified_stage_rows==142 && terminal.worker_launches==0 && terminal.actual_source_access===false &&
            terminal.authority_granted===false && terminal.scientific_qualification===false || error("fixed142 metadata plan differs")
    end
    terminal
end
run_package_controls()=run_fixed(:tests)
function run_example()
    plan=run_fixed(:plan)
    descriptor=CandidateDescriptorPreviewSynthetic.run_example()
    @assert descriptor.descriptor_only && descriptor.metadata_unverified && !descriptor.authority_granted
    println("FIXED142_METADATA_PLAN;INTENDED_FIT_OPERATIONS=38;ACTUAL_FITS=0;ARCHIVE_WORKERS=0;AUTHORITY=false")
    (;stage_count=plan.stage_count,intended_fit_operations=plan.fit_operations,actual_fits=0,archive_workers=0,
        descriptor_only=descriptor.descriptor_only,authority_granted=false,native_actual_qualification=false)
end
end
