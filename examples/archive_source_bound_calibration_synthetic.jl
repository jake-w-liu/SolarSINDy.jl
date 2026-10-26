module ArchiveSourceBoundCalibrationExample
include(joinpath(@__DIR__,"..","research","dbdt_completed_source","source_bound_calibration.jl"))
end
if !isdefined(@__MODULE__,:run_archive_source_bound_point_fit_synthetic)
    include(joinpath(@__DIR__,"archive_source_bound_point_fit_synthetic.jl"))
end
"""Reload generated four-family calibration artifacts and bind the 31-phase clock."""
function run_archive_source_bound_calibration_synthetic()
    run_archive_source_bound_point_fit_synthetic(;calibration_job=ArchiveSourceBoundCalibrationExample.SourceBoundCalibration)
    println("Generated source-bound calibration publication and replay projection agree; actual fits=0.")
    nothing
end
