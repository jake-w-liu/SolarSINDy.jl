#!/usr/bin/env julia
# Fail-closed manifest closure for canonical top-level data artifacts.

isdefined(@__MODULE__, :validation_output_paths) ||
    include(joinpath(@__DIR__, "output_paths.jl"))
isdefined(@__MODULE__, :verify_artifact_closure) ||
    include(joinpath(@__DIR__, "canonical_provenance.jl"))
isdefined(@__MODULE__, :CANONICAL_DATA_ARTIFACT_INVENTORY) ||
    include(joinpath(@__DIR__, "canonical_artifact_inventory.jl"))

function _canonical_inventory_sha256(data_inventory, figure_inventory)
    rows = ["$kind\0$name\0$(abspath(String(producer)))"
            for (kind, inventory) in (("data", data_inventory),
                                      ("figure", figure_inventory))
            for (name, producer) in inventory]
    return bytes2hex(SHA.sha256(join(sort!(rows), "\n")))
end

function _write_canonical_run_receipt(paths, data_result, figure_result;
                                      data_inventory, figure_inventory,
                                      package_root)
    receipt_path = joinpath(paths.root, "canonical_run_receipt.json")
    manifest_path = receipt_path * ".manifest.json"
    artifacts = [(kind=kind, basename=basename(path), path=path,
                  sha256=provenance_sha256(path), bytes=filesize(path),
                  manifest_path=path * ".manifest.json",
                  manifest_sha256=provenance_sha256(path * ".manifest.json"),
                  expected_producer=abspath(String(inventory[basename(path)])))
                 for (kind, result, inventory) in
                     (("data", data_result, data_inventory),
                      ("figure", figure_result, figure_inventory))
                 for path in result.artifacts if haskey(inventory, basename(path))]
    sort!(artifacts; by=row -> (row.kind, row.basename))
    inventory_sha256 = _canonical_inventory_sha256(
        data_inventory, figure_inventory,
    )
    receipt = (
        schema_version=1,
        run_mode=String(paths.mode),
        inventory_sha256,
        source_identity=provenance_source_identity(package_root),
        artifacts,
        timestamp_utc=Dates.format(
            now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sss",
        ) * "Z",
    )
    had_receipt = isfile(receipt_path)
    had_manifest = isfile(manifest_path)
    staging = mktempdir(paths.root)
    receipt_backup = joinpath(staging, "receipt.backup")
    manifest_backup = joinpath(staging, "manifest.backup")
    had_receipt && cp(receipt_path, receipt_backup)
    had_manifest && cp(manifest_path, manifest_backup)
    try
        _provenance_atomic_json(receipt_path, receipt)
        input_paths = Dict{String,String}()
        for row in artifacts
            prefix = "$(row.kind)__$(row.basename)"
            input_paths["$(prefix)__artifact"] = row.path
            input_paths["$(prefix)__manifest"] = row.manifest_path
        end
        write_output_manifest(receipt_path;
            producer_script=@__FILE__,
            input_paths,
            selection_record=(
                kind="canonical_artifact_closure_receipt",
                inventory_sha256,
                required_data_artifacts=length(data_inventory),
                required_figure_artifacts=length(figure_inventory),
            ),
            deterministic=true,
            metadata=(artifacts=length(artifacts),),
            package_root,
            mode=paths.mode,
        )
        verify_output_manifest(receipt_path;
            package_root,
            require_canonical=paths.mode == :canonical,
        )
    catch
        if had_receipt
            _provenance_atomic_replace(receipt_backup, receipt_path)
        elseif isfile(receipt_path) && !islink(receipt_path)
            rm(receipt_path; force=true)
        end
        if had_manifest
            _provenance_atomic_replace(manifest_backup, manifest_path)
        elseif isfile(manifest_path) && !islink(manifest_path)
            rm(manifest_path; force=true)
        end
        rethrow()
    finally
        rm(staging; recursive=true, force=true)
    end
    return receipt_path
end

function run_artifact_closure(paths=validation_output_paths();
                              data_inventory=CANONICAL_DATA_ARTIFACT_INVENTORY,
                              figure_inventory=CANONICAL_FIGURE_ARTIFACT_INVENTORY,
                              package_root=normpath(joinpath(@__DIR__, "..")))
    paths.explicit || error("artifact closure requires SOLARSINDY_OUTPUT_ROOT")
    paths.mode in (:canonical, :test) || error(
        "artifact closure requires canonical mode or explicit test mode",
    )
    result = verify_artifact_closure(paths.data;
        expected_artifacts=data_inventory,
        package_root,
        require_canonical=paths.mode == :canonical,
    )
    figure_result = verify_artifact_closure(paths.figs;
        extensions=(".pdf", ".png", ".svg"),
        expected_artifacts=figure_inventory,
        package_root,
        require_canonical=paths.mode == :canonical,
    )
    receipt = _write_canonical_run_receipt(
        paths, result, figure_result;
        data_inventory, figure_inventory, package_root,
    )
    println("Verified $(result.count) data and $(figure_result.count) figure artifacts")
    println("Canonical run receipt: $receipt")
    return (; data=result, figures=figure_result, receipt)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run_artifact_closure()
end
