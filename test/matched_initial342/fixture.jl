# Reuse genuine source-file construction only. Its old join/reference helpers
# remain unused; all new numerical expectations are independent below.
include(joinpath(@__DIR__,"..","b_input_v2","fixture.jl"))
const ORACLE=b2_json(joinpath(@__DIR__,"..","b_input_v2","source_dag_literals_v1.json"))
const GEOMETRY=b2_json(joinpath(@__DIR__,"geometry_literals_v1.json"))
const T0=DateTime(1900)

function generated_signal!(values;flat=false,after_training=false,routing_stress=false)
    # Integer arithmetic gives reproducible, variable increments. A constant
    # Dst fixture is reserved for exact ties, never full-shape resource proof.
    for i in axes(values,2)
        values[15,i]=flat ? -40.0 : -80.0+Float64(mod(17*(i-1),151))
        if after_training&&i>=400
            values[1,i]=50.0+Float64(mod(i,13))
            values[15,i]+=20.0
        end
    end
    if routing_stress
        # At issue hour i, raw row i is the latest completed driver row. These
        # four required-driver failures route to anchor persistence in both Ls.
        values[8,600]=NaN;values[6,650]=NaN;values[12,700]=NaN;values[8,730]=-1.
        values[1,620]=NaN # optional channel: encoded, not a persistence reason
    end
    values
end
function source_pair_fixture(root;flat=false,after_training=false,routing_stress=false)
    fixtures=ntuple(k->b2_source_fixture(joinpath(root,"source-L$(k-1)");delay=k-1,
        raw_change=v->generated_signal!(v;flat,after_training,routing_stress)),2)
    admissions=map(b2_admit_source,fixtures)
    pair=FC.bind_source_pair(admissions...;loss_contract_sha256=Entry.LOSS_CONTRACT_SHA256,
        expected_sources=Entry.source_identity())
    (;fixtures,admissions,pair)
end
function owned_case(f,parent,label)
    root=realpath(mktempdir(parent;prefix="matched342-$label-",cleanup=false))
    try
        result=f(root)
        # Delete only the fresh, canonical test root whose boundary is checked.
        canonical_parent=realpath(parent)
        @assert dirname(root)==canonical_parent&&startswith(basename(root),"matched342-$label-")
        @assert realpath(root)==root
        rm(root;recursive=true)
        result
    catch
        println(stderr,"MATCHED342_FAILURE_ARTIFACTS ",root)
        rethrow()
    end
end
f64words(values)=reinterpret(UInt64,vec(Float64.(values)))

function independent_input(source)
    M.HourlyInputs342(copy(source.view.input.time),copy(source.view.input.values);
        delay_hours=source.metadata.delay_hours,source_sha256=source.view.source_sha256)
end
function independent_fit_binding(source,fixture,horizon,setting)
    input=independent_input(source)
    # Both external pins refer to real generated fixture files. This is a
    # synthetic input witness, never a software/native qualification receipt.
    admission=M.SyntheticAdmission342(input;fixture_sha256=b2_file(joinpath(fixture.directory,"raw.bin")),
        external_audit_witness_sha256=fixture.proof_sha)
    m=source.metadata
    split=(;training_boundary=m.training_boundary,validation_start=m.validation_start,validation_end=m.validation_end)
    expected=M.prepare_synthetic_bindings(input,admission,horizon,setting;split...,
        loss_contract_sha256=Entry.LOSS_CONTRACT_SHA256)
    (;input,admission,split,expected)
end
