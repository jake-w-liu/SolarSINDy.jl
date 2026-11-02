"""Private generated API spies only; no real CDF module, data or authority."""
module CMEGSMCDFReaderSynthetic
include(joinpath(@__DIR__,"..","research","cme_gsm_bootstrap","test_cdf_minute_reader.jl"))
const T=CMEGSMCDFReaderTests
function run_example()
    for big_endian in (false,true)
        T.with_raw() do path,digest
            fixture=T.fixture(;big_endian)
            time_pass=T.pass(fixture.api,path,digest;memory_guard=phase->nothing)
            @assert fixture.ds.reads==["Epoch"]
            schedule=T.G.freeze_phase_domains(time_pass.axis)
            qc=T.R._qc_pass(fixture.api,path,time_pass,schedule;memory_guard=phase->nothing)
            @assert fixture.ds.reads==["Epoch","Epoch","label_bgsm","cartesian","BGSM","NUM_PTS"]
            @assert fixture.ds.opens==fixture.ds.closes==2 && qc.bz==[-2.,0.,-7.]
            label=T.G.label_event(T.G.mo_event("generated-example",0,180000,1,2),[schedule],[qc])
            @assert label.peak_bs_nt==7 && label.valid_count==label.expected_count==3
            @assert label.extrema_supports==[(centre_ms=150000,support_start_ms=120000,support_stop_ms=180000)]
        end
    end
    println("GENERATED_CDF_API_SPIES_ONLY;PEAK_BS_LITERAL=7;ACTUAL_CDF_VALUES=0;REAL_READER_NOT_IMPORTED")
    (;actual_CDF_values=0,private_generated_API=true,peak_bs_literal=7,native_actual_qualification=false)
end
end
