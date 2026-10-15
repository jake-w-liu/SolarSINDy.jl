# Read-only receipt reporting CLI; copy inputs before taking any verifier lock.
if !isdefined(@__MODULE__,:LiveReceiptMaturity)
    include(joinpath(@__DIR__,"live_receipt_maturity.jl"))
end
using .LiveReceiptMaturity,Dates,JSON3

function receipt_maturity_cli(args)
    if args==["--help"]
        println("Usage: julia --project examples/live_receipt_maturity_report.jl --log CSV --evidence-root DIRECTORY --output NEW_DIRECTORY --as-of UTC_TIMESTAMP")
        println("No fetch, source/store edit, forecast issuance or legacy057 claim promotion. Output needs COMPLETED.json.")
        return 0
    end
    allowed=Set(["--log","--evidence-root","--output","--as-of"]);options=Dict{String,String}()
    iseven(length(args)) || throw(ArgumentError("options require one value each"))
    for i in 1:2:length(args)
        name=String(args[i]);name in allowed || throw(ArgumentError("unknown receipt-report option"))
        haskey(options,name) && throw(ArgumentError("duplicate receipt-report option"))
        isempty(args[i+1]) && throw(ArgumentError("empty receipt-report option"));options[name]=String(args[i+1])
    end
    Set(keys(options))==allowed || throw(ArgumentError("all receipt-report options are required"))
    endswith(options["--as-of"],"Z") || throw(ArgumentError("explicit UTC Z clock required"))
    asof=DateTime(chopsuffix(options["--as-of"],"Z"))
    report=snapshot_report(options["--log"],options["--evidence-root"];asof)
    publication=publish_report(report,options["--output"])
    println(JSON3.write((;publication...,scope="nominal completion reporting only; issuance qualification absent",claim_ready=false)))
    0
end

if abspath(PROGRAM_FILE)==@__FILE__
    try
        exit(receipt_maturity_cli(ARGS))
    catch error
        println(stderr,"Receipt report unavailable: ",sprint(showerror,error))
        exit(2)
    end
end
