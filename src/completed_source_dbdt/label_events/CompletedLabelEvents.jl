"""
    CompletedLabelEvents

Synthetic shared receipt/completion events and fixed static, phase, FIFO, early
and maturity subscribers. Uses assumed retrospective availability. Preparation
has no owner mutation or callbacks; commit replaces one immutable snapshot.
Actual-data provenance and publisher-UP numerical state are outside this module.
"""
module CompletedLabelEvents
using Dates, SHA
import ..CompletedSourceDBDT
const D = CompletedSourceDBDT
const CMR = D.CMR
const MINUTE_MS = Int64(60_000)
export SubscriberSpec, IssueRequest, EventReplay, prepare_time, commit_time!,
       source_certificate, subscriber_certificate, safe_terminal_period,
       warmup_membership, replay_shared

include("types.jl")
include("transition.jl")
include("replay.jl")
end
