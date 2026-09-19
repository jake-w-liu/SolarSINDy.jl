module V24PointEvaluation

using DataFrames, Dates, SHA, Statistics
using SolarSINDy: operational_v22_regime
include("v2_4_point_upgrade.jl")
const P = V24PointUpgrade
const U = P.V24IntervalUpgrade

export annotate, cells, metrics, summarize, uncertainty, interval_summary,
       advancement, choose_candidate, model_names

model_names(panel) = [String.(P.POINT_MODELS)...,
    [String(c) for c in (:v2_4f, :oracle_realized, :logged_l1_ablation) if hasproperty(panel,c)]...,
    P.POINT_CANDIDATES...]
is_candidate(name) = name in P.POINT_CANDIDATES
point_column(name) = Symbol(is_candidate(name) ? name * "_point" : name)

"Attach scoring-only strata; historical fallback is not relabeled as a live driver gap."
function annotate(input, events)
    panel = copy(input)
    panel[!, :issue_day] = Date.(panel.issue_time_utc)
    panel[!, :issue_week] = panel.issue_day .- Day.(dayofweek.(panel.issue_day) .- 1)
    panel[!, :issue_regime] = String.(operational_v22_regime.(panel.latest_dst_nt,
        panel.dst_delta_1h_nt, panel.coupling_active_mvm))
    panel[!, :anchor_lag_hours] = Dates.value.(panel.issue_time_utc .- panel.latest_dst_time_utc) ./ 3_600_000
    panel[!, :storm_event] = [isfinite(y) && y <= -50 ? get(events,t,0) : 0
        for (t,y) in zip(panel.target_time_utc,panel.observation_dst_nt)]
    all(i -> !isfinite(panel.observation_dst_nt[i]) || panel.observation_dst_nt[i] > -50 ||
        panel.storm_event[i] > 0, 1:nrow(panel)) || error("scored storm lacks an event identity")
    hasproperty(panel,:driver_gap) || (panel[!, :driver_gap] = fill("unknown",nrow(panel)))
    panel[!, :driver_gap] = string.(panel.driver_gap)
    return panel
end

"Named, overlapping scoring views; target-state strata never enter residual history."
function cells(panel)
    return ["all" => trues(nrow(panel)),
        [name => panel.issue_regime .== name for name in ("quiet","active_deepening","recovery")]...,
        "storm" => panel.observation_dst_nt .<= -50,
        "deep_dst" => panel.observation_dst_nt .<= -100,
        "anchor_storm" => panel.latest_dst_nt .<= -50,
        "anchor_deep" => panel.latest_dst_nt .<= -100,
        "lag_le_1h" => panel.anchor_lag_hours .<= 1,
        "lag_gt_1h" => panel.anchor_lag_hours .> 1,
        "rate_le_minus5" => panel.dst_delta_1h_nt .<= -5,
        "rate_minus5_to_0" => (-5 .< panel.dst_delta_1h_nt .< 0),
        "rate_ge_0" => panel.dst_delta_1h_nt .>= 0,
        "coupling_zero" => panel.coupling_active_mvm .== 0,
        "coupling_positive" => panel.coupling_active_mvm .> 0,
        ["driver_" * name => panel.driver_gap .== name for name in ("true","false","unknown")]...,
        ["archive_fallback_" * string(flag) => hasproperty(panel,:fallback) ?
            panel.fallback .== flag : falses(nrow(panel)) for flag in (true,false)]...]
end

"Metrics on a shared scored panel; no candidate-specific finite-row deletion is allowed."
function metrics(raw, name)
    group = raw[isfinite.(raw.observation_dst_nt),:]
    col = point_column(name)
    hasproperty(group,col) || error("missing comparator $name")
    all(isfinite,group[!,col]) || error("nonfinite comparator $name on common scored rows")
    basic = P.point_metrics(group[!,col],group.observation_dst_nt)
    errors = group.observation_dst_nt .- group[!,col]
    q(p) = isempty(errors) ? NaN : quantile(errors,p)
    corrected = is_candidate(name) ? group[!,Symbol(name,"_corrected")] : falses(nrow(group))
    lo,hi = if is_candidate(name)
        (group[!,Symbol(name,"_lower")],group[!,Symbol(name,"_upper")])
    elseif name == "v2_4e"
        (group.static_lo_nt,group.static_hi_nt)
    else
        (fill(NaN,nrow(group)),fill(NaN,nrow(group)))
    end
    band = nrow(group)>0 && all(isfinite,[lo;hi])
    band && !(all(lo .< hi) && all(lo .<= group[!,col] .<= hi)) && error("invalid scored interval")
    score = band ? U.interval_score.(lo,hi,group.observation_dst_nt) : Float64[]
    static_score = U.interval_score.(group.static_lo_nt,group.static_hi_nt,group.observation_dst_nt)
    key_hash = bytes2hex(sha256(collect(codeunits(join(sort(group.row_key),"\n")))))
    return (;basic...,raw_n=nrow(raw),corrected_n=count(corrected),
        event_n=length(unique(filter(>(0),group.storm_event))),
        corrected_event_n=length(unique(filter(>(0),group.storm_event[corrected]))),
        row_key_sha256=key_hash,
        error_min_nt=isempty(errors) ? NaN : minimum(errors),error_q01_nt=q(.01),
        error_q05_nt=q(.05),error_q95_nt=q(.95),error_q99_nt=q(.99),
        error_max_nt=isempty(errors) ? NaN : maximum(errors),
        coverage=band ? mean(lo .<= group.observation_dst_nt .<= hi) : NaN,
        lower_misses=band ? count(group.observation_dst_nt .< lo) : 0,
        upper_misses=band ? count(group.observation_dst_nt .> hi) : 0,
        mean_width_nt=band ? mean(hi.-lo) : NaN,
        interval_score_nt=band ? mean(score) : NaN,
        width_ratio=band ? mean(hi.-lo)/mean(group.static_hi_nt.-group.static_lo_nt) : NaN,
        score_difference_nt=band ? mean(score.-static_score) : NaN,
        cap_n=is_candidate(name) ? count(group[!,Symbol(name,"_cap_applied")]) : 0,
        projection_n=is_candidate(name) ? count(group[!,Symbol(name,"_projection_applied")]) : 0,
        changed_n=is_candidate(name) ? count(!=(0),group[!,Symbol(name,"_actual_shift")]) : 0,
        mean_shift_nt=is_candidate(name) && nrow(group)>0 ? mean(group[!,Symbol(name,"_actual_shift")]) : 0.0)
end

function summarize(panel; dataset, scope, daily=false)
    rows = NamedTuple[]
    views = daily ? [string(day) => panel.issue_day .== day for day in sort(unique(panel.issue_day))] : cells(panel)
    for (cell,mask) in views, step in (daily ? (0,) : (0,P.STEPS...))
        selected = panel[mask .& (step==0 ? trues(nrow(panel)) : panel.model_step_hours .== step),:]
        for name in model_names(panel)
            push!(rows,(;dataset,scope,cell,model_step_hours=step,candidate=name,metrics(selected,name)...))
        end
    end
    return DataFrame(rows)
end

"Persist paired point-gain draws on issuance days/weeks or independent storm events."
function uncertainty(panel; dataset, scope)
    rows,draws = NamedTuple[],NamedTuple[]
    for cell in ("all","storm"), step in (0,P.STEPS...)
        mask = isfinite.(panel.observation_dst_nt) .& (cell=="all" ? trues(nrow(panel)) : panel.storm_event .> 0)
        step==0 || (mask .&= panel.model_step_hours .== step)
        group = panel[mask,:]
        blocks = cell=="storm" ? group.storm_event : dataset=="live" ? group.issue_day : group.issue_week
        for name in P.POINT_CANDIDATES, reference in ("v2_4e","static_v2_2")
            result = P.paired_rmse_gain(group[!,point_column(name)],group[!,Symbol(reference)],
                group.observation_dst_nt,blocks)
            identity = (;dataset,scope,cell,model_step_hours=step,candidate=name,reference)
            push!(rows,(;identity...,n=nrow(group),block_n=length(unique(blocks)),
                gain_nt=result.gain_nt,lower_nt=result.lower_nt,upper_nt=result.upper_nt))
            for (draw,gain_nt) in enumerate(result.draws)
                push!(draws,(;identity...,draw,gain_nt))
            end
        end
    end
    return (summary=DataFrame(rows),draws=DataFrame(draws))
end

"Live interval controls retain their own availability; A3 uses only actually issued bands."
function interval_summary(panel; dataset, scope, start, stop)
    start < stop || throw(ArgumentError("interval evaluation start must precede stop"))
    panel=panel[(start .<= panel.issue_time_utc .< stop),:]
    rows,draws = NamedTuple[],NamedTuple[]
    for name in (P.POINT_CANDIDATES...,"v2_4e","A3_issued")
        mask = isfinite.(panel.observation_dst_nt)
        if name=="A3_issued"
            hasproperty(panel,:a3_status) || continue
            mask .&= (panel.a3_status .== "ok") .& isfinite.(panel.a3_lo_nt) .& isfinite.(panel.a3_hi_nt)
        end
        group = panel[mask,:]
        lo,hi = name=="A3_issued" ? (group.a3_lo_nt,group.a3_hi_nt) : name=="v2_4e" ?
            (group.static_lo_nt,group.static_hi_nt) : (group[!,Symbol(name,"_lower")],group[!,Symbol(name,"_upper")])
        all(lo .< hi) || error("invalid interval control")
        hits = Float64.(lo .<= group.observation_dst_nt .<= hi)
        delta = U.interval_score.(lo,hi,group.observation_dst_nt) .-
            U.interval_score.(group.static_lo_nt,group.static_hi_nt,group.observation_dst_nt)
        cc = U.bootstrap_days(hits,group.issue_day)
        sc = U.bootstrap_days(delta,group.issue_day)
        present = Set(group.issue_day)
        full_days = [day for day in Date(ceil(start,Day)):Day(1):(Date(floor(stop,Day))-Day(1)) if day in present]
        full = Set(full_days)
        weeks = Float64[]
        for day in full_days
            window = Set(day+Day(k) for k in 0:6)
            issubset(window,full) || continue
            push!(weeks,mean(hits[in.(group.issue_day,Ref(window))]))
        end
        push!(rows,(;dataset,scope,candidate=name,n=nrow(group),
            row_key_sha256=bytes2hex(sha256(collect(codeunits(join(sort(group.row_key),"\n"))))),
            coverage=isempty(hits) ? NaN : mean(hits),complete_utc_days=length(full_days),
            minimum_seven_day_coverage=isempty(weeks) ? NaN : minimum(weeks),
            coverage_ci_lower=cc.lower,coverage_ci_upper=cc.upper,
            score_ci_lower_nt=sc.lower,score_ci_upper_nt=sc.upper,
            mean_width_nt=isempty(hits) ? NaN : mean(hi.-lo),
            width_ratio=isempty(hits) ? NaN : mean(hi.-lo)/mean(group.static_hi_nt.-group.static_lo_nt),
            score_difference_nt=isempty(hits) ? NaN : mean(delta)))
        for i in eachindex(cc.draws)
            push!(draws,(;dataset,scope,candidate=name,draw=i,coverage=cc.draws[i],score_difference_nt=sc.draws[i]))
        end
    end
    return (summary=DataFrame(rows),draws=DataFrame(draws))
end

function unique_row(table; kwargs...)
    mask = trues(nrow(table))
    for (column,value) in kwargs
        mask .&= table[!,column] .== value
    end
    selected = table[mask,:]
    nrow(selected)==1 || error("expected exactly one decision input row for $(kwargs), found $(nrow(selected))")
    return first(eachrow(selected))
end

"Every frozen requirement yields a visible Boolean gate; absent support fails closed."
function advancement(live_metrics,live_gain,live_interval,historical_metrics,historical_gain)
    required=Set([String.(P.POINT_MODELS)...,P.POINT_CANDIDATES...])
    for table in (live_metrics,historical_metrics[0],historical_metrics[1])
        for group in groupby(table,[:cell,:model_step_hours])
            issubset(required,Set(group.candidate)) || error("incomplete decision comparator set")
            allunique(group.candidate) || error("duplicate decision metric")
            all(==(first(group.n)),group.n) || error("decision metrics do not share scored rows")
            all(==(first(group.row_key_sha256)),group.row_key_sha256) || error("decision row identities differ")
            all(0 .<= group.corrected_n .<= group.n) || error("invalid corrected support")
        end
    end
    gates = NamedTuple[]
    add(name,condition,dataset,delay,cell,step,pass,value,threshold) = push!(gates,
        (candidate=name,condition,dataset,delay_hours=delay,cell,model_step_hours=step,
         pass=Bool(pass),value=Float64(value),threshold=Float64(threshold)))
    select(table,name,cell,step) = unique_row(table;candidate=name,cell,model_step_hours=step)
    function best(table,cell,step)
        rows = table[(table.cell .== cell) .& (table.model_step_hours .== step) .&
            in.(table.candidate,Ref(Set([String.(P.POINT_MODELS)...,"v2_4f","logged_l1_ablation"]))),:]
        issubset(Set(String.(P.POINT_MODELS)),Set(rows.candidate)) || error("incomplete comparator set")
        all(isfinite,rows.rmse_nt) || error("nonfinite strongest-comparator inputs")
        return minimum(rows.rmse_nt)
    end
    for name in P.POINT_CANDIDATES
        for (dataset,delay,ms,gs) in [("live",0,live_metrics,live_gain),
            [("historical",delay,historical_metrics[delay],historical_gain[delay]) for delay in (0,1)]...]
            pooled = select(ms,name,"all",0)
            for ref in ("v2_4e","static_v2_2")
                g=unique_row(gs;candidate=name,reference=ref,cell="all",model_step_hours=0)
                add(name,"pooled_gain_"*ref,dataset,delay,"all",0,g.gain_nt>=.25,g.gain_nt,.25)
            end
            for step in P.STEPS
                m=select(ms,name,"all",step)
                add(name,"common_support",dataset,delay,"all",step,m.n>=40,m.n,40)
                add(name,"corrected_support",dataset,delay,"all",step,m.corrected_n>=40,m.corrected_n,40)
                for ref in ("v2_4e","static_v2_2")
                    g=unique_row(gs;candidate=name,reference=ref,cell="all",model_step_hours=step)
                    add(name,"positive_gain_"*ref,dataset,delay,"all",step,g.gain_nt>0,g.gain_nt,0)
                    add(name,"positive_lower_"*ref,dataset,delay,"all",step,g.lower_nt>0,g.lower_nt,0)
                end
                loss = m.n>0 ? m.rmse_nt-best(ms,"all",step) : NaN
                add(name,"strongest_comparator",dataset,delay,"all",step,loss<=.5,loss,.5)
            end
            dataset=="historical" || continue
            storm=select(ms,name,"storm",0)
            add(name,"storm_rows",dataset,delay,"storm",0,storm.n>=200,storm.n,200)
            add(name,"storm_events",dataset,delay,"storm",0,storm.event_n>=5,storm.event_n,5)
            for step in P.STEPS
                m=select(ms,name,"storm",step)
                add(name,"storm_step_rows",dataset,delay,"storm",step,m.n>=30,m.n,30)
                add(name,"storm_step_events",dataset,delay,"storm",step,m.event_n>=5,m.event_n,5)
                for ref in ("v2_4e","static_v2_2")
                    g=unique_row(gs;candidate=name,reference=ref,cell="storm",model_step_hours=step)
                    add(name,"storm_positive_lower_"*ref,dataset,delay,"storm",step,g.lower_nt>0,g.lower_nt,0)
                end
                loss = m.n>0 ? m.rmse_nt-best(ms,"storm",step) : NaN
                add(name,"storm_strongest",dataset,delay,"storm",step,loss<=.5,loss,.5)
                add(name,"storm_bias",dataset,delay,"storm",step,abs(m.bias_nt)<=10,abs(m.bias_nt),10)
                add(name,"storm_coverage",dataset,delay,"storm",step,m.coverage>=.8,m.coverage,.8)
                for cell in ("active_deepening","recovery","deep_dst")
                    item=select(ms,name,cell,step)
                    item.n>0 || continue
                    loss=item.rmse_nt-best(ms,cell,step)
                    add(name,"state_strongest",dataset,delay,cell,step,loss<=.5,loss,.5)
                end
            end
        end
        int=unique_row(live_interval;candidate=name)
        add(name,"complete_days","live",0,"all",0,int.complete_utc_days>=7,int.complete_utc_days,7)
        add(name,"pooled_coverage_low","live",0,"all",0,int.coverage>=.88,int.coverage,.88)
        add(name,"pooled_coverage_high","live",0,"all",0,int.coverage<=.92,int.coverage,.92)
        add(name,"weekly_coverage","live",0,"all",0,int.minimum_seven_day_coverage>=.80,int.minimum_seven_day_coverage,.80)
        add(name,"width_ratio","live",0,"all",0,int.width_ratio<=1.25+32eps(1.25),int.width_ratio,1.25)
        add(name,"mean_score","live",0,"all",0,int.score_difference_nt<=0,int.score_difference_nt,0)
        add(name,"coverage_ci_lower","live",0,"all",0,int.coverage_ci_lower>=.85,int.coverage_ci_lower,.85)
        add(name,"coverage_ci_contains90_low","live",0,"all",0,int.coverage_ci_lower<=.9,int.coverage_ci_lower,.9)
        add(name,"coverage_ci_contains90_high","live",0,"all",0,int.coverage_ci_upper>=.9,int.coverage_ci_upper,.9)
        add(name,"score_ci_upper","live",0,"all",0,int.score_ci_upper_nt<=0,int.score_ci_upper_nt,0)
        for step in P.STEPS
            m=select(live_metrics,name,"all",step)
            add(name,"step_coverage","live",0,"all",step,m.coverage>=.85,m.coverage,.85)
        end
    end
    return DataFrame(gates)
end

function choose_candidate(gates, ranks)
    isempty(gates) && return nothing
    all(name -> name in P.POINT_CANDIDATES,gates.candidate) || error("unknown candidate in gates")
    Set(gates.candidate)==Set(P.POINT_CANDIDATES) || error("candidate gate set is incomplete")
    key_columns=[:condition,:dataset,:delay_hours,:cell,:model_step_hours]
    expected=nothing
    for name in P.POINT_CANDIDATES
        subset=gates[gates.candidate .== name,key_columns]
        keys=Set(Tuple(row) for row in eachrow(subset))
        length(keys)==nrow(subset) || error("duplicate advancement gate")
        expected===nothing ? (expected=keys) : keys==expected || error("candidate gates differ")
    end
    passing = [name for name in P.POINT_CANDIDATES if any(==(name),gates.candidate) &&
        all(gates.pass[gates.candidate .== name])]
    isempty(passing) && return nothing
    order = NamedTuple[]
    for name in passing
        row=unique_row(ranks;candidate=name)
        all(isfinite,(row.worst_step_gain_nt,row.live_rmse_nt)) || error("nonfinite candidate rank")
        push!(order,(candidate=name,gain=row.worst_step_gain_nt,rmse=row.live_rmse_nt,
            index=findfirst(==(name),P.POINT_CANDIDATES)))
    end
    sort!(order;by=r -> (-r.gain,r.rmse,r.index))
    return first(order).candidate
end

end
