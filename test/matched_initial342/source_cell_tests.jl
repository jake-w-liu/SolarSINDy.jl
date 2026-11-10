function source_and_cell_checks(root)
    f=source_pair_fixture(root)
    @test FC.recheck_pair(f.pair)
    @test FC.source_admissions(f.pair)===f.admissions
    meta=FC.source_pair_metadata(f.pair)
    @test meta.evidence_mode===:synthetic_only
    @test meta.source_rows==1024
    @test meta.training_history_start==T0
    @test meta.training_boundary==meta.validation_start==T0+Hour(512)
    @test meta.validation_end==meta.prefix_stop==T0+Hour(1024)
    @test meta.source_bindings[1].source_proof_sha256==f.fixtures[1].proof_sha
    @test meta.source_bindings[2].source_proof_sha256==f.fixtures[2].proof_sha
    @test meta.source_bindings[1].input_view_id!=meta.source_bindings[2].input_view_id
    @test meta.source_bindings[1].context_id!=meta.source_bindings[2].context_id
    @test meta.source_bindings[1].raw_prefix_sha256==meta.source_bindings[2].raw_prefix_sha256
    @test meta.source_bindings[1].input_prefix_sha256==meta.source_bindings[2].input_prefix_sha256
    @test length(Entry.source_identity())>length(P.source_identity())
    @test all(p->p in Entry.source_identity(),P.source_identity())
    @test_throws ArgumentError FC.bind_source_pair(reverse(f.admissions)...;
        loss_contract_sha256=Entry.LOSS_CONTRACT_SHA256,expected_sources=Entry.source_identity())
    @test_throws ArgumentError FC.bind_source_pair(f.admissions[1],f.admissions[1];
        loss_contract_sha256=Entry.LOSS_CONTRACT_SHA256,expected_sources=Entry.source_identity())
    @test_throws ArgumentError FC.bind_source_pair(f.admissions...;
        loss_contract_sha256=repeat("0",64),expected_sources=Entry.source_identity())
    @test_throws ArgumentError FC.bind_source_pair(f.admissions...;
        loss_contract_sha256=Entry.LOSS_CONTRACT_SHA256,expected_sources=P.source_identity())
    @test_throws MethodError FC.bind_source_pair(f.fixtures[1].values,f.fixtures[2].values;
        loss_contract_sha256=Entry.LOSS_CONTRACT_SHA256,expected_sources=Entry.source_identity())

    for oracle in GEOMETRY["cells"]
        delay=oracle["delay"];h=oracle["horizon"]
        cell=FC.prepare_cell(f.pair,delay,h);c=FC.cell_metadata(cell)
        @test c.training_rows==collect(oracle["train_first"]:oracle["train_last"])
        @test c.lead_validation_rows==collect(oracle["validation_first"]:oracle["validation_last"])
        @test c.validation_union_rows==collect(512:oracle["union_last"])
        @test length(c.training_rows)==oracle["train_count"]
        @test length(c.lead_validation_rows)==oracle["validation_count"]
        @test length(c.validation_union_rows)==oracle["union_count"]
        @test c.union_issues==[T0+Hour(i) for i in c.validation_union_rows]
        @test c.union_anchors==[f.fixtures[delay+1].values[15,i-delay] for i in c.validation_union_rows]
        @test c.max_fit_target_complete==T0+Hour(oracle["max_fit_complete_hour"])
        @test c.max_fit_label_receipt==T0+Hour(oracle["max_fit_receipt_hour"])
        @test c.max_validation_target_complete==T0+Hour(oracle["max_validation_complete_hour"])
        @test c.max_validation_label_receipt==T0+Hour(oracle["max_validation_receipt_hour"])
        @test c.max_fit_label_receipt+Hour(168)==c.training_boundary
        @test c.max_validation_label_receipt+Hour(168)==c.validation_end
        @test c.model_training_mask_sha256!=c.reference_training_mask_sha256
        @test length(unique(FC.expected_bindings(cell,s).fit_input_sha256 for s in 1:8))==1
        @test collect(cell.encoder.training_rows)==c.training_rows
        for (j,i) in enumerate(c.validation_union_rows)
            if i<=oracle["validation_last"]
                @test c.union_targets[j]==f.fixtures[delay+1].values[15,i+h]
            else
                @test c.union_targets[j]===nothing
            end
        end
        if h==6
            @test c.validation_union_rows[end-1:end]==GEOMETRY["h4_only_issue_hours"][delay+1]
            @test all(isnothing,c.union_targets[end-1:end])
        end
        # Mutating a detached projection cannot change the checked fit buffers.
        c.training_rows[1]=-1;c.union_anchors[1]=123456.
        @test FC.recheck_cell(cell)
        @test first(FC.cell_metadata(cell).training_rows)==oracle["train_first"]
        # The mutable preparation caches are sealed, including X and y bytes.
        old=cell.standardized[1,1];cell.standardized[1,1]=old+1.
        @test_throws ArgumentError FC.recheck_cell(cell)
        cell.standardized[1,1]=old
        @test FC.recheck_cell(cell)
        @test_throws ArgumentError FC.expected_bindings(cell,0)
        @test_throws ArgumentError FC.expected_bindings(cell,9)
    end
    @test_throws ArgumentError FC.prepare_cell(f.pair,0,5)
    @test_throws ArgumentError FC.prepare_cell(f.pair,2,4)
    # A valid independently admitted different source cannot be paired by delay.
    other=b2_source_fixture(joinpath(root,"different-physical-L1");delay=1,
        raw_change=v->(v[1,1]+=1.))
    @test_throws ArgumentError FC.bind_source_pair(f.admissions[1],b2_admit_source(other);
        loss_contract_sha256=Entry.LOSS_CONTRACT_SHA256,expected_sources=Entry.source_identity())
    raw=f.admissions[1].view.input.values;original=raw[1,1];raw[1,1]+=1.
    @test_throws ArgumentError FC.recheck_pair(f.pair)
    raw[1,1]=original
    @test FC.recheck_pair(f.pair)
end

function training_only_encoder_checks(root)
    first_root=mkdir(joinpath(root,"original"));other_root=mkdir(joinpath(root,"validation-changed"))
    f=source_pair_fixture(first_root);g=source_pair_fixture(other_root;after_training=true)
    @test FC.source_pair_metadata(f.pair).pair_id!=FC.source_pair_metadata(g.pair).pair_id
    for (delay,h) in ((0,4),(0,6),(1,4),(1,6))
        a=FC.prepare_cell(f.pair,delay,h);b=FC.prepare_cell(g.pair,delay,h)
        ma=FC.cell_metadata(a);mb=FC.cell_metadata(b)
        @test ma.training_rows==mb.training_rows
        @test ma.training_design_sha256==mb.training_design_sha256
        @test ma.preprocessing_sha256==mb.preprocessing_sha256
        @test ma.fit_input_sha256!=mb.fit_input_sha256
        @test f64words(a.standardized[ma.training_rows,:])==f64words(b.standardized[mb.training_rows,:])
        @test f64words(a.targets)==f64words(b.targets)
    end
end
