function private_context_checks(root,a,panel,context,model)
    v=a.view;law=panel.training.law;mask=T._masks(v,law.boundary,law.boundary,a.metadata.validation_end,0).training
    last_training=findlast(mask);@test last_training==338&&!mask[last_training+1]
    captured=Dict{Int,NamedTuple}();recipe=T.Adapter.settings()[1]
    T.Stream.stream_basis(v.input;delay_hours=0,taus=recipe.taus,sigma=recipe.sigma,chunk_hours=7,sink=_->nothing,issue_sink=p->begin
        if p.index in (last_training,last_training+1);captured[p.index]=p;end
        nothing
    end)
    inside=captured[last_training];outside=captured[last_training+1]
    called=Ref(0);badmask=copy(mask);badmask[last_training+1]=true
    run(f;training_mask=mask,boundary=law.boundary,delay_hours=0,contract_sha256=T.CONTRACT)=T._with_batch_training_context(f,v,law;fold_id="initial",boundary,delay_hours,training_mask,contract_sha256)
    @test_throws ArgumentError run(_->(called[]+=1);training_mask=badmask)
    @test_throws ArgumentError run(_->(called[]+=1);boundary=law.boundary+Hour(1))
    @test_throws ArgumentError run(_->(called[]+=1);delay_hours=true)
    @test_throws ArgumentError run(_->(called[]+=1);contract_sha256=repeat("0",64))
    @test called[]==0
    run() do c
        @test T._payload(c,inside,1)===inside
        before=copy(c.scope.counts)
        @test_throws ArgumentError T._payload(c,outside,1)
        @test_throws ArgumentError T._payload(c,merge(inside,(;driver_receipt=inside.issue+Millisecond(1))),1)
        @test_throws ArgumentError T._payload(c,merge(inside,(;issue_basis=zeros(3,2))),1)
        b=c.binding;fields=NamedTuple{fieldnames(typeof(b))}(ntuple(i->getfield(b,i),fieldcount(typeof(b))))
        wrong=typeof(b)(values(merge(fields,(;fold_id="different-fold")))...)
        wrong_context=T._BatchTrainingContext(wrong,c.view,c.law,c.mask,c.scope,c.beta_index)
        @test_throws ArgumentError T._payload(wrong_context,inside,1)
        @test c.scope.counts==before
    end
    current=T.D.current_state(v.wind,outside.issue)
    @test outside.issue<law.boundary&&current.status==:available
    @test_throws ArgumentError T.D.mean_affine_rollout(law,current,outside.issue_checkpoint,outside.issue_basis;lambdas=1 ./ recipe.taus,sigmas=(recipe.sigma,recipe.sigma),pressure=2.,target_observation_variance=0.,horizon=6)
    @test_throws MethodError T.predict_point_rows(model,panel.training)
    @test_throws MethodError T.fit_point_gains(panel.forecast,T.settings()[1])
    output=joinpath(root,"wrong-role-fit-export")
    @test_throws ArgumentError P.export_fit_input(output,context,T.settings()[1];max_chunk_rows=2) do consume
        consume(panel.forecast.rows[1:1])
    end
    @test !isfile(joinpath(output,"manifest.json"))
    oversize=joinpath(root,"over-source-cap")
    @test_throws ArgumentError P.export_fit_input(_->nothing,oversize,context,T.settings()[1];max_source_rows=500001)
    @test !ispath(oversize)
    @test_throws ArgumentError T.Adapter._shift(DateTime(Dates.UTInstant{Millisecond}(Millisecond(typemax(Int64)))),6)
    @test_throws ArgumentError T.Adapter._shift(DateTime(Dates.UTInstant{Millisecond}(Millisecond(typemin(Int64)))),-168)
    true
end
function support_failure_checks(root)
    a,ref=F.fixture(joinpath(root,"unsupported-input"),700)
    output=joinpath(root,"unsupported-stage")
    @test_throws ArgumentError B.BInitialPointFits.run_initial_l0(a,ref,output;chunk_hours=7)
    support,_=P._json(joinpath(output,"support.json"))
    @test support.all36_checked===true&&support.accepted===false&&support.gain_fit_count==0&&support.selected_index===nothing
    @test length(support.settings)==36&&getproperty.(support.settings,:setting_index)==collect(1:36)
    @test all(x->x.joint_origins==176&&x.informative_origins<100&&x.supported===false,support.settings)
    @test all(isempty(readdir(joinpath(output,name))) for name in ("fits","models","points","gain-journal"))
    @test !isfile(joinpath(output,"completion.json"))
    true
end
