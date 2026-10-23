function selector_checks(root)
    v=view_fixture()
    refs=ntuple(k->admit(write_reference(joinpath(root,"joint-ref-L$(k-1)"),v;delay=k-1)),2)
    start,stop=refs[1].metadata.validation_start,refs[1].metadata.validation_end
    function candidates(reference,error)
        [T.BPointPredictionRow(r.issue,(r.target[1]+error[1],r.target[2]+error[2]),r.anchor,r.target,true,:synthetic_selector_unit,r.source_sha256,r.delay_hours) for r in reference.rows]
    end
    function panels(error)
        S.BJointPanels(ntuple(k->S.BL0Panels([candidates(refs[k],error(i)) for i in 1:36]),2))
    end
    # Independently prescribed four-cell errors: baseline=(2,1,2,1),
    # candidates2/3=(1/2,1/4,1/2,1/4). Their storm-ratio mean is1/4.
    tied=panels(i->i in (2,3) ? (.5,.25) : (3.,3.))
    selected=S.select_joint_setting(tied,S.BMatchedReference(refs);validation_start=start,validation_end=stop).data
    @test selected.status==:selected&&selected.selected_index==2&&selected.provisional_index===nothing
    @test selected.cell_order==(:L0h4,:L0h6,:L1h4,:L1h6)&&length(selected.settings)==36
    @test selected.settings[2].objective==.25&&selected.settings[3].objective==.25
    @test selected.settings[2].qualified&&selected.settings[3].qualified
    @test getproperty.(selected.settings[2].metrics,:count)==[341,339,340,338]
    none=S.select_joint_setting(panels(_->(3.,3.)),S.BMatchedReference(refs);validation_start=start,validation_end=stop).data
    @test none.status==:none&&none.selected_index===nothing&&length(none.settings)==36
    provisional=S.provisional_l0(tied.delays[1],refs[1];validation_start=start,validation_end=stop).data
    @test provisional.status==:provisional_l0&&provisional.selected_index===nothing&&provisional.provisional_index==2
    missing=S.BJointPanels((S.BL0Panels(tied.delays[1].rows[1:35]),tied.delays[2]))
    @test_throws ArgumentError S.select_joint_setting(missing,S.BMatchedReference(refs);validation_start=start,validation_end=stop)
    shortened=copy(tied.delays[1].rows);shortened[1]=shortened[1][1:end-1]
    @test_throws ArgumentError S.select_joint_setting(S.BJointPanels((S.BL0Panels(shortened),tied.delays[2])),S.BMatchedReference(refs);validation_start=start,validation_end=stop)
    diagnostic=ntuple(k->S._reference(merge(refs[k].metadata,(;reference_kind=:original_reference_diagnostic_only,selected_setting_ids=[29,29])),copy(refs[k].rows)),2)
    blocked=S.select_joint_setting(tied,S.BMatchedReference(diagnostic);validation_start=start,validation_end=stop).data
    @test blocked.status==:blocked_reference&&blocked.selected_index===nothing
    zero=view_fixture(true)
    zero_refs=ntuple(k->admit(write_reference(joinpath(root,"zero-storm-ref-L$(k-1)"),zero;delay=k-1)),2)
    zero_panels=S.BJointPanels(ntuple(k->S.BL0Panels([candidates(zero_refs[k],(.5,.25)) for _ in 1:36]),2))
    blocked_storm=S.select_joint_setting(zero_panels,S.BMatchedReference(zero_refs);validation_start=start,validation_end=stop).data
    @test blocked_storm.status==:blocked_storm_reference&&blocked_storm.selected_index===nothing&&length(blocked_storm.settings)==36
    true
end
