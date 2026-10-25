function check_completed_window_maximum(p,a)
    W=SolarSINDy.ArchiveCompletedWindows;A=SolarSINDy.ArchiveCompletedWindowAudit
    issue=DateTime(2020)+Day(15)+Hour(12);tag=issue-Minute(1)
    text=string(Dates.format(Date(tag),dateformat"yyyy-mm-dd"),"T",Dates.format(Time(tag),dateformat"HH:MM:SS.sss"),"Z,1,-2,400,5\n")
    kw=(;station="CMO",product="OMNI_HRO_1MIN",start=DateTime(Date(issue)),stop=DateTime(Date(issue))+Day(1))
    wp=C.build_catalog([P.parse_wind(text;kw...)]);wa=I.audit_catalog([SolarSINDy.ArchiveFormatAudit.audit_wind(text;kw...)])
    function measured(f)
        f();value=nothing;allocated=@allocated value=f();value,allocated
    end
    @testset "Maximum catalog completed-window allocation and support" begin
        for (kind,pf,af) in (
            (:ground11,()->W.prepare_catalog_issue(p,issue),()->A.audit_catalog_issue(a,issue)),
            (:mixed8,()->W.prepare_catalog_issue(p,issue;mode=:mixed8,wind=wp),()->A.audit_catalog_issue(a,issue;mode=:mixed8,wind=wa)),
            (:label,()->W.evaluate_catalog_label(p,issue,issue+Minute(31)),()->A.audit_catalog_label(a,issue,issue+Minute(31))))
            pr,pa=measured(pf);ar,aa=measured(af)
            @test pr.accepted && ar.accepted
            @test isequal(pr,ar)
            @test pa<=8*1024^2 && aa<=8*1024^2
            @test length(pr.ground_trace)==120
            if kind===:label
                @test pr.target===0.0
            else
                @test all(iszero,pr.ground11[1:7])
                @test length(pr.wind_trace)==(kind===:mixed8 ? 2 : 0)
                kind===:mixed8 && @test pr.features==(0.0,0.0,0.0,0.0,400.0,-2.0,2.0,800.0)
            end
        end
    end
end
