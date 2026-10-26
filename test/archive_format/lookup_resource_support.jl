# The caller supplies the independently verified, complete maximum catalog.
function _lookup_resource_batch(c,tags)
    hits=0
    for tag in tags
        row=I.audit_locate(c,tag)
        row!==nothing && row.tag==tag && row.valid_mask==3 &&
            isequal(row.values,(1.0,2.0,0.0,0.0)) || error("maximum lookup witness differs")
        hits+=1
    end
    hits
end
function check_catalog_lookup_resource(c)
    first=DateTime(2020)
    indices=[1+fld(k*(length(c)-1),4095) for k in 0:4095]
    tags=[first+Minute(index-1) for index in indices]
    @test first==tags[1] && last(tags)==first+Minute(length(c)-1)
    @test length(unique(tags))==4096
    for _ in 1:3
        @test _lookup_resource_batch(c,tags)==4096
    end
    GC.gc();hits=0
    allocated=@allocated hits=_lookup_resource_batch(c,tags)
    @test hits==4096
    @test allocated<=16*1024^2
    println("GENERATED_MAXIMUM_LOOKUP_RESOURCE_VERIFIED queries=4096 allocated=",allocated," ceiling=16777216")
    flush(stdout)
end
