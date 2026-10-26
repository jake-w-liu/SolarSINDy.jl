"""Exercise generated numeric model export and byte reconstruction; no archival fit."""
function run_archive_point_numeric_synthetic()
    P=SolarSINDy.ArchivePointNumericTrees
    X=zeros(128,8);X[1:64,1].=.25;X[65:128,1].=.75
    y=vcat(zeros(64),fill(8.,64))
    native=SolarSINDy.gbm_fit(X,y;max_depth=2,nrounds=1,eta=1.,min_weight=1,seed=27102026,nbins=2,lambda=0.,L2=0.)
    numeric=P.from_native(native;mode=:mixed8,depth=2,rounds=1,bins=2)
    bytes=P.encode_model(numeric);loaded=P.decode_model(bytes)
    @assert P.predict(loaded,X)==y && P.predict(loaded,X)==SolarSINDy.gbm_predict(native,X)
    @assert P.encode_model(loaded)==bytes
    println("Generated archive point numeric model agrees; actual fits=0; no source-bound refit authority.")
    nothing
end
