module TLSTransportTests

using Test, SolarSINDy, HTTP, MbedTLS, Sockets, Dates

@testset "Outbound TLS transport and verification" begin
    resources = normpath(joinpath(dirname(pathof(HTTP)), "..", "test", "resources"))
    # SHA-256 re-signing of HTTP's public localhost test certificate with its public fixture key.
    # The original SHA-1 signature is rejected by MbedTLS's certificate profile.
    certificate_path = joinpath(@__DIR__, "fixtures", "localhost_sha256.pem")
    certificate = read(certificate_path, String)
    server_config = MbedTLS.SSLConfig(certificate_path, joinpath(resources, "key.pem"))
    trusted_config = MbedTLS.SSLConfig(true)
    MbedTLS.ca_chain!(trusted_config, MbedTLS.crt_parse(certificate))
    payload = """[{"time_tag":"2026-01-01T00:00:00","dst":-10},
                   {"time_tag":"2026-01-01T01:00:00","dst":-20}]"""
    server = HTTP.serve!(_ -> HTTP.Response(200, payload), ip"127.0.0.1", 0;
                         listenany=true, sslconfig=server_config, verbose=false)
    pool = HTTP.Pool(4)
    untrusted_pool = HTTP.Pool(4)
    original_default = HTTP.SOCKET_TYPE_TLS[]
    calls = Ref(0)
    function checked_get(url; kwargs...)
        calls[] += 1
        @test get(kwargs, :socket_type_tls, nothing) === MbedTLS.SSLContext
        @test !haskey(kwargs, :require_ssl_verification)
        return HTTP.get(url; kwargs..., sslconfig=trusted_config,
                        require_ssl_verification=true, pool,
                        retries=0, proxy=nothing)
    end
    try
        port = HTTP.port(server)
        url = "https://localhost:$port/"
        times, dst = fetch_swpc_dst(; url, max_retries=1, http_get=checked_get)
        @test times == [DateTime(2026, 1, 1), DateTime(2026, 1, 1, 1)]
        @test dst == [-10.0, -20.0]
        @test calls[] == 1
        # The certificate is trusted but its localhost identity cannot authenticate an IP URL.
        @test_throws r"Certificate verification failed" fetch_swpc_dst(;
            url="https://127.0.0.1:$port/", max_retries=1, http_get=checked_get,
        )
        untrusted_get(url; kwargs...) = HTTP.get(url; kwargs...,
            sslconfig=MbedTLS.SSLConfig(true), require_ssl_verification=true,
            pool=untrusted_pool, retries=0, proxy=nothing)
        @test_throws r"Certificate verification failed" fetch_swpc_dst(;
            url, max_retries=1, http_get=untrusted_get,
        )
        @test HTTP.SOCKET_TYPE_TLS[] === original_default
    finally
        HTTP.Connections.closeall(pool)
        HTTP.Connections.closeall(untrusted_pool)
        HTTP.forceclose(server)
        wait(server)
    end
end

end
