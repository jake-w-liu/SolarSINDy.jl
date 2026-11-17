module TLSTransportTests

using Test, SolarSINDy, HTTP, Reseau, Dates

@testset "Outbound TLS transport and verification" begin
    resources = normpath(joinpath(dirname(pathof(HTTP)), "..", "test", "resources"))
    certificate_path = joinpath(resources, "localhost-only.crt")
    server_config = Reseau.TLS.Config(verify_peer=false, cert_file=certificate_path,
        key_file=joinpath(resources, "localhost-only.key"))
    trusted_config = Reseau.TLS.Config(ca_file=certificate_path)
    payload = """[{"time_tag":"2026-01-01T00:00:00","dst":-10},
                   {"time_tag":"2026-01-01T01:00:00","dst":-20}]"""
    listener = Reseau.TLS.listen("tcp", "127.0.0.1:0", server_config)
    server = HTTP.serve!(_ -> HTTP.Response(200, payload), listener)
    trusted_client = HTTP.Client(transport=HTTP.Transport(tls_config=trusted_config, proxy=nothing))
    untrusted_client = HTTP.Client(transport=HTTP.Transport(proxy=nothing))
    calls = Ref(0)
    function checked_get(url; kwargs...)
        calls[] += 1
        @test !haskey(kwargs, :require_ssl_verification)
        return HTTP.get(url; kwargs..., client=trusted_client,
                        require_ssl_verification=true,
                        retries=0)
    end
    try
        port = HTTP.port(server)
        url = "https://localhost:$port/"
        times, dst = fetch_swpc_dst(; url, max_retries=1, http_get=checked_get)
        @test times == [DateTime(2026, 1, 1), DateTime(2026, 1, 1, 1)]
        @test dst == [-10.0, -20.0]
        @test calls[] == 1
        # The certificate is trusted but its localhost identity cannot authenticate an IP URL.
        @test_throws r"(?i)(certificate|not valid for host)" fetch_swpc_dst(;
            url="https://127.0.0.1:$port/", max_retries=1, http_get=checked_get,
        )
        untrusted_get(url; kwargs...) = HTTP.get(url; kwargs...,
            client=untrusted_client, require_ssl_verification=true,
            retries=0)
        @test_throws r"(?i)(certificate|not valid for host)" fetch_swpc_dst(;
            url, max_retries=1, http_get=untrusted_get,
        )
    finally
        close(trusted_client)
        close(untrusted_client)
        HTTP.forceclose(server)
        wait(server)
    end
end

end
