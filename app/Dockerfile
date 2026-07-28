# Space-Weather Threat Monitor — dashboard + alerting container image.
# Build from the package root so the frozen operational replay is included:
#   docker build -t swmonitor -f app/Dockerfile .
# Run standalone (read-only replay of a log produced elsewhere):
#   docker run --rm -p 127.0.0.1:8723:8723 \
#     -v "$(pwd)/var/monitor:/data:ro" \
#     -e SOLARSINDY_LOG=/data/live_forecast_log.csv swmonitor
# For a supervised daemon+dashboard pair, prefer deploy/docker-compose.yml.
FROM julia:1.12.6-bookworm

# Dedicated non-root runtime user (shared uid with the monitor image so a log written mode-600 by
# the daemon is readable by the dashboard over the shared volume).
RUN useradd --create-home --uid 10001 swm
WORKDIR /app

# Resolve dependencies first (from Project.toml; a committed Manifest.toml is used if present).
COPY app/Project.toml app/Manifest.toml* ./
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

COPY app/src ./src
COPY app/public ./public
COPY app/models ./models
COPY app/vendor-plotly.sh ./vendor-plotly.sh
COPY data/operational_validation ./data/operational_validation

# Vendor Plotly at build time at a pinned version so the image is self-contained and never reaches
# cdn.plot.ly at runtime. This removes the builder-state dependence on a gitignored asset: the build
# fails loudly if the asset cannot be produced, instead of silently shipping a chartless dashboard.
ENV PLOTLY_VERSION=2.35.2
RUN bash vendor-plotly.sh && chown -R swm:swm /app

ENV SWM_HOST=0.0.0.0 SWM_PORT=8723 SOLARSINDY_LOG=/data/live_forecast_log.csv \
    SOLARSINDY_OPERATIONAL_EVIDENCE_DIR=/app/data/operational_validation JULIA_NUM_THREADS=4
EXPOSE 8723
USER swm
# Orchestrator-visible liveness: a hung or crashed server fails the check so it can be restarted.
HEALTHCHECK --interval=30s --timeout=8s --start-period=180s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${SWM_PORT}/api/health" || exit 1
CMD ["julia", "--project=.", "src/server.jl"]
