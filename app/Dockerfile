# Space-Weather Threat Monitor — container image.
# Build from the package root so the frozen operational replay is included:
#   docker build -t swmonitor -f app/Dockerfile .
# Run:    docker run --rm -p 8723:8723 \
#           -v "$(pwd)/var/monitor:/data:ro" \
#           -e SOLARSINDY_LOG=/data/live_forecast_log.csv \
#           -e SWM_HOST=0.0.0.0 swmonitor
FROM julia:1.12.6-bookworm

WORKDIR /app
# Resolve dependencies first (from Project.toml; a committed Manifest.toml is used if present).
COPY app/Project.toml app/Manifest.toml* ./
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

COPY app/src ./src
COPY app/public ./public
COPY app/models ./models
COPY data/operational_validation ./data/operational_validation

ENV SWM_HOST=0.0.0.0 SWM_PORT=8723 SOLARSINDY_LOG=/data/live_forecast_log.csv \
    SOLARSINDY_OPERATIONAL_EVIDENCE_DIR=/app/data/operational_validation JULIA_NUM_THREADS=4
EXPOSE 8723
CMD ["julia", "--project=.", "src/server.jl"]
