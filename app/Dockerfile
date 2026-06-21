# Space-Weather Threat Monitor — container image.
# Build:  docker build -t swmonitor app/
# Run:    docker run --rm -p 8723:8723 \
#           -v "$(pwd)/live_forecasts:/data:ro" \
#           -e SOLARSINDY_LOG=/data/live_forecast_log.csv \
#           -e SWM_HOST=0.0.0.0 swmonitor
FROM julia:1.10-bookworm

WORKDIR /app
# Resolve dependencies first (from Project.toml; a committed Manifest.toml is used if present).
COPY Project.toml Manifest.toml* ./
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

COPY src ./src
COPY public ./public

ENV SWM_HOST=0.0.0.0 SWM_PORT=8723 SOLARSINDY_LOG=/data/live_forecast_log.csv
EXPOSE 8723
CMD ["julia", "--project=.", "src/server.jl"]
