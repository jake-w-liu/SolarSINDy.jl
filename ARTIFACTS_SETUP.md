# Data & Artifacts Setup

## Current Status (Development)

Data files are stored locally in `SolarSINDy.jl/data/` and loaded via `get_data_dir()` function, which is available to users as part of the public API.

```julia
using SolarSINDy
data_dir = get_data_dir()
```

## Future: GitHub Releases (Pkg.add Installation)

For package manager installations, the data will be distributed as a Julia Artifact with the following workflow:

### Step 1: Create GitHub Release
When publishing v0.1.0, create a release with:
- File: `data.tar.gz` (contents: 30 CSV/DAT files, ~X MB)
- SHA256: `f7c145c81956c27ced7ffc23069ea54bd3b114b940bf06385d7862792961b9d9`

```bash
cd SolarSINDy.jl
tar -czf data.tar.gz data/
shasum -a 256 data.tar.gz  # Verify SHA256
git tag v0.1.0
git push --tags
# Create GitHub release with data.tar.gz attached
```

### Step 2: Update Artifacts.toml
Update the download URL in `Artifacts.toml`:
```toml
[[SolarSINDyData]]
arch = "*"
git-tree-sha1 = "..."
os = "*"

[SolarSINDyData.download]
sha256 = "f7c145c81956c27ced7ffc23069ea54bd3b114b940bf06385d7862792961b9d9"
url = "https://github.com/YOUR_ORG/SolarSINDy.jl/releases/download/v0.1.0/data.tar.gz"
```

### Step 3: Register Package
Once registered in JuliaHub, users installing via `Pkg.add("SolarSINDy")` will have data automatically downloaded and cached on first use.

## Current Architecture

- **Project.toml**: Includes `Artifacts` (built-in Julia stdlib)
- **Artifacts.toml**: Template ready for future configuration
- **src/utils.jl**: `get_data_dir()` function with local fallback
- **examples/storm_monitor.jl**: Uses `get_data_dir()` for data loading
- **data/**: 30 files (CSV metrics, coefficients, validation data)

## Files Requiring No Changes

- Local development: `data/` directory works as-is
- Tests and validation scripts: Use existing data paths
- LaTeX/paper: References only to discovered coefficients (not file locations)
