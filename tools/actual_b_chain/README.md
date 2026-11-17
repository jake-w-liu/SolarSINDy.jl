# Ordinary B scientific execution

The matched342 entry fits the fixed L0/L1 × h4/h6 × eight-recipe grid, independently
reconstructs saved encoder/model predictions and metrics, publishes the two
reference artifacts, and reads the completed independent audit back. It uses
admitted archival source files and their data/code digests. CI logs, execution
permits and prior resource profiles are not inputs.

Run from the package checkout with one Julia thread and one BLAS/OpenMP thread:

```powershell
$env:JULIA_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$request = 'C:/experiments/b342/request.json'
$requestSha = (Get-FileHash -LiteralPath $request -Algorithm SHA256).Hash.ToLowerInvariant()
julia --startup-file=no --threads=1 --project=. tools/actual_b_chain/run_matched342_initial_actual.jl $request $requestSha
```

The request is canonical JSON in the package's ordinary data format (`T._bytes`),
with exactly these fields:

```json
{
  "schema": "matched342_initial_actual_request_v2",
  "evidence_mode": "archival_assumed_completion_research",
  "sources": [
    {"delay_hours": 0, "source_proof": {"path": "ABSOLUTE_L0_SOURCE_METADATA_JSON", "sha256": "SHA256"}, "roots": {"source": "ABSOLUTE_L0_SOURCE_ROOT"}},
    {"delay_hours": 1, "source_proof": {"path": "ABSOLUTE_L1_SOURCE_METADATA_JSON", "sha256": "SHA256"}, "roots": {"source": "ABSOLUTE_L1_SOURCE_ROOT"}}
  ],
  "output_directory": "ABSOLUTE_NEW_PRODUCER_DIRECTORY",
  "audit_directory": "ABSOLUTE_NEW_AUDIT_DIRECTORY",
  "audit_contract_sha256": "60e3e0aa40b9753fffa0a3df534185774f9eb01831f2916cd89575d61153483a"
}
```

Root labels must match those in each existing source metadata member table.
The source metadata is the existing `independently_audited_b_source_prefix_v2`
data record, with the actual archive identity, original 1976–2022 prefix and
2020–2022 validation split. The two output directories must be fresh, disjoint
from the checkout and inputs, and have existing parents. Per-fit and whole-run
time, allocation, RSS and thread observations go to stdout, outside model trees.
The audit-contract identifier above is unchanged from the original numerical
audit; it is independent of CI topology.

For ordinary B initial-L0 fitting, use `ActualBChain.BInitialPointFits.run_initial_l0`
and `verify_initial_l0` with admitted file inputs. The existing
`run_initial_l0_native.jl` compatibility module retains `reconstruct_inputs_v2`
and generated numerical evaluation helpers; its former permit/native-owner CLI
has been retired. New metadata omits qualification and execution-permit fields.
Readers tolerate only named historical fields as inert, externally hashed
provenance; no historical grant is loaded or executed.
