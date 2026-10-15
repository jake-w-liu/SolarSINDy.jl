# Receipt-based Dst reports

Run the read-only CLI with explicit inputs, a UTC as-of clock and an absent output directory:

```sh
julia --project --pkgimages=no examples/live_receipt_maturity_report.jl --log /path/log.csv --evidence-root /path/log.csv.observation_evidence --output /path/new_report --as-of 2026-10-15T15:52:33.620Z
```

The CLI never fetches or changes an input CSV/store. It takes no lock in the actual evidence store: it copies a complete stable inventory into a new owned temporary directory and runs the canonical journal verifier there. Active locks, changing files/head, incomplete/corrupt evidence, links and resource-budget failures return unavailable. An absent legacy store remains absent and its labels remain unknown. These checks target cooperating source writers and static path validation, not an adversarial namespace/CAS guarantee.

Snapshot entry budgets count directories as well as files; the default entry budget equals the file budget, and CSV counts in both. Source/CSV reads use bounded chunks and reject excess bytes while reading, including a file that grows after its size check. These limits bound accepted input sizes and metadata counts; they are not a strict peak-RSS promise. Directory enumeration, parsing, JSON serialization and runtime overhead remain separate. Native Windows containment resolves existing parents and uses ordinal case comparison before writes. Native leaf file-symlink behavior remains unverified on the current unprivileged Windows host; the distinct directory-junction guard was exercised.

Output is claimed as a new directory and never replaces an old report. Consumers require `COMPLETED.json`, validate all output hashes, and should check the returned completion SHA against a retained trusted receipt. An incomplete directory is not a completed report. The publication verifier checks consistency and critical bindings; its unsigned completion is not independent evidence that an arbitrary producer is authentic.

Reports retain historical served-label descriptive metrics separately from exact-manifest and actual nominal-post-end label views. First eligible nominal-post-end captures define primary scores; later revisions are sensitivity, not new support. Nominal `[provider tag,tag+1h)` completion is a software convention, not a claim of final/stable Dst or established physical sampling precision. Preliminary and legacy-unknown labels cannot become mature merely because wall time advances.

The fixed legacy057 claim input remains empty and blocked. No issuance-ledger option exists, and a newer observation consumer cannot qualify an older forecast producer. Existing comparison/claim files, identity057 and forecast arithmetic are unchanged. A genuine original-issuance qualification ledger and separate gates are required before scientific/operational claims; this helper does not provide them.

The synthetic experiment is run by `examples/experiments.jl`; it uses a mock response and new temporary paths, never a live endpoint or historical receipt reconstruction.
