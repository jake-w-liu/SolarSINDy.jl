The Wind metadata capture reads the fixed NASA `WI_H0_MFI` master skeleton and
three HAPI `info` resources. It never requests the HAPI `data` endpoint or runs a
model. Python 3.11 or newer and its standard library are sufficient.

Run `python -B -X utf8 tools/capture_wind_metadata.py --output /absolute/new-output`.
The output parent must exist. The capture creates a new directory and refuses
to overwrite an existing capture. Each response is limited to 2 MiB; at most
two requests run concurrently. HTTPS certificate and hostname checks remain
enabled, and redirects must retain the same metadata resource and group.

Raw bodies, a small set of response headers, hashes and per-resource receipts
are retained, including bounded partial bodies and HTTP error responses.
`CAPTURE.json` is written after all resource attempts finish. A successful exit
requires a nonempty HTTP 200 master response and at least one nonempty HTTP 200
HAPI info response. This establishes transport only. It does not certify the
metadata semantics, historical availability, scientific results or source
access permission. Those require independent review of the original metadata
and the experiment's admission protocol before magnetic values are read.

The separate Wind metadata workflow runs on candidate/main pushes and can be
triggered manually after registration. Its transport fixtures are also included
in the complete Julia package tests. The original 18-job CI is unchanged.


Each fixed resource runs in its own Python worker. The owner gives the request
45 seconds, then kills and reaps a blocked worker, allowing a separate 5-second
reaping grace. The four attempts share a 120-second transport deadline with
cleanup time reserved; at most two workers are owned concurrently. A late
zero exit does not count as successful transport. Workers create no children,
and their standard output and standard error use the null device.

Worker receipts are staged separately from authoritative owner receipts. After
reaping, the owner checks bounded JSON, exact resource identity and retained
body hashes. Persisted partial bytes remain available after timeout. Damaged
or oversized evidence is retained without admission and receives an explicit
failure record. Receipt or body admission failure cannot grant transport or
scientific authority. Cancellation reaps an active child before propagating.
