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
