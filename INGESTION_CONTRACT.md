# INGESTION CONTRACT — Drilling Demo (v1.1)

**Purpose.** This document is the canonical contract between the scenario *generator* and any downstream *consumers* (agents, demo UI, validators, CI). It lists required files, exact fields/types, allowed values, and validation rules. Any missing or malformed field makes a scenario **invalid**.

## Versioning & change control
- Contract version: `1.1`
- Update this file whenever fields change. Add a short entry to the `CHANGELOG` (see bottom).
- Consumers must check `ingestion_contract_version` (if present in manifest or metadata) to accept scenario.

---

## REQUIRED FILES (exact names)
Each scenario directory **must** contain the following files (exact filenames):

logs.csv
logs_meta.json
seismic.png
seismic_meta.json
location.png
risk.png
site_meta.json
expected_outcome.json
qa_report.json
manifest.json

pgsql
Copy code

No required optional files. Extra files are allowed but must not be relied upon by agents.

---

## GENERAL RULES
- All JSON files must be UTF-8 encoded and valid JSON.
- Timestamps must be ISO-8601 in UTC (e.g., `2025-11-18T03:04:48.378489Z`).
- Numeric fields must be JSON numbers (not strings).
- Units MUST match the contract (e.g., `m`, `GAPI`, `g/cc`, `fraction`).
- Filenames in `manifest.json` `files[]` must exactly match the directory filenames.
- Images must be standard PNG (lossless). `mime_type` in manifest should be `image/png`.
- The manifest `input_hash` is the SHA256 of the deterministic canonical_inputs JSON (see Manifest section).

---

## 1) site_meta.json
**Required fields (exact names & types):**
```json
{
  "site_id": "string",
  "operator": "string",
  "basin": "string",
  "protected_area": true|false,
  "currency": "string",                 // ISO currency code optional recommendation (e.g. "USD")
  "estimated_price_per_bbl": 0.0,       // numeric
  "water_depth_m": 0.0,                 // numeric (meters)
  "coordinates": { "lat": 0.0, "lon": 0.0 } // decimal degrees
}
Validation rules

site_id must be unique per scenario and match manifest.canonical_inputs.site_id if present.

coordinates.lat in [-90, 90], coordinates.lon in [-180, 180].

protected_area boolean impacts qa_report.flags.protected_area.

Example

json
Copy code
{
  "site_id": "ALPHA-PRIME-1",
  "operator": "DemoEnergy Ltd",
  "basin": "Viking Graben",
  "protected_area": false,
  "currency": "USD",
  "estimated_price_per_bbl": 80.5,
  "water_depth_m": 120.0,
  "coordinates": {"lat": 59.9, "lon": 1.2}
}
2) logs.csv
Header (case-sensitive preferred but consumers must treat case-insensitively):

Copy code
depth,GR,NPHI,RHOB
Columns (types & units)

depth — float, meters (MD or TVD depending on logs_meta.json.depth_reference)

GR — float, GAPI

NPHI — float, fraction (0.0–1.0)

RHOB — float, g/cc

Constraints

Depth must be monotonic (ascending or as indicated by logs_meta.json.depth_order). Prefer ascending.

No missing values for these columns (validator should fail otherwise).

Sample rate (spacing between depth rows) should match logs_meta.json.sample_rate_m within floating error.

CSV format

UTF-8, comma-separated, . decimal point, no thousands separator, no BOM.

No header duplicates or extra unnamed columns.

Small example

yaml
Copy code
depth,GR,NPHI,RHOB
2500.0,124.04,0.0919,2.5416
2500.5,110.99,0.0479,2.6114
3) logs_meta.json
Required fields

json
Copy code
{
  "columns": [
    {"name": "depth", "unit": "m", "description": "Measured depth"},
    {"name": "GR", "unit": "GAPI", "description": "Gamma Ray", "typical_range": [10,200]},
    {"name": "NPHI", "unit": "fraction", "description": "Neutron Porosity", "typical_range": [0.01,0.45]},
    {"name": "RHOB", "unit": "g/cc", "description": "Bulk Density", "typical_range": [2.0,2.9]}
  ],
  "depth_reference": "MD" | "TVD",
  "depth_order": "ascending" | "descending",
  "sample_rate_m": 0.5
}
Validation

columns MUST include entries for all CSV columns.

depth_reference must be either "MD" or "TVD".

sample_rate_m must be a positive float and corresponds to CSV spacing.

4) seismic.png & seismic_meta.json
seismic_meta.json required fields

json
Copy code
{
  "type": "synthetic_preview",
  "width": 512,
  "height": 512,
  "channels": 3,
  "attribute": "amplitude",
  "style": "bright_spot" | "flat_layers" | "channel_edge",
  "notes": "string (optional)"
}
Validation

width and height must match the real image dimensions.

channels typically 1 (grayscale) or 3 (RGB); must match image channels.

attribute must be "amplitude".

style must be one of the allowed enumerations.

5) manifest.json
Required fields and structure

json
Copy code
{
  "scenario_name": "string",
  "site_id": "string",
  "basin": "string",
  "created_at": "ISO-8601 timestamp (UTC)",
  "generator_version": "string",
  "seed": 12345,
  "files": [
    {
      "filename": "logs.csv",
      "path": "logs.csv",
      "sha256": "hexstring",
      "size_bytes": 26041,
      "mime_type": "text/csv"
    }
    // ... one entry per file above
  ],
  "canonical_inputs": {
     "scenario_name": "string",
     "site_id": "string",
     "basin": "string",
     "files": [
        {"filename":"logs.csv","sha256":"..."},
        ...
     ]
  },
  "input_hash": "hexstring (SHA256 of canonical_inputs JSON)",
  "notes": "string (optional)"
}
Manifest hashing rules (deterministic)

Construct canonical_inputs as a JSON object containing only deterministic fields used to generate scenario (see example above). Lists must be sorted by filename.

Serialize canonical_inputs to canonical JSON:

Use UTF-8

No whitespace variants — use json.dumps(obj, separators=(',', ':'), sort_keys=True)

Compute SHA256 of the serialized bytes and write hex string into input_hash.

Validation

For each file listed in files[], compute actual SHA256 of the file and confirm it matches the entry.

Confirm input_hash matches SHA256(canonical_inputs).

Example snippet

json
Copy code
"input_hash": "ab45abe3c7f9a6bde5ed2bc54c9dc291e75a8fd7a3d4557331d52309416b420f"
6) qa_report.json (useful fields)
qa_report.json must contain certain fields downstream agents use for quick decisions. The full QA can contain more data, but these keys are required:

json
Copy code
{
  "site_id": "string",
  "generated_at": "ISO-8601 timestamp",
  "logs": {
    "mean_GR": 99.0,
    "std_GR": 36.9,
    "mean_NPHI": 0.12,
    "std_NPHI": 0.07,
    "mean_RHOB": 2.55,
    "reservoir_interval": {
      "start_depth": 2580.0,
      "end_depth": 2630.0,
      "mean_GR": 35.47,
      "mean_NPHI": 0.25
    },
    "gr_contrast_ratio": 3.39,
    "depth_monotonic": true,
    "missing_values": 0
  },
  "seismic": {
    "has_bright_spot": true,
    "style": "bright_spot"
  },
  "flags": {
    "good_gr_contrast": true,
    "reservoir_detected": true,
    "depth_monotonic": true,
    "no_missing_values": true,
    "protected_area": false
  }
}
Notes

gr_contrast_ratio = background_GR / reservoir_mean_GR or equivalent metric you define. Document the exact formula in your generator code and keep it stable.

flags.* are booleans derived from QA thresholds (listed below).

Recommended QA thresholds

good_gr_contrast true if gr_contrast_ratio >= 1.5 (adjust policy as you like).

reservoir_detected true if reservoir_interval.mean_NPHI >= 0.15 and mean_GR below threshold relative to background.

no_missing_values true if missing_values == 0.

7) expected_outcome.json
Required fields

json
Copy code
{
  "expected_recommendation": "DRILL" | "HOLD",
  "rationale": "string (human readable, shown in slides)"
}
Notes

expected_recommendation is used as a ground-truth label for demo evaluation.

rationale should be brief (max ~300 chars) explaining core reason (e.g., "Low GR + high porosity + bright spot → DRILL").

8) location.png & risk.png
location.png — small site map/thumbnail; PNG recommended size ~400×300 but flexible.

risk.png — risk heatmap or gauge used in slides.

Include these in manifest with correct mime_type: image/png.

JSON Schemas (machine-usable)
Below are minimal JSON Schema examples you can save as schemas/*.json and use with jsonschema for validation.

site_meta.schema.json
json
Copy code
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["site_id","operator","basin","protected_area","currency","estimated_price_per_bbl","water_depth_m","coordinates"],
  "properties": {
    "site_id": {"type":"string"},
    "operator": {"type":"string"},
    "basin": {"type":"string"},
    "protected_area": {"type":"boolean"},
    "currency": {"type":"string"},
    "estimated_price_per_bbl": {"type":"number"},
    "water_depth_m": {"type":"number"},
    "coordinates": {
      "type":"object",
      "required":["lat","lon"],
      "properties":{"lat":{"type":"number","minimum":-90,"maximum":90},"lon":{"type":"number","minimum":-180,"maximum":180}}
    }
  }
}
manifest.schema.json (abridged)
json
Copy code
{
  "$schema":"http://json-schema.org/draft-07/schema#",
  "type":"object",
  "required":["seed","created_at","files","input_hash"],
  "properties":{
    "seed":{"type":"number"},
    "created_at":{"type":"string","format":"date-time"},
    "files":{"type":"array","items":{
      "type":"object",
      "required":["filename","path","sha256","size_bytes","mime_type"]
    }},
    "input_hash":{"type":"string"}
  }
}
(You can add more comprehensive schemas from the fields above.)

Validation commands (quick)
(PowerShell) compute sha256:

powershell
Copy code
Get-FileHash -Algorithm SHA256 demo_assets_advanced\SCENARIO_01_PRIME_DRILL_TARGET\logs.csv
(Bash) pretty print manifest:

bash
Copy code
jq . demo_assets_advanced/SCENARIO_01_PRIME_DRILL_TARGET/manifest.json
Python quick validation

Use jsonschema to validate JSON files.

Use the provided validate_scenario.py (recommended) that:

Checks file presence

Validates manifest SHA256 values

Validates CSV headers & types

Checks seismic image dims vs seismic_meta.json

Verifies input_hash

CI Recommendations
Add tools/validate_scenario.py to the repo.

Add a GitHub Action that runs the validator for each scenario on PRs.

Fail PR if any required field is missing or if any SHA mismatch occurs.

Run validate_scenario.py after generator changes.

Error Codes & Troubleshooting (consumer-facing)
When a validator or consumer fails a scenario, return one of these standard errors:

ERR_MISSING_FILE — one or more required files are missing

ERR_INVALID_JSON — JSON parse error

ERR_SCHEMA_MISMATCH — JSON Schema validation failed

ERR_SHA_MISMATCH — checksum mismatch with manifest

ERR_CSV_FORMAT — missing header / non-numeric / NaNs in logs.csv

ERR_IMAGE_MISMATCH — seismic_meta.json dims differ from actual image

ERR_QA_FAIL — QA flags indicate invalid scenario policy (optional warning)

Each error should include:

scenario_name, file, detail text, and validator_version.

Reproducibility & Determinism
Generator MUST set and save a numeric seed in manifest.json.

All RNGs used must be seeded (np.random.seed(seed), random.seed(seed), torch.manual_seed(seed) if used).

Files that contain timestamps or non-deterministic content must not alter deterministic file bytes unless the timestamp is intentionally part of manifest (prefer static timestamps in synthetic assets).

When regenerating a scenario for a given seed, input_hash, and generator version, all file SHA256 values should be identical. If not — debug nondeterministic writes.

Example manifest (minimal)
json
Copy code
{
  "scenario_name":"SCENARIO_01_PRIME_DRILL_TARGET",
  "site_id":"ALPHA-PRIME-1",
  "basin":"Viking Graben",
  "created_at":"2025-11-18T03:04:48.503419Z",
  "generator_version":"1.1.0",
  "seed":3585169560,
  "files":[
     {"filename":"logs.csv","path":"logs.csv","sha256":"617f...2e11","size_bytes":26041,"mime_type":"text/csv"},
     {"filename":"seismic.png","path":"seismic.png","sha256":"e3f8...2f2","size_bytes":26335,"mime_type":"image/png"}
  ],
  "canonical_inputs":{"scenario_name":"SCENARIO_01_PRIME_DRILL_TARGET","site_id":"ALPHA-PRIME-1","basin":"Viking Graben","files":[]},
  "input_hash":"ab45abe3c7f9a6bd..."
}
CHANGELOG (example)
v1.0 — initial contract (basic fields)

v1.1 — added sample_rate_m, depth_order, JSON Schema suggestions, QA thresholds