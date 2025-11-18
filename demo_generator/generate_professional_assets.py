import os
import json
import hashlib
import numpy as np
import pandas as pd
import cv2
from datetime import datetime

# --- CONFIGURATION ---
ASSETS_DIR = "demo_assets_advanced"
IMG_SIZE = (512, 512)
FONT = cv2.FONT_HERSHEY_SIMPLEX
GENERATOR_VERSION = "1.1.0"

# --- HELPER FUNCTIONS ---

def get_deterministic_seed(scenario_name):
    """Generate a deterministic seed from scenario name."""
    return hash(scenario_name) % (2**32)

def compute_sha256(filepath):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def draw_text(img, text, pos, scale=0.7, color=(255, 255, 255), thickness=2):
    """Utility to draw text with a shadow for better visibility."""
    cv2.putText(img, text, (pos[0]+1, pos[1]+1), FONT, scale, (0, 0, 0), thickness)
    cv2.putText(img, text, pos, FONT, scale, color, thickness)

def find_nearest_depth_index(depth_array, target_depth):
    """Find index of nearest depth value (avoids float equality issues)."""
    return np.argmin(np.abs(depth_array - target_depth))

def create_realistic_well_log(filepath, gr_sand, gr_shale, nphi_sand, nphi_shale):
    """
    Creates a more realistic well log with a 'sandwiched' reservoir.
    Shale -> Sand -> Shale
    Returns summary statistics for QA.
    """
    depth = np.arange(2500, 2700, 0.5)
    n_rows = len(depth)
    
    # Use float dtype explicitly
    gr_vals = np.full(n_rows, gr_shale, dtype=float)
    nphi_vals = np.full(n_rows, nphi_shale, dtype=float)
    
    # Define the reservoir zone (e.g., from 2580m to 2630m) - USE SAFE INDEXING
    reservoir_start_idx = find_nearest_depth_index(depth, 2580)
    reservoir_end_idx = find_nearest_depth_index(depth, 2630)
    
    gr_vals[reservoir_start_idx:reservoir_end_idx] = gr_sand
    nphi_vals[reservoir_start_idx:reservoir_end_idx] = nphi_sand
    
    # Add noise to make it look real
    gr_vals += np.random.normal(0, 4, n_rows)
    nphi_vals += np.random.normal(0, 0.015, n_rows)
    rhob_vals = 2.65 - (nphi_vals * 0.8) + np.random.normal(0, 0.03, n_rows)
    
    df = pd.DataFrame({
        'depth': depth,
        'GR': np.clip(gr_vals, 10, 200),
        'NPHI': np.clip(nphi_vals, 0.01, 0.45),
        'RHOB': np.clip(rhob_vals, 2.0, 2.9)
    })
    
    df.to_csv(filepath, index=False)
    
    # Calculate QA statistics
    reservoir_gr_mean = df.iloc[reservoir_start_idx:reservoir_end_idx]['GR'].mean()
    reservoir_nphi_mean = df.iloc[reservoir_start_idx:reservoir_end_idx]['NPHI'].mean()
    background_gr_mean = df.drop(df.index[reservoir_start_idx:reservoir_end_idx])['GR'].mean()
    
    qa_stats = {
        'mean_GR': float(df['GR'].mean()),
        'std_GR': float(df['GR'].std()),
        'mean_NPHI': float(df['NPHI'].mean()),
        'std_NPHI': float(df['NPHI'].std()),
        'mean_RHOB': float(df['RHOB'].mean()),
        'reservoir_interval': {
            'start_depth': float(depth[reservoir_start_idx]),
            'end_depth': float(depth[reservoir_end_idx]),
            'mean_GR': float(reservoir_gr_mean),
            'mean_NPHI': float(reservoir_nphi_mean)
        },
        'gr_contrast_ratio': float(background_gr_mean / reservoir_gr_mean) if reservoir_gr_mean > 0 else 0,
        'depth_monotonic': bool(df['depth'].is_monotonic_increasing),
        'missing_values': int(df.isnull().sum().sum())
    }
    
    return qa_stats

def create_logs_metadata(filepath):
    """Create metadata file for well logs."""
    metadata = {
        "columns": [
            {"name": "depth", "unit": "m", "description": "Measured depth"},
            {"name": "GR", "unit": "GAPI", "description": "Gamma Ray", "typical_range": [10, 200]},
            {"name": "NPHI", "unit": "fraction", "description": "Neutron Porosity", "typical_range": [0.01, 0.45]},
            {"name": "RHOB", "unit": "g/cc", "description": "Bulk Density", "typical_range": [2.0, 2.9]}
        ],
        "depth_reference": "MD",
        "depth_order": "ascending",
        "sample_rate_m": 0.5
    }
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=4)

def create_seismic_map(filepath, style='bright_spot'):
    """Creates a synthetic seismic map with different styles. Returns QA info."""
    img = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
    has_bright_spot = False
    
    if style == 'bright_spot':
        # Create layered background
        for y in range(0, IMG_SIZE[0], 20):
            noise = np.random.randint(-10, 10)
            cv2.line(img, (0, y + noise), (IMG_SIZE[1], y), (100, 80, 80), 1)
        
        # Add a bright spot - a strong indicator of gas
        center = (int(IMG_SIZE[1] * 0.5), int(IMG_SIZE[0] * 0.6))
        axes = (80, 25)
        cv2.ellipse(img, center, axes, 0, 0, 360, (255, 255, 200), -1)
        img = cv2.GaussianBlur(img, (15, 15), 0)
        draw_text(img, "Seismic: Strong Amplitude Anomaly (Bright Spot)", (20, 30))
        has_bright_spot = True
        
    elif style == 'flat_layers':
        # Create flat, uninteresting layers - indicates no trap
        for y in range(0, IMG_SIZE[0], 15):
            noise = np.random.randint(-5, 5)
            color_val = np.random.randint(80, 120)
            cv2.line(img, (0, y + noise), (IMG_SIZE[1], y + noise), (color_val, color_val, color_val), 2)
        draw_text(img, "Seismic: Flat Layers, No Visible Trap", (20, 30))

    cv2.imwrite(filepath, img)
    return {'has_bright_spot': has_bright_spot, 'style': style}

def create_seismic_metadata(filepath, style):
    """Create metadata file for seismic map."""
    metadata = {
        "type": "synthetic_preview",
        "width": IMG_SIZE[1],
        "height": IMG_SIZE[0],
        "channels": 3,
        "attribute": "amplitude",
        "style": style,
        "notes": "Synthetic seismic amplitude map for demonstration purposes"
    }
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=4)

def create_location_map(filepath, site_id, basin):
    """Creates a simple map showing the well's location."""
    img = np.full((IMG_SIZE[0], IMG_SIZE[1], 3), (20, 40, 20), dtype=np.uint8)
    
    # Fake coastline - BOUNDS SAFE
    pts = np.array([[min(500, IMG_SIZE[1]-1), 50], 
                    [min(450, IMG_SIZE[1]-1), 150], 
                    [min(480, IMG_SIZE[1]-1), 300], 
                    [min(400, IMG_SIZE[1]-1), min(500, IMG_SIZE[0]-1)]], np.int32)
    cv2.polylines(img, [pts], isClosed=False, color=(60, 120, 60), thickness=30)
    
    # Well location
    well_pos = (int(IMG_SIZE[1] * 0.4), int(IMG_SIZE[0] * 0.5))
    cv2.circle(img, well_pos, 10, (0, 0, 255), -1)
    cv2.circle(img, well_pos, 15, (255, 255, 255), 2)
    
    draw_text(img, f"Location Map: {basin}", (20, 30))
    draw_text(img, f"Site: {site_id}", (well_pos[0] - 50, well_pos[1] + 40))
    cv2.imwrite(filepath, img)

def create_risk_map(filepath, is_sensitive):
    """Creates a map showing operational or environmental risk."""
    img = np.full((IMG_SIZE[0], IMG_SIZE[1], 3), (230, 230, 230), dtype=np.uint8)
    well_pos = (int(IMG_SIZE[1] * 0.5), int(IMG_SIZE[0] * 0.5))

    if is_sensitive:
        cv2.rectangle(img, (50, 50), (IMG_SIZE[1] - 50, IMG_SIZE[0] - 50), (200, 200, 255), -1)
        cv2.rectangle(img, (50, 50), (IMG_SIZE[1] - 50, IMG_SIZE[0] - 50), (0, 0, 255), 3)
        draw_text(img, "Risk Map: HIGH (Protected Marine Area)", (20, 30), color=(0, 0, 180))
        draw_text(img, "ESG VETO LIKELY", (int(IMG_SIZE[1]*0.3), int(IMG_SIZE[0]*0.8)), color=(0,0,200))
    else:
        cv2.rectangle(img, (150, 150), (IMG_SIZE[1] - 150, IMG_SIZE[0] - 150), (200, 255, 200), -1)
        cv2.rectangle(img, (150, 150), (IMG_SIZE[1] - 150, IMG_SIZE[0] - 150), (0, 200, 0), 3)
        draw_text(img, "Risk Map: LOW (Standard Operations)", (20, 30), color=(0, 100, 0))

    cv2.circle(img, well_pos, 8, (0, 0, 0), -1)
    draw_text(img, "Target", (well_pos[0] + 15, well_pos[1] + 10), color=(0,0,0), scale=0.6)
    cv2.imwrite(filepath, img)

def create_site_metadata(filepath, site_id, is_protected, basin):
    """Create site metadata with enhanced fields."""
    meta = {
        "site_id": site_id,
        "operator": "Gemini Oil & Gas",
        "basin": basin,
        "protected_area": is_protected,
        "currency": "USD",
        "estimated_price_per_bbl": 75.0,
        "water_depth_m": 150,
        "coordinates": {
            "lat": 60.5 if basin == "Barents Sea" else 58.0,
            "lon": 5.0
        }
    }
    with open(filepath, 'w') as f:
        json.dump(meta, f, indent=4)

def create_expected_outcome(filepath, recommendation, rationale):
    """Create expected outcome for validation."""
    outcome = {
        "expected_recommendation": recommendation,
        "rationale": rationale
    }
    with open(filepath, 'w') as f:
        json.dump(outcome, f, indent=4)

def create_qa_report(filepath, logs_qa, seismic_qa, site_id, is_protected):
    """Create QA report with summary statistics."""
    qa_report = {
        "site_id": site_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "logs": logs_qa,
        "seismic": seismic_qa,
        "flags": {
            "good_gr_contrast": logs_qa['gr_contrast_ratio'] > 1.5,
            "reservoir_detected": logs_qa['reservoir_interval']['mean_GR'] < 80,
            "depth_monotonic": logs_qa['depth_monotonic'],
            "no_missing_values": logs_qa['missing_values'] == 0,
            "protected_area": is_protected
        }
    }
    with open(filepath, 'w') as f:
        json.dump(qa_report, f, indent=4)

def create_manifest(scenario_dir, scenario_name, site_id, basin, seed):
    """Create manifest with checksums and provenance."""
    files_info = []
    
    # List all files in scenario directory
    for filename in sorted(os.listdir(scenario_dir)):
        if filename == 'manifest.json':  # Skip manifest itself
            continue
        filepath = os.path.join(scenario_dir, filename)
        if os.path.isfile(filepath):
            file_info = {
                "filename": filename,
                "path": filename,
                "sha256": compute_sha256(filepath),
                "size_bytes": os.path.getsize(filepath),
                "mime_type": get_mime_type(filename)
            }
            files_info.append(file_info)
    
    # Create canonical inputs for input_hash
    canonical_inputs = {
        "scenario_name": scenario_name,
        "site_id": site_id,
        "basin": basin,
        "files": [{"filename": f["filename"], "sha256": f["sha256"]} for f in files_info]
    }
    
    # Compute input_hash from canonical inputs
    input_hash = hashlib.sha256(
        json.dumps(canonical_inputs, sort_keys=True).encode()
    ).hexdigest()
    
    manifest = {
        "scenario_name": scenario_name,
        "site_id": site_id,
        "basin": basin,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "files": files_info,
        "canonical_inputs": canonical_inputs,
        "input_hash": input_hash,
        "notes": f"Synthetic drilling scenario generated for demonstration purposes"
    }
    
    manifest_path = os.path.join(scenario_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
    
    return input_hash

def get_mime_type(filename):
    """Simple MIME type detection."""
    ext = filename.lower().split('.')[-1]
    mime_types = {
        'csv': 'text/csv',
        'json': 'application/json',
        'png': 'image/png',
        'txt': 'text/plain'
    }
    return mime_types.get(ext, 'application/octet-stream')

# --- MAIN ORCHESTRATOR ---

def generate_scenario(name, site_id, basin, log_params, seismic_style, is_protected, expected_rec, expected_rationale):
    """A master function to generate a complete scenario with all metadata and QA."""
    scenario_dir = os.path.join(ASSETS_DIR, name)
    print(f"\n--- Generating Scenario: {name} ---")
    
    try:
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Set deterministic seed
        seed = get_deterministic_seed(name)
        np.random.seed(seed)
        print(f"  Using seed: {seed}")
        
        # Create all assets for the scenario
        print("  Creating well logs...")
        logs_qa = create_realistic_well_log(os.path.join(scenario_dir, "logs.csv"), **log_params)
        
        print("  Creating logs metadata...")
        create_logs_metadata(os.path.join(scenario_dir, "logs_meta.json"))
        
        print("  Creating seismic map...")
        seismic_qa = create_seismic_map(os.path.join(scenario_dir, "seismic.png"), style=seismic_style)
        
        print("  Creating seismic metadata...")
        create_seismic_metadata(os.path.join(scenario_dir, "seismic_meta.json"), seismic_style)
        
        print("  Creating location map...")
        create_location_map(os.path.join(scenario_dir, "location.png"), site_id, basin)
        
        print("  Creating risk map...")
        create_risk_map(os.path.join(scenario_dir, "risk.png"), is_sensitive=is_protected)
        
        print("  Creating site metadata...")
        create_site_metadata(os.path.join(scenario_dir, "site_meta.json"), site_id, is_protected, basin)
        
        print("  Creating expected outcome...")
        create_expected_outcome(os.path.join(scenario_dir, "expected_outcome.json"), expected_rec, expected_rationale)
        
        print("  Creating QA report...")
        create_qa_report(os.path.join(scenario_dir, "qa_report.json"), logs_qa, seismic_qa, site_id, is_protected)
        
        print("  Creating manifest...")
        input_hash = create_manifest(scenario_dir, name, site_id, basin, seed)
        
        print(f"✅ Successfully created all assets for {name}")
        print(f"  Input Hash: {input_hash[:16]}...")
        
    except Exception as e:
        error_msg = f"❌ Error generating {name}: {str(e)}"
        print(error_msg)
        # Write error log
        error_log_path = os.path.join(scenario_dir, "generation_error.txt")
        with open(error_log_path, 'w') as f:
            f.write(error_msg)
        raise

if __name__ == "__main__":
    print("==============================================")
    print("   Generating Advanced Demo Asset Scenarios   ")
    print(f"   Generator Version: {GENERATOR_VERSION}")
    print("==============================================")

    # SCENARIO 1: The perfect target. Should be a clear DRILL.
    generate_scenario(
        name="SCENARIO_01_PRIME_DRILL_TARGET",
        site_id="ALPHA-PRIME-1",
        basin="Viking Graben",
        log_params={'gr_sand': 35, 'gr_shale': 120, 'nphi_sand': 0.25, 'nphi_shale': 0.08},
        seismic_style='bright_spot',
        is_protected=False,
        expected_rec="DRILL",
        expected_rationale="Excellent reservoir quality (low GR, high porosity), strong seismic bright spot indicating gas, low environmental risk"
    )
    
    # SCENARIO 2: Good seismic, but bad rocks (shale). Petrophysical agent is key. Should be HOLD.
    generate_scenario(
        name="SCENARIO_02_GEOLOGICAL_FAIL_SHALE",
        site_id="BETA-DUD-2",
        basin="Viking Graben",
        log_params={'gr_sand': 140, 'gr_shale': 150, 'nphi_sand': 0.05, 'nphi_shale': 0.04},
        seismic_style='bright_spot',
        is_protected=False,
        expected_rec="HOLD",
        expected_rationale="Despite seismic bright spot, petrophysics show high GR (shale) and very low porosity - no commercial reservoir"
    )

    # SCENARIO 3: Good rocks, but bad seismic (no trap). Seismic agent is key. Should be HOLD.
    generate_scenario(
        name="SCENARIO_03_SEISMIC_FAIL_NO_TRAP",
        site_id="GAMMA-RISK-3",
        basin="Barents Sea",
        log_params={'gr_sand': 45, 'gr_shale': 110, 'nphi_sand': 0.22, 'nphi_shale': 0.10},
        seismic_style='flat_layers',
        is_protected=False,
        expected_rec="HOLD",
        expected_rationale="Good reservoir quality rocks but seismic shows flat layers with no structural trap - hydrocarbons cannot accumulate"
    )
    
    # SCENARIO 4: Perfect geology and seismic, but in a protected area. ESG agent is key. Should be HOLD (VETO).
    generate_scenario(
        name="SCENARIO_04_ESG_VETO_SENSITIVE_AREA",
        site_id="DELTA-ECO-4",
        basin="Barents Sea",
        log_params={'gr_sand': 35, 'gr_shale': 120, 'nphi_sand': 0.25, 'nphi_shale': 0.08},
        seismic_style='bright_spot',
        is_protected=True,
        expected_rec="HOLD",
        expected_rationale="Perfect technical target but located in protected marine area - ESG concerns override commercial viability"
    )

    print("\n-----------------------------------------")
    print("🎉 All advanced asset scenarios generated!")
    print(f"   Find them in the '{ASSETS_DIR}' directory.")
    print("-----------------------------------------")