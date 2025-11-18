#!/usr/bin/env python3
"""
Clean pipeline implementation for oil & gas scenario processing.
Handles ingestion, validation, and deterministic agent processing.

Usage:
    python pipeline.py --scenario demo_assets_advanced/SCENARIO_01_PRIME_DRILL_TARGET
    python pipeline.py --all
"""

import os
import sys
import json
import hashlib
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_timestamp() -> str:
    """Return ISO8601 timestamp."""
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha.update(chunk)
    return sha.hexdigest()


def compute_json_hash(obj: Dict) -> str:
    """Compute SHA256 of canonical JSON representation."""
    json_str = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def save_json(filepath: str, data: Dict) -> None:
    """Save JSON with proper formatting."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def log_message(artifacts_dir: str, message: str) -> None:
    """Append timestamped message to pipeline log."""
    log_path = os.path.join(artifacts_dir, 'pipeline_log.txt')
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"{get_timestamp()} - {message}\n")


# ============================================================================
# INGESTION MODULE
# ============================================================================

class IngestionError(Exception):
    """Raised when ingestion fails."""
    pass


def load_json_file(filepath: str) -> Dict:
    """Load and parse JSON file. Accepts UTF-8 with or without BOM."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Required file missing: {filepath}")
    # use utf-8-sig so files with a BOM decode correctly
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        return json.load(f)



def canonicalize_logs_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names in logs dataframe."""
    header_map = {}
    for col in df.columns:
        col_lower = col.strip().lower()
        if 'depth' in col_lower:
            header_map[col] = 'depth'
        elif col_lower in ('gr', 'gamma', 'gamma_ray'):
            header_map[col] = 'GR'
        elif 'nphi' in col_lower or 'phi' in col_lower:
            header_map[col] = 'NPHI'
        elif 'rhob' in col_lower or 'rho' in col_lower:
            header_map[col] = 'RHOB'
    
    return df.rename(columns=header_map)


def validate_manifest_hash(scenario_path: str, manifest: Dict) -> str:
    """
    Validate that manifest input_hash matches recomputed hash.
    Returns the validated hash.
    """
    expected_hash = manifest.get('input_hash')
    if not expected_hash:
        raise IngestionError("manifest.json missing 'input_hash' field")
    
    # Reconstruct canonical inputs
    site_meta = load_json_file(os.path.join(scenario_path, 'site_meta.json'))
    
    files_list = []
    for file_entry in manifest.get('files', []):
        filename = file_entry['filename']
        filepath = os.path.join(scenario_path, filename)
        if not os.path.exists(filepath):
            raise IngestionError(f"Manifest references missing file: {filename}")
        
        file_hash = compute_file_hash(filepath)
        files_list.append({'filename': filename, 'sha256': file_hash})
    
    canonical_inputs = {
        'scenario_name': os.path.basename(os.path.normpath(scenario_path)),
        'site_id': site_meta.get('site_id', ''),
        'basin': site_meta.get('basin', ''),
        'files': sorted(files_list, key=lambda x: x['filename'])
    }
    
    computed_hash = compute_json_hash(canonical_inputs)
    
    if computed_hash != expected_hash:
        raise IngestionError(
            f"Hash mismatch!\n"
            f"  Expected: {expected_hash}\n"
            f"  Computed: {computed_hash}\n"
            f"Manifest may be stale - regenerate with demo generator."
        )
    
    print(f"✓ Input hash validated: {computed_hash[:16]}...")
    return computed_hash


def ingest_scenario(scenario_path: str) -> Dict[str, Any]:
    """
    Ingest and validate a scenario folder.
    Returns ingest_package with all loaded data.
    """
    scenario_path = os.path.abspath(scenario_path)
    warnings = []
    assumptions = []
    
    print(f"\n{'='*60}")
    print(f"INGESTING: {scenario_path}")
    print(f"{'='*60}")
    
    # Load manifest
    manifest_path = os.path.join(scenario_path, 'manifest.json')
    manifest = load_json_file(manifest_path)
    
    # Validate hash
    input_hash = validate_manifest_hash(scenario_path, manifest)
    
    # Load metadata
    site_meta = load_json_file(os.path.join(scenario_path, 'site_meta.json'))
    logs_meta = load_json_file(os.path.join(scenario_path, 'logs_meta.json'))
    
    # Load logs CSV
    logs_csv_path = os.path.join(scenario_path, 'logs.csv')
    if not os.path.exists(logs_csv_path):
        raise IngestionError(f"Missing logs.csv in {scenario_path}")
    
    logs_df = pd.read_csv(logs_csv_path)
    logs_df = canonicalize_logs_headers(logs_df)
    
    # Validate required columns
    required_cols = ['depth', 'GR', 'NPHI']
    missing_cols = [col for col in required_cols if col not in logs_df.columns]
    if missing_cols:
        raise IngestionError(
            f"Missing required columns after canonicalization: {missing_cols}\n"
            f"Available columns: {list(logs_df.columns)}"
        )
    
    # Check NPHI scaling
    nphi_max = logs_df['NPHI'].max()
    nphi_units = logs_meta.get('nphi_units', '').lower()
    if nphi_max > 1.0 and 'frac' in nphi_units:
        warnings.append("NPHI values > 1 but metadata says 'fraction' - possible scaling issue")
        assumptions.append("nphi_scale_suspect")
    
    # Ensure depth is sorted ascending
    if not logs_df['depth'].is_monotonic_increasing:
        warnings.append("Depth not monotonic - sorting ascending")
        logs_df = logs_df.sort_values('depth').reset_index(drop=True)
        assumptions.append("depth_sorted")
    
    # Load seismic if present
    seismic_img = None
    seismic_meta = None
    seismic_path = os.path.join(scenario_path, 'seismic.png')
    
    if os.path.exists(seismic_path):
        try:
            seismic_img = Image.open(seismic_path).convert('RGB')
            seismic_meta_path = os.path.join(scenario_path, 'seismic_meta.json')
            if os.path.exists(seismic_meta_path):
                seismic_meta = load_json_file(seismic_meta_path)
            else:
                warnings.append("seismic.png found but seismic_meta.json missing")
                assumptions.append("seismic_meta_missing")
        except Exception as e:
            warnings.append(f"Failed to load seismic.png: {e}")
            seismic_img = None
    
    site_id = str(site_meta.get('site_id', 'UNKNOWN'))
    
    if warnings:
        print(f"⚠ Warnings: {len(warnings)}")
        for w in warnings:
            print(f"  - {w}")
    
    print(f"✓ Ingestion complete for site_id: {site_id}")
    
    return {
        'scenario_path': scenario_path,
        'manifest': manifest,
        'site_meta': site_meta,
        'logs_meta': logs_meta,
        'logs_df': logs_df,
        'seismic_img': seismic_img,
        'seismic_meta': seismic_meta,
        'input_hash': input_hash,
        'site_id': site_id,
        'warnings': warnings,
        'assumptions': assumptions
    }


# ============================================================================
# PETROPHYSICAL AGENT
# ============================================================================

def run_petrophysical_agent(ingest_pkg: Dict, artifacts_dir: str) -> Dict:
    """
    Compute petrophysical properties from well logs.
    Returns artifact dictionary.
    """
    print("\n--- Running Petrophysical Agent ---")
    
    logs_df = ingest_pkg['logs_df'].copy()
    logs_meta = ingest_pkg['logs_meta']
    site_id = ingest_pkg['site_id']
    input_hash = ingest_pkg['input_hash']
    assumptions = list(ingest_pkg['assumptions'])
    
    # Handle NPHI scaling
    nphi_max = logs_df['NPHI'].max()
    if nphi_max > 1.0 and logs_meta.get('nphi_units', '').lower() in ['fraction', 'frac']:
        logs_df['NPHI'] = logs_df['NPHI'] / 100.0
        assumptions.append("nphi_scaled_by_100")
        print("  Scaled NPHI by 100 (appeared to be percentage)")
    
    # Compute metrics (deterministic)
    mean_gr = float(logs_df['GR'].mean())
    mean_nphi = float(logs_df['NPHI'].mean())
    
    # Porosity estimate: clipped mean NPHI
    porosity_est = float(np.clip(mean_nphi, 0.02, 0.35))
    
    # Sand probability: inverse relation with GR
    sand_prob = float(np.clip(1.0 - mean_gr / 180.0, 0.0, 1.0))
    
    # Confidence based on GR variability
    gr_std = float(logs_df['GR'].std(ddof=0))
    confidence = float(np.clip(0.5 + 0.5 * (1.0 - gr_std / 100.0), 0.0, 1.0))
    
    # Uncertainty estimates
    epistemic = 1.0 - confidence
    nphi_std = float(logs_df['NPHI'].std(ddof=0))
    aleatoric = float(np.clip(nphi_std / 0.35, 0.0, 1.0))
    
    # SHAP-like feature importance (based on variance)
    var_gr = float(logs_df['GR'].var(ddof=0))
    var_nphi = float(logs_df['NPHI'].var(ddof=0))
    var_rhob = float(logs_df['RHOB'].var(ddof=0)) if 'RHOB' in logs_df.columns else 0.0
    
    total_var = var_gr + var_nphi + var_rhob
    if total_var > 0:
        shap_features = [
            {'feature': 'GR', 'shap_value': var_gr / total_var},
            {'feature': 'NPHI', 'shap_value': var_nphi / total_var},
            {'feature': 'RHOB', 'shap_value': var_rhob / total_var}
        ]
    else:
        shap_features = [
            {'feature': 'GR', 'shap_value': 1/3},
            {'feature': 'NPHI', 'shap_value': 1/3},
            {'feature': 'RHOB', 'shap_value': 1/3}
        ]
    
    # Lithology probabilities
    lithology = {
        'sand': sand_prob,
        'shale': max(0.0, 1.0 - sand_prob),
        'silt': 0.0
    }
    
    # Create output directory
    site_dir = os.path.join(artifacts_dir, site_id)
    os.makedirs(site_dir, exist_ok=True)
    
    timestamp = get_timestamp()
    
    # Generate plot
    plot_path = os.path.join(site_dir, 'petrophysical_plot.png')
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10), sharey=True)
        
        ax1.plot(logs_df['GR'], logs_df['depth'], 'b-', linewidth=1)
        ax1.set_xlabel('Gamma Ray (API)', fontsize=10)
        ax1.set_ylabel('Depth (m)', fontsize=10)
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('GR Log')
        
        ax2.plot(logs_df['NPHI'], logs_df['depth'], 'g-', linewidth=1)
        ax2.set_xlabel('Neutron Porosity', fontsize=10)
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('NPHI Log')
        
        fig.suptitle(f'Site: {site_id} | Sand Prob: {sand_prob:.2f}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        plot_hash = compute_file_hash(plot_path)
        print(f"  ✓ Plot saved: {plot_path}")
    except Exception as e:
        print(f"  ✗ Plot failed: {e}")
        plot_hash = None
    
    # Build artifact
    artifact = {
        'artifact_id': f"petro-{site_id}-{timestamp}",
        'agent': 'petrophysical',
        'model_version': 'demo-v1',
        'inputs': {
            'files_consumed': [
                os.path.join(ingest_pkg['scenario_path'], 'logs.csv'),
                os.path.join(ingest_pkg['scenario_path'], 'logs_meta.json')
            ]
        },
        'outputs': {
            'porosity_estimate': porosity_est,
            'lithology_probs': lithology
        },
        'metrics': {
            'confidence': confidence
        },
        'uncertainty': {
            'epistemic': epistemic,
            'aleatoric': aleatoric
        },
        'explanation': {
            'explanation_text': f"Porosity = mean(NPHI) clipped to [0.02, 0.35]. Sand probability uses GR scale of 180 API.",
            'shap_top_features': shap_features
        },
        'attachments': [
            {
                'name': 'petrophysical_plot.png',
                'path': os.path.relpath(plot_path),
                'sha256': plot_hash
            }
        ],
        'provenance': {
            'input_hash': input_hash,
            'created_at': timestamp
        },
        'assumptions': assumptions
    }
    
    # Save artifact
    artifact_path = os.path.join(site_dir, 'petrophysical.json')
    save_json(artifact_path, artifact)
    log_message(site_dir, f"Petrophysical agent completed -> {artifact_path}")
    
    print(f"  ✓ Porosity: {porosity_est:.3f}")
    print(f"  ✓ Sand prob: {sand_prob:.3f}")
    print(f"  ✓ Artifact saved: {artifact_path}")
    
    return artifact


# ============================================================================
# SEISMIC AGENT
# ============================================================================

def run_seismic_agent(ingest_pkg: Dict, artifacts_dir: str) -> Dict:
    """
    Process seismic image to detect reservoir potential.
    Returns artifact dictionary.
    """
    print("\n--- Running Seismic Agent ---")
    
    site_id = ingest_pkg['site_id']
    input_hash = ingest_pkg['input_hash']
    site_dir = os.path.join(artifacts_dir, site_id)
    os.makedirs(site_dir, exist_ok=True)
    timestamp = get_timestamp()
    
    # Handle missing seismic
    if ingest_pkg['seismic_img'] is None:
        print("  No seismic image - using neutral fallback")
        artifact = {
            'artifact_id': f"seismic-{site_id}-{timestamp}",
            'agent': 'seismic',
            'model_version': 'demo-v1',
            'status': 'not_run',
            'inputs': {'files_consumed': []},
            'outputs': {
                'reservoir_probability': 0.25,
                'blob_centers': []
            },
            'metrics': {'confidence': 0.1},
            'explanation': 'No seismic image provided - neutral fallback values used',
            'provenance': {
                'input_hash': input_hash,
                'created_at': timestamp
            }
        }
        artifact_path = os.path.join(site_dir, 'seismic.json')
        save_json(artifact_path, artifact)
        log_message(site_dir, "Seismic agent: no image (not_run)")
        return artifact
    
    # Process seismic image
    img_pil = ingest_pkg['seismic_img']
    img_gray = np.array(img_pil.convert('L')).astype(np.float32)
    
    # Deterministic blob detection
    img_blur = cv2.GaussianBlur(img_gray, (9, 9), 0)
    threshold = float(img_blur.mean() + 0.8 * img_blur.std())
    mask = (img_blur > threshold).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = img_gray.shape
    total_blob_area = sum(cv2.contourArea(c) for c in contours)
    
    # Reservoir probability from blob coverage
    reservoir_prob = float(np.clip((total_blob_area / (width * height)) * 5.0, 0.0, 1.0))
    
    # Extract blob centers
    blob_centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            blob_centers.append({'x': cx, 'y': cy})
    
    num_blobs = len(contours)
    confidence = float(0.4 + 0.6 * min(1.0, num_blobs / 3.0))
    
    # Create annotated image
    color_img = cv2.cvtColor(img_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(color_img, contours, -1, (0, 0, 255), 2)
    
    for center in blob_centers:
        cv2.circle(color_img, (center['x'], center['y']), 5, (255, 0, 0), -1)
    
    processed_path = os.path.join(site_dir, 'seismic_processed.png')
    Image.fromarray(color_img).save(processed_path)
    processed_hash = compute_file_hash(processed_path)
    
    # Build artifact
    artifact = {
        'artifact_id': f"seismic-{site_id}-{timestamp}",
        'agent': 'seismic',
        'model_version': 'demo-v1',
        'inputs': {
            'files_consumed': [
                os.path.join(ingest_pkg['scenario_path'], 'seismic.png')
            ]
        },
        'outputs': {
            'reservoir_probability': reservoir_prob,
            'blob_centers': blob_centers
        },
        'metrics': {
            'confidence': confidence,
            'num_blobs': num_blobs,
            'blob_coverage': total_blob_area / (width * height)
        },
        'attachments': [
            {
                'name': 'seismic_processed.png',
                'path': os.path.relpath(processed_path),
                'sha256': processed_hash
            }
        ],
        'explanation': f"Threshold = mean + 0.8*std. Detected {num_blobs} blobs. Coverage scaled by 5.0.",
        'provenance': {
            'input_hash': input_hash,
            'created_at': timestamp
        }
    }
    
    artifact_path = os.path.join(site_dir, 'seismic.json')
    save_json(artifact_path, artifact)
    log_message(site_dir, f"Seismic agent completed -> {artifact_path}")
    
    print(f"  ✓ Reservoir prob: {reservoir_prob:.3f}")
    print(f"  ✓ Blobs detected: {num_blobs}")
    print(f"  ✓ Artifact saved: {artifact_path}")
    
    return artifact


# ============================================================================
# RESERVOIR AGENT
# ============================================================================

def run_reservoir_agent(artifacts_dir: str, site_id: str, input_hash: str) -> Dict:
    """
    Fuse petrophysical and seismic results.
    Returns artifact dictionary.
    """
    print("\n--- Running Reservoir Agent ---")
    
    site_dir = os.path.join(artifacts_dir, site_id)
    
    # Load prerequisites
    petro_path = os.path.join(site_dir, 'petrophysical.json')
    seismic_path = os.path.join(site_dir, 'seismic.json')
    
    if not os.path.exists(petro_path):
        raise FileNotFoundError(f"Missing {petro_path}")
    if not os.path.exists(seismic_path):
        raise FileNotFoundError(f"Missing {seismic_path}")
    
    petro = load_json_file(petro_path)
    seismic = load_json_file(seismic_path)
    
    # Extract inputs
    porosity_est = float(petro['outputs']['porosity_estimate'])
    porosity_norm = min(1.0, porosity_est / 0.35)
    
    reservoir_prob = float(seismic['outputs']['reservoir_probability'])
    if seismic.get('status') == 'not_run':
        reservoir_prob = 0.35  # Neutral fallback
    
    # Compute reservoir quality index
    rqi = 0.6 * porosity_norm + 0.4 * reservoir_prob
    
    # Pseudo R² metric
    r2_like = 0.5 + 0.5 * min(porosity_norm, reservoir_prob)
    
    timestamp = get_timestamp()
    
    artifact = {
        'artifact_id': f"reservoir-{site_id}-{timestamp}",
        'agent': 'reservoir',
        'model_version': 'demo-v1',
        'inputs': {
            'petrophysical_artifact': petro_path,
            'seismic_artifact': seismic_path
        },
        'outputs': {
            'reservoir_quality_index': float(rqi),
            'porosity_norm': float(porosity_norm)
        },
        'metrics': {
            'r2_like': float(r2_like)
        },
        'explanation': 'RQI = 0.6*porosity_norm + 0.4*reservoir_prob',
        'provenance': {
            'input_hash': input_hash,
            'used_artifacts': [
                petro.get('artifact_id'),
                seismic.get('artifact_id')
            ],
            'created_at': timestamp
        }
    }
    
    artifact_path = os.path.join(site_dir, 'reservoir.json')
    save_json(artifact_path, artifact)
    log_message(site_dir, f"Reservoir agent completed -> {artifact_path}")
    
    print(f"  ✓ RQI: {rqi:.3f}")
    print(f"  ✓ Artifact saved: {artifact_path}")
    
    return artifact


# ============================================================================
# RISK/ESG AGENT
# ============================================================================

def run_risk_agent(artifacts_dir: str, scenario_path: str, site_id: str, input_hash: str) -> Dict:
    """
    Compute economic metrics and ESG veto.
    Returns artifact dictionary.
    """
    print("\n--- Running Risk/ESG Agent ---")
    
    site_dir = os.path.join(artifacts_dir, site_id)
    
    # Load prerequisites
    reservoir_path = os.path.join(site_dir, 'reservoir.json')
    if not os.path.exists(reservoir_path):
        raise FileNotFoundError(f"Missing {reservoir_path}")
    
    reservoir = load_json_file(reservoir_path)
    site_meta = load_json_file(os.path.join(scenario_path, 'site_meta.json'))
    
    # Compute reserves
    rqi = float(reservoir['outputs']['reservoir_quality_index'])
    reserves_bbl = rqi * 1_000_000
    
    # Economic calculation
    recovery_factor = 0.12
    price_per_bbl = site_meta.get('estimated_price_per_bbl', 75.0)
    capex_stub = 10_000_000
    
    npv_p50 = reserves_bbl * price_per_bbl * recovery_factor - capex_stub
    
    # ESG check
    protected_area = site_meta.get('protected_area', False)
    esg_status = 'FAIL' if protected_area else 'PASS'
    veto = (esg_status == 'FAIL')
    
    timestamp = get_timestamp()
    
    artifact = {
        'artifact_id': f"risk-{site_id}-{timestamp}",
        'agent': 'risk_esg',
        'model_version': 'demo-v1',
        'inputs': {
            'reservoir_artifact': reservoir_path,
            'site_meta': os.path.join(scenario_path, 'site_meta.json')
        },
        'outputs': {
            'reserves_bbl': float(reserves_bbl),
            'NPV_P50': float(npv_p50)
        },
        'esg': {
            'ESG_status': esg_status,
            'veto': bool(veto),
            'reason': 'protected_area' if veto else 'none'
        },
        'explanation': f"Reserves = RQI * 1M bbl. NPV = reserves * ${price_per_bbl}/bbl * {recovery_factor} RF - ${capex_stub:,} CAPEX",
        'provenance': {
            'input_hash': input_hash,
            'created_at': timestamp
        }
    }
    
    artifact_path = os.path.join(site_dir, 'risk_esg.json')
    save_json(artifact_path, artifact)
    log_message(site_dir, f"Risk/ESG agent completed -> {artifact_path}")
    
    print(f"  ✓ NPV P50: ${npv_p50:,.0f}")
    print(f"  ✓ ESG: {esg_status} (veto={veto})")
    print(f"  ✓ Artifact saved: {artifact_path}")
    
    return artifact


# ============================================================================
# CONSENSUS AGENT
# ============================================================================

def run_consensus_agent(artifacts_dir: str, site_id: str, input_hash: str) -> Dict:
    """
    Fuse all agent outputs into final recommendation.
    Returns artifact dictionary.
    """
    print("\n--- Running Consensus Agent ---")
    
    site_dir = os.path.join(artifacts_dir, site_id)
    
    # Load all prerequisites
    petro = load_json_file(os.path.join(site_dir, 'petrophysical.json'))
    reservoir = load_json_file(os.path.join(site_dir, 'reservoir.json'))
    risk = load_json_file(os.path.join(site_dir, 'risk_esg.json'))
    
    # Extract values
    sand_prob = float(petro['outputs']['lithology_probs']['sand'])
    rqi = float(reservoir['outputs']['reservoir_quality_index'])
    npv_p50 = float(risk['outputs']['NPV_P50'])
    veto = bool(risk['esg']['veto'])
    
    # Normalize NPV
    npv_min = -50_000_000.0
    npv_max = 150_000_000.0
    npv_norm = float(np.clip((npv_p50 - npv_min) / (npv_max - npv_min), 0.0, 1.0))
    
    # Compute final score (weighted combination)
    final_score = 0.4 * sand_prob + 0.4 * rqi + 0.2 * npv_norm
    
    # Determine recommendation
    if veto:
        recommendation = "HOLD (ESG VETO)"
    elif final_score >= 0.6:
        recommendation = "DRILL"
    else:
        recommendation = "HOLD"
    
    # Build explanation bullets
    explain_bullets = [
        f"Sand probability contribution: 0.4 × {sand_prob:.3f} = {0.4 * sand_prob:.3f}",
        f"Reservoir quality contribution: 0.4 × {rqi:.3f} = {0.4 * rqi:.3f}",
        f"NPV contribution: 0.2 × {npv_norm:.3f} = {0.2 * npv_norm:.3f}",
        f"Final score: {final_score:.3f} (threshold: 0.60 for DRILL)",
        f"ESG veto: {veto}"
    ]
    
    timestamp = get_timestamp()
    
    artifact = {
        'artifact_id': f"consensus-{site_id}-{timestamp}",
        'agent': 'consensus',
        'model_version': 'demo-v1',
        'inputs': {
            'petrophysical': os.path.join(site_dir, 'petrophysical.json'),
            'reservoir': os.path.join(site_dir, 'reservoir.json'),
            'risk_esg': os.path.join(site_dir, 'risk_esg.json')
        },
        'outputs': {
            'final_score': float(final_score),
            'recommendation': recommendation,
            'sand_prob': float(sand_prob),
            'rqi': float(rqi),
            'npv_norm': float(npv_norm)
        },
        'weights': {
            'sand_weight': 0.4,
            'rqi_weight': 0.4,
            'npv_weight': 0.2
        },
        'explain_bullets': explain_bullets,
        'provenance': {
            'input_hash': input_hash,
            'used_artifacts': [
                petro.get('artifact_id'),
                reservoir.get('artifact_id'),
                risk.get('artifact_id')
            ],
            'created_at': timestamp
        }
    }
    
    artifact_path = os.path.join(site_dir, 'consensus.json')
    save_json(artifact_path, artifact)
    log_message(site_dir, f"Consensus agent completed -> {artifact_path}")
    
    print(f"  ✓ Final score: {final_score:.3f}")
    print(f"  ✓ Recommendation: {recommendation}")
    print(f"  ✓ Artifact saved: {artifact_path}")
    
    return artifact


# ============================================================================
# PIPELINE ORCHESTRATION
# ============================================================================

def run_pipeline_for_scenario(scenario_path: str, artifacts_dir: str = 'artifacts') -> Dict:
    """
    Execute complete pipeline for a single scenario.
    Returns dictionary of all artifacts.
    """
    print(f"\n{'='*70}")
    print(f"PIPELINE EXECUTION: {os.path.basename(scenario_path)}")
    print(f"{'='*70}")
    
    try:
        # Step 1: Ingest
        ingest_pkg = ingest_scenario(scenario_path)
        site_id = ingest_pkg['site_id']
        input_hash = ingest_pkg['input_hash']
        
        # Step 2: Petrophysical Agent
        petro_artifact = run_petrophysical_agent(ingest_pkg, artifacts_dir)
        
        # Step 3: Seismic Agent
        seismic_artifact = run_seismic_agent(ingest_pkg, artifacts_dir)
        
        # Step 4: Reservoir Agent
        reservoir_artifact = run_reservoir_agent(artifacts_dir, site_id, input_hash)
        
        # Step 5: Risk/ESG Agent
        risk_artifact = run_risk_agent(artifacts_dir, scenario_path, site_id, input_hash)
        
        # Step 6: Consensus Agent
        consensus_artifact = run_consensus_agent(artifacts_dir, site_id, input_hash)
        
        print(f"\n{'='*70}")
        print(f"✓ PIPELINE COMPLETE: {site_id}")
        print(f"  Recommendation: {consensus_artifact['outputs']['recommendation']}")
        print(f"  Final Score: {consensus_artifact['outputs']['final_score']:.3f}")
        print(f"  Artifacts: artifacts/{site_id}/")
        print(f"{'='*70}")
        
        return {
            'site_id': site_id,
            'input_hash': input_hash,
            'petrophysical': petro_artifact,
            'seismic': seismic_artifact,
            'reservoir': reservoir_artifact,
            'risk': risk_artifact,
            'consensus': consensus_artifact,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ PIPELINE FAILED: {e}")
        print(f"{'='*70}")
        raise


def find_all_scenarios(base_dir: str = 'demo_assets_advanced') -> List[str]:
    """Find all scenario directories in base directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    scenarios = [
        str(p) for p in base_path.iterdir() 
        if p.is_dir() and p.name.startswith('SCENARIO_')
    ]
    return sorted(scenarios)


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Oil & Gas Multi-Agent Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --scenario demo_assets_advanced/SCENARIO_01_PRIME_DRILL_TARGET
  python pipeline.py --all
  python pipeline.py --all --artifacts my_artifacts
        """
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        help='Path to specific scenario folder'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run pipeline on all scenarios in demo_assets_advanced/'
    )
    
    parser.add_argument(
        '--artifacts',
        type=str,
        default='artifacts',
        help='Output directory for artifacts (default: artifacts/)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.scenario and not args.all:
        parser.print_help()
        print("\n" + "="*70)
        print("Available scenarios:")
        print("="*70)
        scenarios = find_all_scenarios()
        if scenarios:
            for s in scenarios:
                print(f"  {s}")
        else:
            print("  No scenarios found in demo_assets_advanced/")
            print("  Run generate_professional_assets.py first!")
        print("="*70)
        sys.exit(1)
    
    # Ensure artifacts directory exists
    os.makedirs(args.artifacts, exist_ok=True)
    
    # Execute pipeline
    results = []
    
    if args.all:
        scenarios = find_all_scenarios()
        if not scenarios:
            print("ERROR: No scenarios found in demo_assets_advanced/")
            print("Run generate_professional_assets.py first!")
            sys.exit(1)
        
        print(f"\nFound {len(scenarios)} scenarios to process\n")
        
        for scenario_path in scenarios:
            try:
                result = run_pipeline_for_scenario(scenario_path, args.artifacts)
                results.append(result)
            except Exception as e:
                print(f"ERROR processing {scenario_path}: {e}")
                results.append({
                    'scenario': scenario_path,
                    'status': 'failed',
                    'error': str(e)
                })
    else:
        if not os.path.exists(args.scenario):
            print(f"ERROR: Scenario not found: {args.scenario}")
            sys.exit(1)
        
        result = run_pipeline_for_scenario(args.scenario, args.artifacts)
        results.append(result)
    
    # Print summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    
    success_count = sum(1 for r in results if r.get('status') == 'success')
    fail_count = len(results) - success_count
    
    print(f"Total scenarios: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    
    if success_count > 0:
        print("\nSuccessful scenarios:")
        for r in results:
            if r.get('status') == 'success':
                rec = r['consensus']['outputs']['recommendation']
                score = r['consensus']['outputs']['final_score']
                print(f"  {r['site_id']}: {rec} (score: {score:.3f})")
    
    if fail_count > 0:
        print("\nFailed scenarios:")
        for r in results:
            if r.get('status') != 'success':
                print(f"  {r.get('scenario', 'unknown')}: {r.get('error', 'unknown error')}")
    
    print("="*70)
    
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == '__main__':
    main()