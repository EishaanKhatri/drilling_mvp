import os
import json
import hashlib
import pandas as pd
from PIL import Image

# --- CONFIGURATION ---
ASSETS_DIR = "demo_assets_advanced"

# --- VALIDATION FUNCTIONS ---

def compute_sha256(filepath):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def validate_manifest_integrity(scenario_dir, scenario_name):
    """Validate that all files in manifest exist and checksums match."""
    manifest_path = os.path.join(scenario_dir, "manifest.json")
    issues = []
    
    if not os.path.exists(manifest_path):
        issues.append("❌ manifest.json missing")
        return issues
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print(f"\n  Checking manifest integrity for {scenario_name}...")
    
    for file_info in manifest.get('files', []):
        filename = file_info['filename']
        expected_sha256 = file_info['sha256']
        filepath = os.path.join(scenario_dir, filename)
        
        if not os.path.exists(filepath):
            issues.append(f"  ❌ File missing: {filename}")
            continue
        
        actual_sha256 = compute_sha256(filepath)
        if actual_sha256 != expected_sha256:
            issues.append(f"  ❌ Checksum mismatch for {filename}")
            issues.append(f"     Expected: {expected_sha256}")
            issues.append(f"     Got:      {actual_sha256}")
        else:
            print(f"    ✅ {filename}: checksum valid")
    
    # Verify input_hash can be recomputed
    canonical_inputs = manifest.get('canonical_inputs', {})
    expected_input_hash = manifest.get('input_hash', '')
    
    recomputed_hash = hashlib.sha256(
        json.dumps(canonical_inputs, sort_keys=True).encode()
    ).hexdigest()
    
    if recomputed_hash != expected_input_hash:
        issues.append(f"  ❌ input_hash mismatch")
        issues.append(f"     Expected: {expected_input_hash}")
        issues.append(f"     Got:      {recomputed_hash}")
    else:
        print(f"    ✅ input_hash valid: {expected_input_hash[:16]}...")
    
    return issues

def validate_csv_structure(scenario_dir, scenario_name):
    """Validate CSV has required columns and good data quality."""
    csv_path = os.path.join(scenario_dir, "logs.csv")
    issues = []
    
    if not os.path.exists(csv_path):
        issues.append("  ❌ logs.csv missing")
        return issues
    
    print(f"\n  Checking CSV structure for {scenario_name}...")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        issues.append(f"  ❌ Cannot read CSV: {e}")
        return issues
    
    # Check required columns
    required_cols = ['depth', 'GR', 'NPHI', 'RHOB']
    actual_cols = [col.lower() for col in df.columns]
    
    for col in required_cols:
        if col.lower() not in actual_cols:
            issues.append(f"  ❌ Missing column: {col}")
    
    if issues:
        return issues
    
    print(f"    ✅ All required columns present")
    
    # Check depth monotonicity
    if not df['depth'].is_monotonic_increasing:
        issues.append(f"  ❌ Depth is not monotonic increasing")
    else:
        print(f"    ✅ Depth is monotonic")
    
    # Check for excessive NaN values
    nan_count = df.isnull().sum().sum()
    nan_pct = (nan_count / (len(df) * len(df.columns))) * 100
    
    if nan_pct > 20:
        issues.append(f"  ❌ Too many NaN values: {nan_pct:.1f}%")
    else:
        print(f"    ✅ NaN values: {nan_pct:.1f}%")
    
    # Check value ranges
    if df['GR'].min() < 0 or df['GR'].max() > 250:
        issues.append(f"  ⚠️  GR values outside typical range: [{df['GR'].min():.1f}, {df['GR'].max():.1f}]")
    
    if df['NPHI'].min() < 0 or df['NPHI'].max() > 1:
        issues.append(f"  ⚠️  NPHI values outside typical range: [{df['NPHI'].min():.3f}, {df['NPHI'].max():.3f}]")
    
    print(f"    ✅ Value ranges reasonable")
    
    return issues

def validate_reservoir_contrast(scenario_dir, scenario_name):
    """Validate that reservoir interval shows good contrast."""
    qa_report_path = os.path.join(scenario_dir, "qa_report.json")
    issues = []
    
    if not os.path.exists(qa_report_path):
        issues.append("  ❌ qa_report.json missing")
        return issues
    
    print(f"\n  Checking reservoir contrast for {scenario_name}...")
    
    with open(qa_report_path, 'r') as f:
        qa_report = json.load(f)
    
    gr_contrast = qa_report.get('logs', {}).get('gr_contrast_ratio', 0)
    reservoir_gr = qa_report.get('logs', {}).get('reservoir_interval', {}).get('mean_GR', 999)
    
    if gr_contrast < 1.2:
        issues.append(f"  ⚠️  Low GR contrast ratio: {gr_contrast:.2f}")
    else:
        print(f"    ✅ Good GR contrast: {gr_contrast:.2f}x")
    
    if reservoir_gr > 80:
        issues.append(f"  ⚠️  High reservoir GR (likely shale): {reservoir_gr:.1f} GAPI")
    else:
        print(f"    ✅ Reservoir GR indicates sand: {reservoir_gr:.1f} GAPI")
    
    return issues

def validate_seismic_detectability(scenario_dir, scenario_name):
    """Validate seismic map is readable and matches expected style."""
    seismic_path = os.path.join(scenario_dir, "seismic.png")
    seismic_meta_path = os.path.join(scenario_dir, "seismic_meta.json")
    issues = []
    
    if not os.path.exists(seismic_path):
        issues.append("  ❌ seismic.png missing")
        return issues
    
    print(f"\n  Checking seismic map for {scenario_name}...")
    
    try:
        img = Image.open(seismic_path)
        print(f"    ✅ Seismic image readable: {img.size[0]}x{img.size[1]}")
    except Exception as e:
        issues.append(f"  ❌ Cannot read seismic image: {e}")
        return issues
    
    if os.path.exists(seismic_meta_path):
        with open(seismic_meta_path, 'r') as f:
            meta = json.load(f)
        style = meta.get('style', 'unknown')
        print(f"    ✅ Seismic style: {style}")
    else:
        issues.append("  ⚠️  seismic_meta.json missing")
    
    return issues

def validate_metadata_completeness(scenario_dir, scenario_name):
    """Check that all expected metadata files exist."""
    required_files = [
        'manifest.json',
        'logs.csv',
        'logs_meta.json',
        'seismic.png',
        'seismic_meta.json',
        'location.png',
        'risk.png',
        'site_meta.json',
        'expected_outcome.json',
        'qa_report.json'
    ]
    
    issues = []
    print(f"\n  Checking file completeness for {scenario_name}...")
    
    for filename in required_files:
        filepath = os.path.join(scenario_dir, filename)
        if not os.path.exists(filepath):
            issues.append(f"  ❌ Missing file: {filename}")
        else:
            print(f"    ✅ {filename}")
    
    return issues

def validate_expected_outcome(scenario_dir, scenario_name):
    """Validate expected outcome is properly formatted."""
    outcome_path = os.path.join(scenario_dir, "expected_outcome.json")
    issues = []
    
    if not os.path.exists(outcome_path):
        issues.append("  ❌ expected_outcome.json missing")
        return issues
    
    print(f"\n  Checking expected outcome for {scenario_name}...")
    
    with open(outcome_path, 'r') as f:
        outcome = json.load(f)
    
    recommendation = outcome.get('expected_recommendation', '')
    rationale = outcome.get('rationale', '')
    
    if recommendation not in ['DRILL', 'HOLD']:
        issues.append(f"  ❌ Invalid recommendation: {recommendation}")
    else:
        print(f"    ✅ Expected recommendation: {recommendation}")
    
    if len(rationale) < 20:
        issues.append(f"  ⚠️  Rationale seems too short")
    else:
        print(f"    ✅ Rationale provided ({len(rationale)} chars)")
    
    return issues

def validate_scenario(scenario_dir, scenario_name):
    """Run all validations for a scenario."""
    print(f"\n{'='*60}")
    print(f"Validating: {scenario_name}")
    print(f"{'='*60}")
    
    all_issues = []
    
    # Run all validation checks
    all_issues.extend(validate_metadata_completeness(scenario_dir, scenario_name))
    all_issues.extend(validate_manifest_integrity(scenario_dir, scenario_name))
    all_issues.extend(validate_csv_structure(scenario_dir, scenario_name))
    all_issues.extend(validate_reservoir_contrast(scenario_dir, scenario_name))
    all_issues.extend(validate_seismic_detectability(scenario_dir, scenario_name))
    all_issues.extend(validate_expected_outcome(scenario_dir, scenario_name))
    
    return all_issues

def main():
    """Main validation routine."""
    print("="*60)
    print("  SCENARIO VALIDATION TOOL")
    print("="*60)
    
    if not os.path.exists(ASSETS_DIR):
        print(f"\n❌ Assets directory not found: {ASSETS_DIR}")
        print("   Run generate_professional_assets.py first!")
        return
    
    # Find all scenario directories
    scenarios = [d for d in os.listdir(ASSETS_DIR) 
                 if os.path.isdir(os.path.join(ASSETS_DIR, d)) 
                 and d.startswith('SCENARIO_')]
    
    if not scenarios:
        print(f"\n❌ No scenarios found in {ASSETS_DIR}")
        return
    
    print(f"\nFound {len(scenarios)} scenarios to validate\n")
    
    all_scenario_issues = {}
    
    for scenario_name in sorted(scenarios):
        scenario_dir = os.path.join(ASSETS_DIR, scenario_name)
        issues = validate_scenario(scenario_dir, scenario_name)
        all_scenario_issues[scenario_name] = issues
    
    # Print summary
    print("\n" + "="*60)
    print("  VALIDATION SUMMARY")
    print("="*60)
    
    total_issues = 0
    for scenario_name, issues in all_scenario_issues.items():
        if issues:
            print(f"\n{scenario_name}: {len(issues)} issues")
            for issue in issues:
                print(issue)
            total_issues += len(issues)
        else:
            print(f"\n{scenario_name}: ✅ ALL CHECKS PASSED")
    
    print("\n" + "="*60)
    if total_issues == 0:
        print("🎉 ALL SCENARIOS VALIDATED SUCCESSFULLY!")
    else:
        print(f"⚠️  Found {total_issues} issues across all scenarios")
    print("="*60)

if __name__ == "__main__":
    main()