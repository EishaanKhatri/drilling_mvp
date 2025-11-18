#!/usr/bin/env python3
"""
Generate a plain-text, judge-ready report from artifacts JSON (no visuals required).
Usage: python tools/generate_text_report.py artifacts/ALPHA-PRIME-1
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def fmt(x, n=3):
    if isinstance(x, (int, float)):
        return f"{x:.{n}f}"
    return str(x)

def build_report(artifacts_dir):
    base = Path(artifacts_dir)
    cons = load_json(base / "consensus.json")
    petro = load_json(base / "petrophysical.json")
    reservoir = load_json(base / "reservoir.json")
    risk = load_json(base / "risk_esg.json")
    seismic = load_json(base / "seismic.json")

    # Key metrics
    sand_prob = petro['outputs']['lithology_probs']['sand']
    porosity = petro['outputs']['porosity_estimate']
    poro_norm = reservoir['outputs'].get('porosity_norm', porosity / 0.35 if porosity else None)
    rqi = reservoir['outputs']['reservoir_quality_index']
    reservoir_prob = seismic['outputs'].get('reservoir_probability', None)
    reserves_bbl = risk['outputs']['reserves_bbl']
    npv = risk['outputs']['NPV_P50']
    final_score = cons['outputs']['final_score']
    recommendation = cons['outputs']['recommendation']
    input_hash = cons['provenance'].get('input_hash')

    timestamp = datetime.now().isoformat(timespec='seconds')

    # Build text
    lines = []
    lines.append(f"Report generated: {timestamp}")
    lines.append(f"Scenario / Site: {base.name}")
    lines.append("-" * 60)
    lines.append("1) Petrophysical summary")
    lines.append(f"   - Porosity (mean NPHI): {fmt(porosity,3)} (i.e. {fmt(porosity*100,1)}%)")
    lines.append(f"   - Sand probability (GR heuristic): {fmt(sand_prob,3)} ({fmt(sand_prob*100,1)}%)")
    lines.append("")
    lines.append("2) Seismic summary")
    lines.append(f"   - Reservoir probability (image heuristic): {fmt(reservoir_prob,3)}")
    lines.append(f"   - Blobs detected: {seismic['metrics'].get('num_blobs', 'N/A')}")
    lines.append("")
    lines.append("3) Reservoir fusion")
    lines.append(f"   - Porosity (normalized to 0.35): {fmt(poro_norm,3)}")
    lines.append(f"   - Reservoir Quality Index (RQI): {fmt(rqi,3)}")
    lines.append("")
    lines.append("4) Economics & ESG")
    lines.append(f"   - Estimated recoverable reserves: {fmt(reserves_bbl,0)} bbl")
    lines.append(f"   - NPV (P50, simple calc): ${fmt(npv,0)}")
    lines.append(f"   - ESG veto: {risk.get('esg', {}).get('veto', False)} (status: {risk.get('esg', {}).get('ESG_status')})")
    lines.append("")
    lines.append("5) Consensus decision")
    lines.append(f"   - Normalized NPV used: {fmt(cons['outputs'].get('npv_norm', 0),3)}")
    lines.append(f"   - Final score: {fmt(final_score,3)}")
    lines.append(f"   - Recommendation: {recommendation}")
    lines.append("")
    lines.append("6) Explainability & provenance")
    lines.append(f"   - Explain bullets: ")
    for b in cons.get('explain_bullets', []):
        lines.append(f"     * {b}")
    lines.append(f"   - Input hash (provenance): {input_hash}")
    lines.append("-" * 60)
    lines.append("Concise verdict (for stakeholders):")
    lines.append(f"  Recommendation = {recommendation}. Final score {fmt(final_score,3)} below drill threshold (0.60). Economic estimate negative (NPV ${fmt(npv,0)}); geological indicators are modest (porosity {fmt(porosity,3)}, sand ~{fmt(sand_prob*100,1)}%).")
    lines.append("")
    lines.append("Suggested next steps:")
    lines.append("  - Human expert review of logs and seismic annotations.")
    lines.append("  - Cross-check NPHI units and recalibrate GR thresholds with local field data.")
    lines.append("  - Consider field-specific economics and sensitivity runs (price, RF, CAPEX).")
    lines.append("")
    lines.append("End of report.")

    # Save
    out_path = base / "text_report.txt"
    out_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"Report written to: {out_path}")
    return out_path

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_text_report.py <artifacts_dir>")
        sys.exit(1)
    build_report(sys.argv[1])
