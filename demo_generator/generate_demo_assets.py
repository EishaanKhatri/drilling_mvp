import os
import json
import numpy as np
import pandas as pd
import cv2 # OpenCV for image generation

# --- CONFIGURATION ---
# You can tweak these parameters to change the generated data.
ASSETS_DIR = "demo_assets"
DRILL_SCENARIO_DIR = os.path.join(ASSETS_DIR, "CLEAR_DRILL_SCENARIO")
HOLD_SCENARIO_DIR = os.path.join(ASSETS_DIR, "CLEAR_HOLD_SCENARIO_ESG")

# Well Log Parameters
DEPTH_START_M = 2500
DEPTH_END_M = 2700
DEPTH_STEP_M = 0.5
N_ROWS = int((DEPTH_END_M - DEPTH_START_M) / DEPTH_STEP_M)

# Seismic Image Parameters
IMG_SIZE = (256, 256)

# --- HELPER FUNCTIONS ---

def create_well_log_dataframe(depth_start, depth_end, step, gr_mean, nphi_mean, rhob_mean):
    """Generates a Pandas DataFrame for a well log with realistic noise."""
    depth = np.arange(depth_start, depth_end, step)
    
    # Add some noise and a slight trend to make it look more realistic
    noise_gr = np.random.normal(0, 5, len(depth))
    noise_nphi = np.random.normal(0, 0.02, len(depth))
    noise_rhob = np.random.normal(0, 0.05, len(depth))
    
    trend = np.linspace(0, 5, len(depth))

    df = pd.DataFrame({
        'depth': depth,
        'GR': np.clip(gr_mean + noise_gr + trend, 10, 200),
        'NPHI': np.clip(nphi_mean + noise_nphi, 0.01, 0.45),
        'RHOB': np.clip(rhob_mean + noise_rhob, 2.0, 2.9)
    })
    return df

def create_seismic_image(has_bright_spot: bool, size: tuple, filepath: str):
    """Generates a synthetic seismic PNG image."""
    # Create a noisy background that looks like seismic static
    background = np.random.normal(loc=128, scale=40, size=size).astype(np.uint8)
    
    if has_bright_spot:
        # A "bright spot" indicates a potential hydrocarbon trap (gas sand)
        # We will create a bright, distinct blob
        center_x, center_y = int(size[1] * 0.6), int(size[0] * 0.5)
        radius = int(size[0] * 0.1)
        # Draw a bright circle
        cv2.circle(background, (center_x, center_y), radius, (255, 255, 255), -1)
        # Blur it slightly to make it look more natural
        background = cv2.GaussianBlur(background, (15, 15), 0)

    cv2.imwrite(filepath, background)
    print(f"✅ Created seismic image at: {filepath}")

def create_site_metadata(site_id: str, is_protected: bool, filepath: str):
    """Generates the site_meta.json file."""
    meta = {
        "site_id": site_id,
        "operator": "DemoCorp",
        "basin": "North Sea Analogue",
        "protected_area": is_protected,
        "notes": "This is a key metadata flag for the Risk/ESG agent."
    }
    with open(filepath, 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"✅ Created metadata file at: {filepath}")

# --- MAIN GENERATION LOGIC ---

def generate_drill_scenario_assets():
    """
    Generates assets for a scenario that should result in a DRILL recommendation.
    - Low Gamma Ray (GR) -> Indicates sandstone (good reservoir rock)
    - Good Porosity (NPHI)
    - Clear seismic "bright spot"
    - Not in a protected area
    """
    print("\n--- Generating assets for [CLEAR DRILL SCENARIO] ---")
    os.makedirs(DRILL_SCENARIO_DIR, exist_ok=True)
    
    # 1. Create Well Log CSV
    log_path = os.path.join(DRILL_SCENARIO_DIR, "logs.csv")
    df_good = create_well_log_dataframe(
        depth_start=DEPTH_START_M, depth_end=DEPTH_END_M, step=DEPTH_STEP_M,
        gr_mean=45,     # Low GR = good (sandstone)
        nphi_mean=0.22, # High NPHI = good (high porosity)
        rhob_mean=2.25  # Lower RHOB often correlates with porosity
    )
    df_good.to_csv(log_path, index=False)
    print(f"✅ Created 'good' well log at: {log_path}")
    
    # 2. Create Seismic PNG
    seismic_path = os.path.join(DRILL_SCENARIO_DIR, "seismic.png")
    create_seismic_image(has_bright_spot=True, size=IMG_SIZE, filepath=seismic_path)

    # 3. Create Metadata JSON
    meta_path = os.path.join(DRILL_SCENARIO_DIR, "site_meta.json")
    create_site_metadata(site_id="PROSPECT_ALPHA_7", is_protected=False, filepath=meta_path)
    

def generate_hold_scenario_assets():
    """
    Generates assets for a scenario that should result in a HOLD (VETO) recommendation.
    The key is that the geology and seismic are IDENTICAL to the DRILL case,
    but the ESG flag is different. This is a very powerful demo point.
    """
    print("\n--- Generating assets for [CLEAR HOLD SCENARIO (ESG VETO)] ---")
    os.makedirs(HOLD_SCENARIO_DIR, exist_ok=True)
    
    # 1. Create Well Log CSV (using the same 'good' parameters)
    log_path = os.path.join(HOLD_SCENARIO_DIR, "logs.csv")
    df_good = create_well_log_dataframe(
        depth_start=DEPTH_START_M, depth_end=DEPTH_END_M, step=DEPTH_STEP_M,
        gr_mean=45,     # Still good geology
        nphi_mean=0.22, # Still good porosity
        rhob_mean=2.25
    )
    df_good.to_csv(log_path, index=False)
    print(f"✅ Created 'good' well log at: {log_path}")
    
    # 2. Create Seismic PNG (still with a bright spot)
    seismic_path = os.path.join(HOLD_SCENARIO_DIR, "seismic.png")
    create_seismic_image(has_bright_spot=True, size=IMG_SIZE, filepath=seismic_path)

    # 3. Create Metadata JSON (THIS IS THE ONLY DIFFERENCE)
    meta_path = os.path.join(HOLD_SCENARIO_DIR, "site_meta.json")
    create_site_metadata(site_id="ECO_SENSITIVE_BETA_2", is_protected=True, filepath=meta_path)

if __name__ == "__main__":
    print("=========================================")
    print("   Generating All Required Demo Assets   ")
    print("=========================================")
    
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)
        
    generate_drill_scenario_assets()
    generate_hold_scenario_assets()
    
    print("\n-----------------------------------------")
    print("🎉 Asset generation complete!")
    print(f"All files have been saved in the '{ASSETS_DIR}' directory.")
    print("You are now ready for the demo.")
    print("-----------------------------------------")