import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================================
# USER SETTINGS
# ==========================================================

# Your aircraft data files
CSV_FILES = [
    "processed_9_13.csv", 
    "processed_9_14.csv", 
    "processed_9_20.csv"
]

# Your new Turbine CSV (Update this filename!)
TURBINE_FILE = "uswtdb_V8_2_20251210.csv" 

# Vertical Bin Size for the profile (meters)
BIN_SIZE_M = 10 

# Buffer distance (degrees) to search for turbines around the flight path
# 0.05 degrees is roughly 5km.
SEARCH_BUFFER = 0.05

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def calculate_vertical_profile(df, bin_size):
    """
    Groups data by Altitude (AGL) to create a clean vertical profile.
    """
    if "AGL" not in df.columns:
        df["AGL"] = df["Altitude"] - df["Terrain_Elev"]

    # Filter out negative AGL
    df = df[df["AGL"] > 0].copy()
    
    max_agl = df["AGL"].max()
    if np.isnan(max_agl): max_agl = 1000 # Fallback
        
    bins = np.arange(0, max_agl + bin_size, bin_size)
    df['agl_bin'] = pd.cut(df['AGL'], bins, labels=bins[:-1])
    
    # Calculate stats
    profile = df.groupby('agl_bin', observed=True)['Wind_Speed'].agg(['mean', 'std']).reset_index()
    profile['z_center'] = profile['agl_bin'].astype(float) + (bin_size / 2)
    
    return profile.dropna()

def get_local_turbines(turbine_df, lat_min, lat_max, lon_min, lon_max):
    """
    Filters the massive turbine list to just those inside the flight box.
    """
    # Create a bounding box with a buffer
    t_subset = turbine_df[
        (turbine_df['ylat'] >= lat_min - SEARCH_BUFFER) & 
        (turbine_df['ylat'] <= lat_max + SEARCH_BUFFER) & 
        (turbine_df['xlong'] >= lon_min - SEARCH_BUFFER) & 
        (turbine_df['xlong'] <= lon_max + SEARCH_BUFFER)
    ].copy()
    
    return t_subset

# ==========================================================
# LOAD TURBINE DATA (ONCE)
# ==========================================================

if os.path.exists(TURBINE_FILE):
    print(f"Loading Turbine Database: {TURBINE_FILE}...")
    # We only read the columns we need to save memory
    cols_to_use = ['case_id', 'ylat', 'xlong', 't_hh', 't_ttlh']
    # t_hh = Hub Height, t_ttlh = Total Tip Height
    
    try:
        all_turbines = pd.read_csv(TURBINE_FILE, usecols=lambda c: c in cols_to_use)
        # Ensure numeric
        all_turbines['t_hh'] = pd.to_numeric(all_turbines['t_hh'], errors='coerce')
        all_turbines['t_ttlh'] = pd.to_numeric(all_turbines['t_ttlh'], errors='coerce')
        print(f"  -> Loaded {len(all_turbines)} turbines.")
    except Exception as e:
        print(f"  [!] Error reading turbine file: {e}")
        all_turbines = pd.DataFrame()
else:
    print(f"  [!] Turbine file not found: {TURBINE_FILE}")
    all_turbines = pd.DataFrame()

# ==========================================================
# MAIN PLOTTING LOOP
# ==========================================================

plt.style.use('seaborn-v0_8-whitegrid')

for filename in CSV_FILES:
    if not os.path.exists(filename):
        continue
        
    print(f"\nProcessing {filename}...")
    df = pd.read_csv(filename)
    
    required = ["Latitude", "Longitude", "Altitude", "Wind_Speed", "Terrain_Elev"]
    if not all(col in df.columns for col in required):
        print("  [!] Missing columns. Skipping.")
        continue

    # Filter bad data
    df = df.dropna(subset=required).copy()
    
    # Calculate AGL
    if "AGL" not in df.columns:
        df["AGL"] = df["Altitude"] - df["Terrain_Elev"]
    
    date_label = filename.replace("processed_", "").replace(".csv", "").replace("_", "/")

    # --- 1. FIND NEARBY TURBINES ---
    if not all_turbines.empty:
        local_turbines = get_local_turbines(
            all_turbines, 
            df['Latitude'].min(), df['Latitude'].max(), 
            df['Longitude'].min(), df['Longitude'].max()
        )
        print(f"  -> Found {len(local_turbines)} turbines in flight area.")
        
        # Calculate stats for plotting lines
        mean_hub_height = local_turbines['t_hh'].mean()
        mean_tip_height = local_turbines['t_ttlh'].mean()
    else:
        local_turbines = pd.DataFrame()
        mean_hub_height = np.nan

    # --- 2. CALCULATE PROFILE ---
    profile = calculate_vertical_profile(df, BIN_SIZE_M)

    # --- 3. CREATE PLOT ---
    fig = plt.figure(figsize=(14, 8))
    
    # PANEL 1: VERTICAL PROFILE
    ax1 = fig.add_subplot(121)
    
    # Plot Wind Profile
    ax1.plot(profile['mean'], profile['z_center'], color='navy', lw=3, label='Mean Wind Speed')
    ax1.fill_betweenx(profile['z_center'], 
                      profile['mean'] - profile['std'], 
                      profile['mean'] + profile['std'], 
                      color='navy', alpha=0.2, label='Variability (1 std)')
    
    # ADD TURBINE HEIGHT REFERENCE
    if not np.isnan(mean_hub_height):
        # Draw Hub Height (The center of the rotor)
        ax1.axhline(mean_hub_height, color='red', linestyle='--', lw=2, label=f'Avg Hub Height ({mean_hub_height:.0f}m)')
        
        # Optional: Shade the Rotor Swept Area (approx +/- 50m from hub? Or use Tip Height?)
        # Let's use Tip Height to show the full danger zone
        if not np.isnan(mean_tip_height):
             ax1.axhspan(mean_hub_height - (mean_tip_height - mean_hub_height), 
                         mean_tip_height, color='red', alpha=0.1, label='Rotor Swept Area')

    ax1.set_xlabel("Wind Speed (m/s)", fontweight='bold')
    ax1.set_ylabel("Height Above Ground (m AGL)", fontweight='bold')
    ax1.set_title("Vertical Wind Profile & Turbine Heights", fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # PANEL 2: MAP VIEW
    ax2 = fig.add_subplot(122)
    
    # Plot Turbines first (Black Triangles)
    if not local_turbines.empty:
        ax2.scatter(local_turbines['xlong'], local_turbines['ylat'], 
                    color='black', marker='^', s=50, label='Wind Turbines')
    
    # Plot Flight Path (Colored by Wind Speed)
    sc = ax2.scatter(df['Longitude'], df['Latitude'], 
                     c=df['Wind_Speed'], cmap='jet', 
                     s=5, alpha=0.7, label='Flight Path')
    
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label('Wind Speed (m/s)')
    
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_title("Flight Path vs Turbine Locations", fontweight='bold')
    ax2.axis('equal')
    # Add legend manually to include the Turbine marker
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    out_file = f"Turbine_Analysis_{date_label.replace('/', '_')}.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"  -> Saved: {out_file}")

print("\nDone.")