import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================================
# USER SETTINGS
# ==========================================================

CSV_FILES = [
    "processed_9_13.csv", 
    "processed_9_14.csv", 
    "processed_9_20.csv"
]

# Vertical Bin Size (in meters)
# We group data into layers of this thickness
BIN_SIZE_M = 10 

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def calculate_vertical_profile(df, bin_size):
    """
    Groups data by Altitude (AGL) to create a clean vertical profile.
    Returns: Binned DataFrame with Mean and Std Dev.
    """
    # 1. Ensure we have AGL
    if "AGL" not in df.columns:
        df["AGL"] = df["Altitude"] - df["Terrain_Elev"]

    # 2. Define Bins (from 0 to max AGL)
    max_agl = df["AGL"].max()
    bins = np.arange(0, max_agl + bin_size, bin_size)
    
    # 3. Cut data into bins
    df['agl_bin'] = pd.cut(df['AGL'], bins, labels=bins[:-1])
    
    # 4. GroupBy to get stats
    # We calculate Mean (the line) and Std (the shaded variance area)
    profile = df.groupby('agl_bin', observed=True)['Wind_Speed'].agg(['mean', 'std', 'count']).reset_index()
    
    # Convert bin labels back to numeric for plotting (center of bin)
    profile['z_center'] = profile['agl_bin'].astype(float) + (bin_size / 2)
    
    # Filter out empty bins
    return profile.dropna()

# ==========================================================
# MAIN PLOTTING LOOP
# ==========================================================

plt.style.use('seaborn-v0_8-whitegrid')

for filename in CSV_FILES:
    if not os.path.exists(filename):
        continue
        
    print(f"Generating profile for {filename}...")
    df = pd.read_csv(filename)
    
    # Check for columns
    required = ["Latitude", "Longitude", "Altitude", "Wind_Speed", "Terrain_Elev"]
    if not all(col in df.columns for col in required):
        print(f"  [!] Missing columns. Skipping.")
        continue

    # Filter bad data
    df = df.dropna(subset=required).copy()
    
    # Calculate AGL if missing
    if "AGL" not in df.columns:
        df["AGL"] = df["Altitude"] - df["Terrain_Elev"]
    
    # Filter: Remove negative AGL (glitches where terrain > altitude)
    df = df[df["AGL"] > 0]
    
    date_label = filename.replace("processed_", "").replace(".csv", "").replace("_", "/")

    # --- CALCULATE STATISTICS ---
    profile = calculate_vertical_profile(df, BIN_SIZE_M)

    # --- CREATE PLOT (2 PANELS) ---
    fig = plt.figure(figsize=(14, 8))
    
    # PANEL 1: VERTICAL PROFILE (The "Clean" View)
    ax1 = fig.add_subplot(121)
    
    # Plot the Mean Line
    ax1.plot(profile['mean'], profile['z_center'], color='navy', lw=3, label='Mean Wind Speed')
    
    # Shade the variability (Mean +/- 1 Standard Deviation)
    # This represents the "chaos" as a structured error bar
    ax1.fill_betweenx(profile['z_center'], 
                      profile['mean'] - profile['std'], 
                      profile['mean'] + profile['std'], 
                      color='navy', alpha=0.2, label='Variability (1 std dev)')
    
    ax1.set_xlabel("Wind Speed (m/s)", fontweight='bold', fontsize=12)
    ax1.set_ylabel("Height Above Ground (m AGL)", fontweight='bold', fontsize=12)
    ax1.set_title(f"Vertical Wind Profile\n(Aggregated by Height)", fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, which='both', alpha=0.3)
    
    # PANEL 2: TOP-DOWN MAP (The "Context" View)
    ax2 = fig.add_subplot(122)
    
    # Scatter plot: Lat/Lon colored by Wind Speed
    sc = ax2.scatter(df['Longitude'], df['Latitude'], 
                     c=df['Wind_Speed'], cmap='jet', 
                     s=5, alpha=0.7)
    
    # Add a colorbar
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label('Wind Speed (m/s)')
    
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_title(f"Flight Path Map\n(Colored by Wind Speed)", fontweight='bold')
    ax2.axis('equal') # Keep map proportions correct
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    out_file = f"Vertical_Profile_{date_label.replace('/', '_')}.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"  -> Saved: {out_file}")

print("\nDone.")