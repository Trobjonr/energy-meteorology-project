import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
import os

# ==========================================================
# USER SETTINGS
# ==========================================================

CSV_FILES = [
    "processed_9_13.csv", 
    "processed_9_14.csv", 
    "processed_9_20.csv"
]

# Bin size for trend lines (km)
BIN_SIZE_KM = 0.5 

# Skip factor for Elevation API
# 100 means we only query 1 point every 100 rows.
# For 900k points, this reduces requests from ~9000 to ~90.
DECIMATION_FACTOR = 100 

# ==========================================================
# HELPER: OPTIMIZED ELEVATION FETCH
# ==========================================================

def get_elevation_interpolated(df):
    """
    1. Subsamples the dataframe (takes every Nth row).
    2. Queries the API for that small subset.
    3. Interpolates the results back to the full dataset size.
    """
    base_url = "https://api.open-elevation.com/api/v1/lookup"
    
    # 1. Create a subset (decimation)
    # We copy to avoid SettingWithCopy warnings
    subset = df.iloc[::DECIMATION_FACTOR].copy()
    
    subset_lats = subset["Latitude"].tolist()
    subset_lons = subset["Longitude"].tolist()
    subset_elevs = []
    
    total_subset = len(subset_lats)
    chunk_size = 100
    
    print(f"  -> Optimized: Querying {total_subset} points (instead of {len(df)})...")
    
    # 2. Query API in chunks
    for i in range(0, total_subset, chunk_size):
        lat_chunk = subset_lats[i:i+chunk_size]
        lon_chunk = subset_lons[i:i+chunk_size]
        
        locations = [{"latitude": lat, "longitude": lon} for lat, lon in zip(lat_chunk, lon_chunk)]
        payload = {"locations": locations}
        
        try:
            response = requests.post(base_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=10)
            data = response.json()
            chunk_results = [r['elevation'] for r in data['results']]
            subset_elevs.extend(chunk_results)
            print(f"     Batch {i//chunk_size + 1}: Retrieved {len(chunk_results)} pts")
            time.sleep(0.1) # Be polite to the server
            
        except Exception as e:
            print(f"     [!] Error in batch: {e}")
            # Fill with NaN so we don't crash, but interpolation will look weird here
            subset_elevs.extend([np.nan] * len(lat_chunk))

    # 3. Interpolate back to full size
    # We use the DataFrame index to interpolate linearly
    print("  -> Interpolating terrain for full flight path...")
    
    # Create the 'x' values for interpolation (the indices of the subset)
    x_subset = subset.index
    y_subset = subset_elevs
    
    # Create the 'x' values for the target (the full dataframe indices)
    x_full = df.index
    
    # Perform interpolation
    full_elevs = np.interp(x_full, x_subset, y_subset)
    
    return full_elevs

def calculate_binned_median(df, x_col, y_col, bin_size):
    bins = np.arange(df[x_col].min(), df[x_col].max() + bin_size, bin_size)
    df['bin'] = pd.cut(df[x_col], bins, labels=bins[:-1])
    median_df = df.groupby('bin', observed=True)[y_col].median().reset_index()
    median_df['bin_center'] = median_df['bin'].astype(float) + (bin_size / 2)
    return median_df

# ==========================================================
# MAIN LOOP
# ==========================================================

plt.style.use('seaborn-v0_8-whitegrid')

for filename in CSV_FILES:
    if not os.path.exists(filename):
        print(f"Skipping {filename} (Not found)")
        continue
        
    print(f"\nProcessing {filename}...")
    df = pd.read_csv(filename)
    
    # --- GET ELEVATION (OPTIMIZED) ---
    if "Terrain_Elev" not in df.columns:
        try:
            elevs = get_elevation_interpolated(df)
            df["Terrain_Elev"] = elevs
            df["AGL"] = df["Altitude"] - df["Terrain_Elev"]
            
            # Save strictly to avoid re-running
            df.to_csv(filename, index=False)
            print("  -> Elevation data saved to CSV.")
        except Exception as e:
            print(f"  -> Failed to get elevation: {e}")
            continue
    else:
        print("  -> Elevation data found in CSV (skipping API).")

    # Drop NaNs
    plot_df = df.dropna(subset=["Terrain_Elev", "Wind_Speed"])

    # --- PLOTTING ---
    wind_med = calculate_binned_median(plot_df, 'distance_km', 'Wind_Speed', BIN_SIZE_KM)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    date_label = filename.replace("processed_", "").replace(".csv", "").replace("_", "/")

    # Panel 1: Wind
    ax1.scatter(plot_df['distance_km'], plot_df['Wind_Speed'], color='silver', s=10, alpha=0.5, label='Raw Wind')
    ax1.plot(wind_med['bin_center'], wind_med['Wind_Speed'], color='navy', lw=2.5, label='Median Wind')
    ax1.set_ylabel('Wind Speed (m/s)', fontweight='bold')
    ax1.set_title(f"Wind Speed vs. Terrain Profile ({date_label})", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Terrain
    ax2.fill_between(plot_df['distance_km'], plot_df['Terrain_Elev'], 0, color='tan', alpha=0.6, label='Terrain')
    ax2.plot(plot_df['distance_km'], plot_df['Terrain_Elev'], color='saddlebrown', lw=1)
    ax2.plot(plot_df['distance_km'], plot_df['Altitude'], color='black', lw=1.5, linestyle='--', label='Aircraft Alt (MSL)')
    
    ax2.set_ylabel('Elevation (m MSL)', fontweight='bold')
    ax2.set_xlabel('Distance (km)', fontweight='bold')
    ax2.set_ylim(0, plot_df['Altitude'].max() + 200)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_img = f"Wind_vs_Elevation_{date_label.replace('/', '_')}.png"
    plt.savefig(out_img, dpi=200)
    plt.close()
    print(f"  -> Plot saved: {out_img}")

print("\nDone.")