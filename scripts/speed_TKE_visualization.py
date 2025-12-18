import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================================
# USER SETTINGS
# ==========================================================

# Added September 13 to the list
CSV_FILES = [
    "processed_9_13.csv",
    "processed_9_14.csv", 
    "processed_9_20.csv"
]

# Bin size for calculating the Median Line (in km)
BIN_SIZE_KM = 0.5 

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def calculate_binned_median(df, x_col, y_col, bin_size):
    """
    Groups data into spatial bins and calculates the median for each bin.
    """
    bins = np.arange(df[x_col].min(), df[x_col].max() + bin_size, bin_size)
    df['bin'] = pd.cut(df[x_col], bins, labels=bins[:-1])
    # Observed=True ensures compatibility with newer Pandas versions
    median_df = df.groupby('bin', observed=True)[y_col].median().reset_index()
    median_df['bin_center'] = median_df['bin'].astype(float) + (bin_size / 2)
    return median_df

# ==========================================================
# PLOTTING LOOP
# ==========================================================

plt.style.use('seaborn-v0_8-whitegrid')

for filename in CSV_FILES:
    if not os.path.exists(filename):
        print(f"Skipping {filename} (File not found)")
        continue
        
    print(f"Generating plots for {filename}...")
    df = pd.read_csv(filename)
    
    date_label = filename.replace("processed_", "").replace(".csv", "").replace("_", "/")
    
    # --- CALCULATE MEDIANS ---
    wind_med = calculate_binned_median(df, 'distance_km', 'Wind_Speed', BIN_SIZE_KM)
    tke_med = calculate_binned_median(df, 'distance_km', 'TKE_resolved', BIN_SIZE_KM)

    # --- CREATE PLOT ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # -----------------------------
    # PLOT 1: WIND SPEED
    # -----------------------------
    ax1.scatter(df['distance_km'], df['Wind_Speed'], 
                color='silver', s=10, alpha=0.5, label='Raw Measurements')
    ax1.plot(wind_med['bin_center'], wind_med['Wind_Speed'], 
             color='navy', linewidth=2.5, label=f'Median ({BIN_SIZE_KM}km bin)')
    
    # *** SMART SCALING (WIND) ***
    # 99th percentile or 1.1x median max
    y_max_wind = df['Wind_Speed'].quantile(0.99) * 1.1
    y_min_wind = max(0, df['Wind_Speed'].quantile(0.01) * 0.9)
    ax1.set_ylim(y_min_wind, y_max_wind)

    ax1.set_ylabel('Wind Speed (m/s)', fontsize=12, fontweight='bold')
    ax1.set_title(f"Wind Profile: {date_label}", fontsize=14)
    ax1.legend(loc='upper right', frameon=True)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # -----------------------------
    # PLOT 2: TKE
    # -----------------------------
    ax2.scatter(df['distance_km'], df['TKE_resolved'], 
                color='silver', s=10, alpha=0.5, label='Raw Measurements')
    ax2.plot(tke_med['bin_center'], tke_med['TKE_resolved'], 
             color='darkred', linewidth=2.5, label=f'Median ({BIN_SIZE_KM}km bin)')
    
    # *** SMART SCALING (TKE) ***
    q98_tke = df['TKE_resolved'].quantile(0.98)
    median_peak = tke_med['TKE_resolved'].max()
    
    # Use larger of 98th percentile or 1.5x median peak
    y_max_tke = max(q98_tke, median_peak * 1.5)
    
    # Safety cap if data is extremely noisy
    if y_max_tke > 20: y_max_tke = 20  
    
    ax2.set_ylim(0, y_max_tke)

    ax2.set_ylabel('TKE ($m^2/s^2$)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Distance from Reference Point (km)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    
    # Save with unique name
    out_file = f"Graph_{date_label.replace('/', '_')}_Zoomed.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    
    print(f"  -> Saved {out_file}")

print("\nDone! All graphs generated.")