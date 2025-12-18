import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, asin, degrees
import os

# ==========================================================
# USER SETTINGS
# ==========================================================

COMPARISON_PAIRS = [
    {
        "aircraft_file": "processed_9_14.csv",
        "lidar_file": "sd.lidar.z01.00.20230914.000000.sta",
        "date_label": "Sept 14"
    },
    {
        "aircraft_file": "processed_9_20.csv",
        "lidar_file": "sd.lidar.z01.00.20230920.000000.sta",
        "date_label": "Sept 20"
    }
]

# LIDAR Location (AWAKEN Chaloupeks)
LIDAR_LAT = 36.3808
LIDAR_LON = -97.6472
MAX_DIST_KM = 3.0  # Keep strictly local

# ==========================================================
# ADVANCED PARSER (Gets Speed & Dispersion)
# ==========================================================

def parse_windcube_extended(filepath):
    """
    Parses .sta files to extract BOTH Wind Speed and Dispersion.
    Returns a dataframe and a dictionary mapping height -> {'speed': col, 'disp': col}
    """
    skip_rows = 0
    # Use latin-1 for degree symbols
    with open(filepath, 'r', encoding='latin-1') as f:
        for i, line in enumerate(f):
            if "Timestamp" in line and "Wind Speed" in line:
                skip_rows = i
                break
    
    df = pd.read_csv(filepath, skiprows=skip_rows, sep='\t', engine='python', encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]
    df['Timestamp'] = pd.to_datetime(df.iloc[:, 0])
    
    # Map heights dynamically
    sensor_map = {}
    for col in df.columns:
        if "Wind Speed (m/s)" in col and "Dispersion" not in col and "min" not in col:
            try:
                h = int(col.split('m')[0])
                if h not in sensor_map: sensor_map[h] = {}
                sensor_map[h]['speed'] = col
            except: pass
        elif "Wind Speed Dispersion" in col:
            try:
                h = int(col.split('m')[0])
                if h not in sensor_map: sensor_map[h] = {}
                sensor_map[h]['disp'] = col
            except: pass
            
    return df, sensor_map

def interpolate_lidar(lidar_row, sensor_map, target_agl):
    """
    Interpolates LIDAR wind speed and dispersion to the aircraft's exact AGL.
    """
    heights = sorted(sensor_map.keys())
    
    # Find bounding heights (e.g., 120m and 140m for a 130m aircraft)
    h_lower, h_upper = None, None
    for h in heights:
        if h <= target_agl: h_lower = h
        if h >= target_agl and h_upper is None: h_upper = h
    
    # Handle edge cases (plane too low or too high)
    if h_lower is None: h_lower = h_upper
    if h_upper is None: h_upper = h_lower
    
    # Extract values
    val_l = lidar_row[sensor_map[h_lower]['speed']]
    val_u = lidar_row[sensor_map[h_upper]['speed']]
    disp_l = lidar_row.get(sensor_map[h_lower].get('disp'), 0) # Default to 0 if missing
    disp_u = lidar_row.get(sensor_map[h_upper].get('disp'), 0)
    
    # Linear Interpolation
    if h_upper == h_lower:
        return val_l, disp_l
    
    fraction = (target_agl - h_lower) / (h_upper - h_lower)
    interp_speed = val_l + (val_u - val_l) * fraction
    interp_disp = disp_l + (disp_u - disp_l) * fraction
    
    return interp_speed, interp_disp

# ==========================================================
# MAIN LOOP
# ==========================================================

for pair in COMPARISON_PAIRS:
    ac_file = pair["aircraft_file"]
    lidar_file = pair["lidar_file"]
    label = pair["date_label"]
    
    print(f"\nProcessing {label}...")
    
    if not os.path.exists(ac_file) or not os.path.exists(lidar_file):
        print("  [!] Missing files. Skipping.")
        continue

    # 1. Load Aircraft
    ac_df = pd.read_csv(ac_file)
    ac_df['Date_Time'] = pd.to_datetime(ac_df['Date_Time'])
    
    # 2. Filter Proximity
    # (Simple Euclidean approx for speed on small scales is fine here, or reuse Haversine)
    dlat = np.radians(ac_df['Latitude'] - LIDAR_LAT)
    dlon = np.radians(ac_df['Longitude'] - LIDAR_LON)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(LIDAR_LAT))**2 * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    ac_df['dist_km'] = 6371 * c
    
    nearby = ac_df[ac_df['dist_km'] < MAX_DIST_KM].copy()
    
    if nearby.empty:
        print("  [!] No close flyover found.")
        continue
        
    fly_time = nearby['Date_Time'].mean()
    
    # 3. Calculate Average Aircraft AGL
    # We use this to pick the right LIDAR values
    if "Terrain_Elev" in nearby.columns:
        avg_terrain = nearby['Terrain_Elev'].mean()
    else:
        avg_terrain = 300 # Fallback
    
    nearby['AGL'] = nearby['Altitude'] - avg_terrain
    target_agl = nearby['AGL'].mean()
    print(f"  -> Aircraft Avg AGL: {target_agl:.1f} m")

    # 4. Load LIDAR & Interpolate
    try:
        lidar_df, sensor_map = parse_windcube_extended(lidar_file)
        
        # Get closest timestamp
        lidar_df['diff'] = abs(lidar_df['Timestamp'] - fly_time)
        closest = lidar_df.loc[lidar_df['diff'].idxmin()]
        
        # Interpolate to aircraft height
        ref_speed, ref_disp = interpolate_lidar(closest, sensor_map, target_agl)
        print(f"  -> LIDAR Reference at {target_agl:.1f}m: {ref_speed:.2f} m/s (Disp: {ref_disp:.2f})")

        # 5. PLOTTING
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # --- TOP PANEL: TIME SERIES ---
        # Plot Aircraft
        ax1.plot(nearby['Date_Time'], nearby['Wind_Speed'], color='red', marker='o', markersize=3, 
                 linestyle='-', linewidth=0.5, alpha=0.7, label='Aircraft (Instantaneous)')
        
        # Plot LIDAR Reference Line
        ax1.axhline(ref_speed, color='black', linewidth=2, label=f'LIDAR Mean (@{target_agl:.0f}m)')
        
        # Plot LIDAR Dispersion (The "Turbulence Band")
        ax1.axhspan(ref_speed - ref_disp, ref_speed + ref_disp, color='black', alpha=0.1, label='LIDAR Dispersion range')
        
        ax1.set_ylabel("Wind Speed (m/s)", fontweight='bold')
        ax1.set_title(f"LIDAR Validation: Time Series ({label})", fontweight='bold', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Format X-axis to show H:M:S
        import matplotlib.dates as mdates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')

        # --- BOTTOM PANEL: HISTOGRAM ---
        # Show distribution of Aircraft speeds vs LIDAR reference
        bins = np.linspace(nearby['Wind_Speed'].min()-1, nearby['Wind_Speed'].max()+1, 30)
        ax2.hist(nearby['Wind_Speed'], bins=bins, color='red', alpha=0.6, label='Aircraft Distribution')
        
        ax2.axvline(ref_speed, color='black', linewidth=3, linestyle='--', label='LIDAR Reference')
        
        ax2.set_xlabel("Wind Speed (m/s)", fontweight='bold')
        ax2.set_ylabel("Count", fontweight='bold')
        ax2.set_title("Bias Check: Aircraft Distribution vs LIDAR Mean", fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"Validation_TimeSeries_{label.replace(' ', '')}.png", dpi=200)
        print(f"  -> Saved plot.")
        
    except Exception as e:
        print(f"  [!] Error: {e}")