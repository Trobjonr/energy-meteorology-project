import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

# ==========================================================
# CONFIGURATION
# ==========================================================

# 1. Target File: Sept 14
FILES = ["processed_9_14.csv"]
TURBINE_FILE = "uswtdb_V8_2_20251210.csv"
TARGET_DISTANCES_KM = [0.5, 2.0, 4.5, 9.0]
SLICE_WIDTH_KM = 0.5  # +/- 0.5 km around target

# ==========================================================
# 1. SETUP & DATA LOADING
# ==========================================================

print("Loading Armadillo Flats data (Sept 14)...")
df_list = []
for f in FILES:
    if os.path.exists(f):
        tmp = pd.read_csv(f)
        df_list.append(tmp)

if not df_list:
    raise FileNotFoundError(f"File {FILES[0]} not found.")
df = pd.concat(df_list, ignore_index=True)

df['Date_Time'] = pd.to_datetime(df['Date_Time'])
df['Time'] = df['Date_Time'].dt.time
df['Hour'] = df['Date_Time'].dt.hour

# ==========================================================
# 2. TIME FILTER (STABLE PHASE < 09:00 CDT)
# ==========================================================

min_hr = df['Hour'].min()
if min_hr >= 12:
    print("  -> Detected UTC Timezone.")
    # 09:00 CDT = 14:00 UTC
    CUTOFF_HOUR, CUTOFF_MIN = 14, 0 
else:
    print("  -> Detected Local Time (CDT).")
    # 09:00 CDT
    CUTOFF_HOUR, CUTOFF_MIN = 9, 0

cutoff_time = datetime.time(CUTOFF_HOUR, CUTOFF_MIN, 0)
print(f"Applying Time Filter: Keeping data BEFORE {cutoff_time} (Stable Phase)...")

# Filter for BEFORE (<) the cutoff time
df = df[df['Time'] < cutoff_time].copy()

if df.empty:
    raise ValueError("Filter removed all data! Check timestamps.")

# ==========================================================
# 3. DEFINE GEOMETRY
# ==========================================================

turbines = pd.read_csv(TURBINE_FILE, low_memory=False)
lat_min, lat_max = df['Latitude'].min(), df['Latitude'].max()
lon_min, lon_max = df['Longitude'].min(), df['Longitude'].max()

farm_turbines = turbines[
    (turbines['ylat'] >= lat_min - 0.05) & (turbines['ylat'] <= lat_max + 0.05) & 
    (turbines['xlong'] >= lon_min - 0.05) & (turbines['xlong'] <= lon_max + 0.05)
].copy()

western_edge = farm_turbines['xlong'].min()
farm_center_lat = farm_turbines['ylat'].mean()
farm_north = farm_turbines['ylat'].max()
farm_south = farm_turbines['ylat'].min()

def lon_to_km(lon1, lon2, lat):
    return (lon1 - lon2) * 111.32 * np.cos(np.radians(lat))

df['Dist_Downwind_km'] = lon_to_km(western_edge, df['Longitude'], farm_center_lat)

# Define Boundaries
buffer_wake = 0.02
buffer_lat  = 0.02

# ==========================================================
# 4. LOCALIZED DEFICIT CALCULATION (Per Distance)
# ==========================================================

results = []
print("\n--- CALCULATING LOCALIZED DEFICITS ---")

for dist in TARGET_DISTANCES_KM:
    min_d, max_d = dist - SLICE_WIDTH_KM, dist + SLICE_WIDTH_KM
    
    # 1. Get ALL data for this distance slice (Wake + Lateral)
    slice_df = df[(df['Dist_Downwind_km'] >= min_d) & 
                  (df['Dist_Downwind_km'] <= max_d)]
    
    if len(slice_df) < 10:
        print(f"  Dist {dist} km | Skipped (Insufficient Data)")
        continue

    # 2. Split into Wake Zone vs. Lateral Zone
    wake_data = slice_df[(slice_df['Latitude'] >= farm_south - buffer_wake) & 
                         (slice_df['Latitude'] <= farm_north + buffer_wake)]
    
    lateral_data = slice_df[(slice_df['Latitude'] > farm_north + buffer_lat) | 
                            (slice_df['Latitude'] < farm_south - buffer_lat)]

    # 3. Calculate Local Reference & Deficit
    if len(lateral_data) > 10 and len(wake_data) > 10:
        # Use the Lateral Mean for THIS SPECIFIC DISTANCE
        U_ref_local = lateral_data['Wind_Speed'].mean()
        U_wake_local = wake_data['Wind_Speed'].mean()
        
        deficit = (U_ref_local - U_wake_local) / U_ref_local * 100
        
        results.append({
            "Dist": dist, 
            "Deficit": deficit,
            "U_Ref": U_ref_local,
            "U_Wake": U_wake_local
        })
        print(f"  Dist {dist} km | Ref: {U_ref_local:.2f} m/s | Wake: {U_wake_local:.2f} m/s | Deficit: {deficit:.1f}%")
        
    else:
        # Fallback if plane didn't fly wide enough at this distance
        print(f"  Dist {dist} km | Missing Lateral Data for Reference")

# ==========================================================
# 5. PLOT
# ==========================================================

if results:
    res_df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['Dist'], res_df['Deficit'], marker='o', lw=2, color='navy', label='Localized Deficit')
    
    plt.axhline(5, color='green', ls='--', label='Recovery Threshold (5%)')
    plt.axhline(0, color='black', lw=1)
    
    plt.title(f"Wake Recovery: Sept 14 (< 09:00 CDT)\nLocalized Reference Method", fontweight='bold')
    plt.xlabel("Distance Downwind (km)", fontweight='bold')
    plt.ylabel("Velocity Deficit (%)", fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks([0, 2, 4.5, 9, 12])
    
    # Auto-scale Y with some padding
    y_max = max(res_df['Deficit'].max() + 5, 10)
    y_min = min(res_df['Deficit'].min() - 5, -5)
    plt.ylim(y_min, y_max)
    
    out_name = "Armadillo_Recovery_Sept14_Stable_Localized.png"
    plt.savefig(out_name, dpi=200)
    print(f"\nSaved: {out_name}")
    
    # Check Conclusion
    final_def = res_df.iloc[-1]['Deficit']
    print(f"\nFinal Deficit at {res_df.iloc[-1]['Dist']}km: {final_def:.1f}%")