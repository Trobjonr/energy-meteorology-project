import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import concurrent.futures
import sys
import os
from scipy.interpolate import griddata

# ==========================================================
# CONFIGURATION
# ==========================================================

DAYS = [
    {"label": "Sept 13 (Control)", "file": "processed_9_13.csv"},
    {"label": "Sept 14 (Stable/Wake)", "file": "processed_9_14.csv"},
    {"label": "Sept 20 (Unstable/No Wake)", "file": "processed_9_20.csv"}
]
TURBINE_FILE = "uswtdb_V8_2_20251210.csv"

# Map Boundaries
LAT_MIN, LAT_MAX = 36.00, 36.55
LON_MIN, LON_MAX = -97.85, -97.30
GRID_RES = 30 

# Zonal Buffers
BUFFER_LAT = 0.02
# We remove BUFFER_LON for the wake zone check to avoid the gap
# BUFFER_LON = 0.02 

# ==========================================================
# 1. FETCH TERRAIN DATA
# ==========================================================
print("--- STEP 1: Fetching Terrain Data (USGS) ---")

lats = np.linspace(LAT_MIN, LAT_MAX, GRID_RES)
lons = np.linspace(LON_MIN, LON_MAX, GRID_RES)
grid_lats, grid_lons = np.meshgrid(lats, lons)
query_points = list(zip(grid_lats.flatten(), grid_lons.flatten()))

def get_elevation(point):
    lat, lon = point
    url = "https://epqs.nationalmap.gov/v1/json"
    params = {'x': lon, 'y': lat, 'units': 'Meters', 'output': 'json'}
    try:
        resp = requests.get(url, params=params, timeout=5)
        return float(resp.json()['value'])
    except:
        return np.nan

elevations = []
completed = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    results = executor.map(get_elevation, query_points)
    for res in results:
        elevations.append(res)
        completed += 1
        if completed % 50 == 0:
            sys.stdout.write(f"\r  -> Progress: {completed}/{len(query_points)} points...")
            sys.stdout.flush()

print("\n  -> Terrain data fetched.")

df_terrain = pd.DataFrame({
    'Lat': grid_lats.flatten(),
    'Lon': grid_lons.flatten(),
    'Elev': elevations
}).dropna()

zi = griddata((df_terrain['Lon'], df_terrain['Lat']), df_terrain['Elev'], 
              (grid_lons, grid_lats), method='cubic')

# ==========================================================
# 2. LOAD TURBINES & DEFINE ZONES
# ==========================================================
print("\n--- STEP 2: Loading Turbines & Defining Zones ---")
turbines = pd.read_csv(TURBINE_FILE, low_memory=False)

farm_turbines = turbines[
    (turbines['ylat'] >= LAT_MIN) & (turbines['ylat'] <= LAT_MAX) & 
    (turbines['xlong'] >= LON_MIN) & (turbines['xlong'] <= LON_MAX)
].copy()

# Define Farm Boundaries
east_edge = farm_turbines['xlong'].max()
west_edge = farm_turbines['xlong'].min()
north_edge = farm_turbines['ylat'].max()
south_edge = farm_turbines['ylat'].min()

print(f"  -> Farm Boundaries defined. Turbines found: {len(farm_turbines)}")

# ==========================================================
# 3. GENERATE ZONAL MAPS
# ==========================================================
print("\n--- STEP 3: Generating Zonal Terrain Maps ---")

for day in DAYS:
    filename = day['file']
    label = day['label']
    
    if not os.path.exists(filename):
        print(f"  [Skipping] {filename} not found.")
        continue

    print(f"  -> Processing {label}...")
    df = pd.read_csv(filename)
    
    # --- CLASSIFY ZONES (ADJUSTED LOGIC) ---
    conditions = [
        # 1. Upstream: Strictly East of the easternmost turbine (+ small buffer)
        (df['Longitude'] > east_edge + 0.01), 
        
        # 2. Wake Zone: 
        # CHANGED: Instead of "west_edge - buffer", we use "west_edge + 0.005"
        # This effectively starts the wake zone slightly INSIDE the farm, ensuring
        # the first leg immediately behind the turbines is caught.
        (df['Longitude'] < west_edge + 0.005) & 
        (df['Latitude'] >= south_edge - BUFFER_LAT) & 
        (df['Latitude'] <= north_edge + BUFFER_LAT)
    ]
    choices = ['Upstream', 'Wake_Zone']
    df['Zone'] = np.select(conditions, choices, default='Lateral')

    # Split for easy plotting
    upstream = df[df['Zone'] == 'Upstream']
    wake = df[df['Zone'] == 'Wake_Zone']
    lateral = df[df['Zone'] == 'Lateral']

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(12, 10))

    # 1. TERRAIN (Background)
    contour = ax.contourf(grid_lons, grid_lats, zi, levels=25, cmap='gist_earth', alpha=0.5)
    cbar = plt.colorbar(contour, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label('Terrain Elevation (m)', fontweight='bold')

    # 2. TURBINES
    ax.scatter(farm_turbines['xlong'], farm_turbines['ylat'], 
               marker='^', color='black', s=60, label='Turbines', zorder=5)

    # 3. ZONES (Colored Points)
    # Lateral (Dark Grey)
    ax.scatter(lateral['Longitude'], lateral['Latitude'], 
               c='dimgray', s=15, alpha=0.7, label='Outside/Lateral', zorder=2)
    
    # Upstream (Blue)
    ax.scatter(upstream['Longitude'], upstream['Latitude'], 
               c='blue', s=20, label='Upstream (Control)', zorder=3)
    
    # Wake Zone (Red)
    ax.scatter(wake['Longitude'], wake['Latitude'], 
               c='red', s=20, label='Wake Zone (Downstream)', zorder=4)

    # Formatting
    ax.set_title(f"Zonal Classification over Terrain: {label}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    # Custom Legend
    ax.legend(loc='lower left', framealpha=0.9, fontsize=10)
    
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)

    # Save
    out_name = f"Zonal_Terrain_Map_{filename.replace('.csv', '.png')}"
    plt.tight_layout()
    plt.savefig(out_name, dpi=200)
    print(f"     Saved: {out_name}")
    plt.close()

print("\nDone! All maps generated.")