import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import concurrent.futures
import sys
import os
from scipy.interpolate import griddata

# ==========================================================
# CONFIGURATION
# ==========================================================

# 1. ADDED SEPT 13 & UPDATED LABELS
DAYS = [
    {"label": "Sept 13 (Control)", "file": "processed_9_13.csv"},
    {"label": "Sept 14 (Stable/Wake)", "file": "processed_9_14.csv"},
    {"label": "Sept 20 (Unstable/No Wake)", "file": "processed_9_20.csv"}
]
TURBINE_FILE = "uswtdb_V8_2_20251210.csv"

# 2. EXPANDED BOUNDARIES (LOWERED LAT_MIN)
LAT_MIN, LAT_MAX = 36.00, 36.55  # Lowered from 36.25 to 36.00 to catch bottom loops
LON_MIN, LON_MAX = -97.85, -97.30
GRID_RES = 30  # Slightly higher res (30x30) for better terrain background

# ==========================================================
# 1. FETCH TERRAIN DATA (ONCE)
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

# Interpolate for smooth contours
zi = griddata((df_terrain['Lon'], df_terrain['Lat']), df_terrain['Elev'], 
              (grid_lons, grid_lats), method='cubic')

# ==========================================================
# 2. LOAD TURBINES
# ==========================================================
print("\n--- STEP 2: Loading Turbines ---")
turbines = pd.read_csv(TURBINE_FILE, low_memory=False)

# Filter for Armadillo Flats region
farm_turbines = turbines[
    (turbines['ylat'] >= LAT_MIN) & (turbines['ylat'] <= LAT_MAX) & 
    (turbines['xlong'] >= LON_MIN) & (turbines['xlong'] <= LON_MAX)
].copy()

print(f"  -> Found {len(farm_turbines)} turbines in the map area.")

# ==========================================================
# 3. GENERATE COMBINED PLOTS
# ==========================================================
print("\n--- STEP 3: Generating Combined Maps ---")

for day in DAYS:
    filename = day['file']
    label = day['label']
    
    if not os.path.exists(filename):
        print(f"  [Skipping] {filename} not found.")
        continue

    print(f"  -> Processing {label}...")
    df_flight = pd.read_csv(filename)
    
    # Create Figure
    fig, ax = plt.subplots(figsize=(12, 10)) # Taller aspect ratio to fit the new lat range
    
    # LAYER 1: TERRAIN CONTOURS (Background)
    # Using 'gist_earth' with transparency
    contour = ax.contourf(grid_lons, grid_lats, zi, levels=25, cmap='gist_earth', alpha=0.5)
    
    # Add terrain colorbar (Left side or separate)
    # We will put it on the right, but distinct from wind
    
    # LAYER 2: TURBINES (Middle)
    ax.scatter(farm_turbines['xlong'], farm_turbines['ylat'], 
               marker='^', color='black', s=50, label='Wind Turbines', zorder=2)

    # LAYER 3: FLIGHT PATH WIND SPEED (Top)
    # Using 'turbo' for high contrast
    sc = ax.scatter(df_flight['Longitude'], df_flight['Latitude'], 
                    c=df_flight['Wind_Speed'], cmap='turbo', s=10, alpha=0.9, zorder=3)
    
    # Dual Colorbars
    cbar_wind = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.03)
    cbar_wind.set_label('Wind Speed (m/s)', fontweight='bold')
    
    cbar_terr = plt.colorbar(contour, ax=ax, pad=0.08, fraction=0.03)
    cbar_terr.set_label('Terrain Elevation (m)', fontweight='bold')

    # Formatting
    ax.set_title(f"Combined Analysis: {label}\nTerrain + Turbines + Flight Winds", fontsize=14, fontweight='bold')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc='upper left', framealpha=0.9)
    
    # Set limits explicitly
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    
    # Save
    out_name = f"Combined_Map_{filename.replace('.csv', '.png')}"
    plt.tight_layout()
    plt.savefig(out_name, dpi=200)
    print(f"     Saved: {out_name}")
    plt.close()

print("\nDone! Check the output folder.")