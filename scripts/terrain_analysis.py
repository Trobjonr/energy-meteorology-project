import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata

# ==========================================================
# CONFIGURATION (Armadillo Flats Coordinates)
# ==========================================================

# Approximate bounding box for the wind farm and wake zone
# derived from your previous flight tracks
LAT_MIN, LAT_MAX = 36.25, 36.50
LON_MIN, LON_MAX = -97.85, -97.35

GRID_RESOLUTION = 30  # 30x30 grid = 900 points (to keep API request fast)

# ==========================================================
# 1. FETCH ELEVATION DATA (USGS API)
# ==========================================================

print(f"Fetching terrain data for area: {LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}W")
print("Querying USGS National Map (this may take 10-20 seconds)...")

lats = np.linspace(LAT_MIN, LAT_MAX, GRID_RESOLUTION)
lons = np.linspace(LON_MIN, LON_MAX, GRID_RESOLUTION)
grid_lats, grid_lons = np.meshgrid(lats, lons)

# Flatten for API querying
query_points = list(zip(grid_lats.flatten(), grid_lons.flatten()))

elevations = []
url = "https://epqs.nationalmap.gov/v1/json"

# Process in chunks to be polite to the API
chunk_size = 50
for i in range(0, len(query_points), chunk_size):
    chunk = query_points[i:i+chunk_size]
    
    # Construct query string (x=Lon, y=Lat)
    # USGS EPQS takes one point at a time or simple params. 
    # For speed in python without complex deps, we loop simply or use a specialized bulk call if available.
    # To ensure reliability without complex libraries, we'll use a simple loop with session.
    
    with requests.Session() as s:
        for lat, lon in chunk:
            try:
                params = {'x': lon, 'y': lat, 'units': 'Meters', 'output': 'json'}
                resp = s.get(url, params=params, timeout=5)
                data = resp.json()
                # Extract elevation
                elev = float(data['value'])
                elevations.append(elev)
            except:
                elevations.append(np.nan)

# Create DataFrame
df_terrain = pd.DataFrame({
    'Latitude': grid_lats.flatten(),
    'Longitude': grid_lons.flatten(),
    'Elevation_m': elevations
})

# Drop failures
df_terrain.dropna(inplace=True)

if df_terrain.empty:
    raise RuntimeError("Failed to fetch data from USGS. Check internet connection.")

# ==========================================================
# 2. ANALYZE HOMOGENEITY
# ==========================================================

min_elev = df_terrain['Elevation_m'].min()
max_elev = df_terrain['Elevation_m'].max()
mean_elev = df_terrain['Elevation_m'].mean()
std_dev = df_terrain['Elevation_m'].std()
terrain_range = max_elev - min_elev

print("\n--- TERRAIN STATISTICS ---")
print(f"Min Elevation: {min_elev:.1f} m")
print(f"Max Elevation: {max_elev:.1f} m")
print(f"Mean Elevation: {mean_elev:.1f} m")
print(f"Total Variation (Range): {terrain_range:.1f} m")
print(f"Roughness (Std Dev): {std_dev:.2f} m")

if std_dev < 15:
    print("CONCLUSION: The terrain is VERY HOMOGENEOUS (Flat).")
elif std_dev < 30:
    print("CONCLUSION: The terrain is MODERATELY HOMOGENEOUS.")
else:
    print("CONCLUSION: The terrain is COMPLEX/HETEROGENEOUS.")

# ==========================================================
# 3. VISUALIZE
# ==========================================================

plt.figure(figsize=(12, 6))

# Interpolate for a smooth contour map
zi = griddata((df_terrain['Longitude'], df_terrain['Latitude']), 
              df_terrain['Elevation_m'], 
              (grid_lons, grid_lats), method='cubic')

# Contour Plot
contour = plt.contourf(grid_lons, grid_lats, zi, levels=20, cmap='terrain')
cbar = plt.colorbar(contour, label='Elevation (m)')

# Add markers for context
plt.title(f"Terrain Homogeneity: Armadillo Flats Region\n(Std Dev: {std_dev:.1f}m)", fontweight='bold')
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Draw the flight corridor approx
plt.plot([-97.8, -97.4], [36.35, 36.35], 'r--', linewidth=2, label='Approx Flight Path')
plt.legend()

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Terrain_Homogeneity_Check.png", dpi=150)
print("\nSaved Map: Terrain_Homogeneity_Check.png")
plt.show()