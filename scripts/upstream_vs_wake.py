import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================================
# USER SETTINGS
# ==========================================================

AIRCRAFT_FILES = [
    "processed_9_13.csv",
    "processed_9_14.csv",
    "processed_9_20.csv"
]
TURBINE_FILE = "uswtdb_V8_2_20251210.csv"
LAT_BUFFER = 0.02 

# ==========================================================
# PROCESSING
# ==========================================================

print("Loading and merging data from all days...")

# 1. Load Aircraft Data
df_list = []
for f in AIRCRAFT_FILES:
    if os.path.exists(f):
        df_list.append(pd.read_csv(f))
    else:
        print(f"  [!] Warning: File {f} not found.")

if not df_list:
    raise FileNotFoundError("No aircraft files found!")

df = pd.concat(df_list, ignore_index=True)
print(f"  -> Combined dataset size: {len(df)} points")

# 2. Load Turbines (Fixing DtypeWarning)
if not os.path.exists(TURBINE_FILE):
    raise FileNotFoundError(f"Turbine file not found: {TURBINE_FILE}")

# Added low_memory=False to silence the warning
turbines = pd.read_csv(TURBINE_FILE, low_memory=False)

# 3. Filter Turbines
lat_min, lat_max = df['Latitude'].min(), df['Latitude'].max()
lon_min, lon_max = df['Longitude'].min(), df['Longitude'].max()

local_turbines = turbines[
    (turbines['ylat'] >= lat_min - 0.05) & 
    (turbines['ylat'] <= lat_max + 0.05) & 
    (turbines['xlong'] >= lon_min - 0.05) & 
    (turbines['xlong'] <= lon_max + 0.05)
].copy()

if local_turbines.empty:
    print("WARNING: No turbines found in this bounding box.")
    exit()

# 4. Define Farm Boundaries
farm_north = local_turbines['ylat'].max()
farm_south = local_turbines['ylat'].min()
farm_east = local_turbines['xlong'].max()
farm_west = local_turbines['xlong'].min()

# 5. Segment Data
def classify_zone(row):
    if (farm_south - LAT_BUFFER) <= row['Latitude'] <= (farm_north + LAT_BUFFER):
        if row['Longitude'] > farm_east:
            return "Upstream (Control)"
        elif row['Longitude'] < farm_west:
            return "Wake Zone (Downstream)"
        else:
            return "Inside Farm"
    else:
        return "Outside/Lateral"

df['Zone'] = df.apply(classify_zone, axis=1)

# Split data for plotting priority
df_bg = df[df['Zone'].isin(["Outside/Lateral", "Inside Farm"])].copy()
df_fg = df[df['Zone'].isin(["Upstream (Control)", "Wake Zone (Downstream)"])].copy()

print(f"  -> Background points: {len(df_bg)}")
print(f"  -> Foreground points: {len(df_fg)} (These will be Red/Blue)")

# ==========================================================
# VISUALIZATION
# ==========================================================

fig = plt.figure(figsize=(14, 8))

# PANEL 1: Map View
ax1 = fig.add_subplot(121)

# A. Plot Background (Grey) FIRST
# We use standard matplotlib scatter for speed and control
ax1.scatter(df_bg['Longitude'], df_bg['Latitude'], 
            c='lightgray', s=1, alpha=0.1, label='Outside/Lateral')

# B. Plot Foreground (Red/Blue) SECOND
# We plot these one by one to ensure colors are correct
# Upstream (Blue)
upstream = df_fg[df_fg['Zone'] == "Upstream (Control)"]
ax1.scatter(upstream['Longitude'], upstream['Latitude'], 
            c='blue', s=15, alpha=1.0, label='Upstream (Control)')

# Downstream (Red)
downstream = df_fg[df_fg['Zone'] == "Wake Zone (Downstream)"]
ax1.scatter(downstream['Longitude'], downstream['Latitude'], 
            c='red', s=15, alpha=1.0, label='Wake Zone (Downstream)')

# C. Plot Turbines ON TOP
ax1.scatter(local_turbines['xlong'], local_turbines['ylat'], 
            c='black', marker='^', s=50, label='Turbines', zorder=10)

ax1.set_title("Combined Zonal Segmentation (All Days)", fontweight='bold')
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.legend(loc='lower left', markerscale=2) # Scale up legend markers
ax1.axis('equal')

# PANEL 2: Boxplots
ax2 = fig.add_subplot(222)
sns.boxplot(data=df_fg, x='Zone', y='Wind_Speed', 
            palette={'Upstream (Control)': 'blue', 'Wake Zone (Downstream)': 'red'}, ax=ax2)
ax2.set_title("Wind Speed Deficit (Aggregate)", fontweight='bold')
ax2.set_ylabel("Wind Speed (m/s)")

ax3 = fig.add_subplot(224)
if 'TKE_resolved' in df_fg.columns:
    y_metric = 'TKE_resolved'
    label = "TKE (m²/s²)"
else:
    y_metric = 'Wind_Speed'
    label = "Wind Variance (Proxy)" 
    
sns.boxplot(data=df_fg, x='Zone', y=y_metric, 
            palette={'Upstream (Control)': 'blue', 'Wake Zone (Downstream)': 'red'}, ax=ax3, showfliers=False)
ax3.set_title("Turbulence Intensity (Aggregate)", fontweight='bold')
ax3.set_ylabel(label)

plt.tight_layout()
plt.savefig("Wake_Effect_Analysis_Combined_Fixed.png", dpi=200)
print("Saved: Wake_Effect_Analysis_Combined_Fixed.png")

# Final Stats
stats = df_fg.groupby('Zone')['Wind_Speed'].mean()
if "Upstream (Control)" in stats and "Wake Zone (Downstream)" in stats:
    deficit = (stats["Upstream (Control)"] - stats["Wake Zone (Downstream)"]) / stats["Upstream (Control)"] * 100
    print(f"\nAGGREGATE RESULTS (3 DAYS):")
    print(f"  Upstream Mean:   {stats['Upstream (Control)']:.2f} m/s")
    print(f"  Downstream Mean: {stats['Wake Zone (Downstream)']:.2f} m/s")
    print(f"  -> WAKE DEFICIT: {deficit:.1f}%")