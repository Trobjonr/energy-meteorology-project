import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ==========================================================
# USER CONFIGURATION
# ==========================================================

DATA_FOLDER = "./" 
TARGET_DATE = "20230920"  # Updated to Sept 20

# ==========================================================
# PROCESSING
# ==========================================================

print(f"Looking for files with date: {TARGET_DATE}...")

# Find all hourly files for that day
search_pattern = os.path.join(DATA_FOLDER, f"*{TARGET_DATE}*.txt")
files = sorted(glob.glob(search_pattern))

if not files:
    print("No files found! Check your folder path and date.")
    exit()

print(f"Found {len(files)} hourly files. Combining...")

dfs = []
for f in files:
    try:
        # Header is on line 1 (index 1), skip units on lines 2,3
        tmp = pd.read_csv(f, header=1, skiprows=[2,3], na_values='NAN')
        dfs.append(tmp)
    except Exception as e:
        print(f"Skipping {f}: {e}")

if not dfs:
    print("Error: Could not read any files.")
    exit()

df = pd.concat(dfs, ignore_index=True)

# Convert Time
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

# CRITICAL: Convert UTC to CDT (-5 Hours)
df['Time_Local'] = df['TIMESTAMP'] - pd.Timedelta(hours=5)
df = df.sort_values('Time_Local')

# Filter for the relevant morning window (06:00 to 12:00 CDT)
# This covers the 07:21 - 09:46 flight nicely
df_zoom = df[(df['Time_Local'].dt.hour >= 6) & (df['Time_Local'].dt.hour <= 12)]

# ==========================================================
# VISUALIZATION
# ==========================================================

fig, ax1 = plt.subplots(figsize=(12, 6))

# 1. Plot Solar Radiation
ax1.fill_between(df_zoom['Time_Local'], df_zoom['SlrFD_W_Avg'], color='gold', alpha=0.3, label='Solar Radiation')
ax1.set_ylabel("Solar Radiation (W/m²)", color='goldenrod', fontweight='bold', fontsize=12)
ax1.tick_params(axis='y', labelcolor='goldenrod')
ax1.set_ylim(0, 600)

# 2. Plot Turbulence
ax2 = ax1.twinx()
ax2.plot(df_zoom['Time_Local'], df_zoom['WS_ms_Std'], color='navy', linewidth=2, label='Turbulence (Wind Std)')
ax2.set_ylabel("Wind Std Deviation (m/s)", color='navy', fontweight='bold', fontsize=12)
ax2.tick_params(axis='y', labelcolor='navy')
ax2.set_ylim(0, 1.5)

# 3. Add Flight Window Indicators (07:21 to 09:46)
flight_start = pd.Timestamp(f"{TARGET_DATE[:4]}-{TARGET_DATE[4:6]}-{TARGET_DATE[6:]} 07:21:00")
flight_end   = pd.Timestamp(f"{TARGET_DATE[:4]}-{TARGET_DATE[4:6]}-{TARGET_DATE[6:]} 09:46:00")

ax1.axvline(flight_start, color='green', linestyle='--', linewidth=2, label="Flight Start (07:21)")
ax1.axvline(flight_end, color='red', linestyle='--', linewidth=2, label="Flight End (09:46)")

# Shade the flight area
ax1.axvspan(flight_start, flight_end, color='gray', alpha=0.1)

# Formatting
ax1.set_title(f"Ground-Based Stability: {TARGET_DATE} (Flight Window)", fontweight='bold', fontsize=14)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.set_xlabel("Time of Day (CDT)", fontweight='bold')
ax1.grid(True, alpha=0.3)

# Legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.tight_layout()
output_file = f"Ground_Truth_Stability_{TARGET_DATE}_Flight.png"
plt.savefig(output_file, dpi=200)
print(f"Saved plot: {output_file}")
plt.show()