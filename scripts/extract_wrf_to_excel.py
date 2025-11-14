import os
import xarray as xr
import numpy as np
import pandas as pd
from wrf import getvar, to_np, latlon_coords

# === USER SETTINGS ===
file_path = r"C:\Users\Andrew\Documents\jlundqu2data\wrfout_d01_2023-08-31_00%3A00%3A00"
output_excel = r"C:\Users\Andrew\Documents\jlundqu2data\wrf_winds.xlsx"

# === OPEN THE FILE ===
print("Opening file:", file_path)
ds = xr.open_dataset(file_path)

# === EXTRACT VARIABLES ===
print("Extracting 10m U and V wind components...")
u10 = getvar(ds, "U10")
v10 = getvar(ds, "V10")

# === COMPUTE WIND SPEED ===
wspd = np.sqrt(u10**2 + v10**2)

# === GET LAT/LON COORDINATES ===
lats, lons = latlon_coords(u10)

# === CONVERT TO NUMPY ARRAYS ===
u10_np = to_np(u10)
v10_np = to_np(v10)
wspd_np = to_np(wspd)
lats_np = to_np(lats)
lons_np = to_np(lons)

# === FLATTEN ARRAYS TO 1D ===
flat_data = {
    "lat": lats_np.flatten(),
    "lon": lons_np.flatten(),
    "u10": u10_np.flatten(),
    "v10": v10_np.flatten(),
    "wspd": wspd_np.flatten()
}

# === CREATE A DATAFRAME ===
df = pd.DataFrame(flat_data)

# === SAVE TO EXCEL ===
df.to_excel(output_excel, index=False)
print(f"✅ Data saved to: {output_excel}")

# === CLOSE FILE ===
ds.close()
