import xarray as xr
import numpy as np
import pandas as pd

file_path = r"C:\Users\Andrew\Documents\jlundqu2data\wrfout_d01_2023-08-31_00%3A00%3A00"
output_excel = r"C:\Users\Andrew\Documents\jlundqu2data\wrf_winds.xlsx"

# Open with xarray
ds = xr.open_dataset(file_path)

# Extract U10 and V10
u10 = ds["U10"]
v10 = ds["V10"]

# Compute wind speed
wspd = np.sqrt(u10**2 + v10**2)

# Get lat/lon
lats = ds["XLAT"].isel(Time=0)
lons = ds["XLONG"].isel(Time=0)

# Flatten and save
df = pd.DataFrame({
    "lat": lats.values.flatten(),
    "lon": lons.values.flatten(),
    "u10": u10.values.flatten(),
    "v10": v10.values.flatten(),
    "wspd": wspd.values.flatten()
})

df.to_excel(output_excel, index=False)
print(f"✅ Data saved to: {output_excel}")

