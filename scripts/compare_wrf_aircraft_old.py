import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar

# ----------------------------
# 1. Load WRF data
# ----------------------------
wrf_file = 'wrfout_d01_2023-08-31_00%3A00%3A00.nc'
ds = xr.open_dataset(wrf_file)

# Extract U and V
U = ds['U']  # east-west wind
V = ds['V']  # north-south wind

# Compute full geopotential height (meters) from PH + PHB
PH = ds['PH']    # perturbation geopotential (staggered vertically)
PHB = ds['PHB']  # base geopotential
g = 9.81         # gravity

# Average staggered levels to mass levels
Z = ((PH[:, :-1, :, :] + PH[:, 1:, :, :] + PHB[:, :-1, :, :] + PHB[:, 1:, :, :]) / 2) / g

# Latitude and longitude grids (take first timestep)
lat_2d = ds['XLAT'][0, :, :].values
lon_2d = ds['XLONG'][0, :, :].values

# Convert WRF time to pandas datetime
wrf_times = pd.to_datetime(ds['XTIME'].values)

# ----------------------------
# 2. Load aircraft data
# ----------------------------
plane_file = 'AWAKEN_20230831_1204.tab'
df = pd.read_csv(plane_file, sep='\t')

# Clean column names
df.columns = [c.strip() for c in df.columns]
print("Aircraft file columns after cleaning:", df.columns)

# Convert datetime
df['Date_Time'] = pd.to_datetime(df['Date/Time'])

# Assign aircraft U and V components
df['U_aircraft'] = df['Aircraft vel U [m/s] (aircraft east)']
df['V_aircraft'] = df['Aircraft vel V [m/s] (aircraft north)']
df['wind_speed'] = np.sqrt(df['U_aircraft']**2 + df['V_aircraft']**2)

# ----------------------------
# 3. Match WRF to aircraft in time
# ----------------------------
df['wrf_time_idx'] = df['Date_Time'].apply(lambda t: np.argmin(np.abs(wrf_times - t)))

# ----------------------------
# 4. Match WRF to aircraft in space (nearest grid point)
# ----------------------------
def nearest_grid(lat_array, lon_array, lat_pt, lon_pt):
    dist = np.sqrt((lat_array - lat_pt)**2 + (lon_array - lon_pt)**2)
    y_idx, x_idx = np.unravel_index(np.argmin(dist), dist.shape)
    return y_idx, x_idx

# Apply nearest_grid and expand into two columns
df[['wrf_y_idx', 'wrf_x_idx']] = df.apply(
    lambda row: pd.Series(nearest_grid(lat_2d, lon_2d, row['Latitude'], row['Longitude'])),
    axis=1
)

# ----------------------------
# 5. Interpolate WRF winds to aircraft altitudes
# ----------------------------
wrf_U, wrf_V, wrf_speed, wrf_altitudes = [], [], [], []

for i, row in tqdm(df.iterrows(), total=len(df), desc="Interpolating WRF to aircraft altitudes"):
    t_idx = row['wrf_time_idx']
    y_idx = row['wrf_y_idx']
    x_idx = row['wrf_x_idx']

    u_profile = U[t_idx, :, y_idx, x_idx].values
    v_profile = V[t_idx, :, y_idx, x_idx].values
    z_profile = Z[t_idx, :, y_idx, x_idx].values

    plane_alt = row['Altitude [m]']

    # Linear interpolation to aircraft altitude
    u_at_alt = np.interp(plane_alt, z_profile, u_profile)
    v_at_alt = np.interp(plane_alt, z_profile, v_profile)

    wrf_U.append(u_at_alt)
    wrf_V.append(v_at_alt)
    wrf_speed.append(np.sqrt(u_at_alt**2 + v_at_alt**2))
    wrf_altitudes.append(plane_alt)

# Convert lists to arrays
wrf_U = np.array(wrf_U)
wrf_V = np.array(wrf_V)
wrf_speed = np.array(wrf_speed)
wrf_altitudes = np.array(wrf_altitudes)

# ----------------------------
# 6. Compute bias and RMSE
# ----------------------------
def compute_stats(obs, model):
    bias = np.mean(model - obs)
    rmse = np.sqrt(np.mean((model - obs)**2))
    return bias, rmse

bias_U, rmse_U = compute_stats(df['U_aircraft'], wrf_U)
bias_V, rmse_V = compute_stats(df['V_aircraft'], wrf_V)
bias_speed, rmse_speed = compute_stats(df['wind_speed'], wrf_speed)

print(f"U component: bias={bias_U:.2f} m/s, RMSE={rmse_U:.2f} m/s")
print(f"V component: bias={bias_V:.2f} m/s, RMSE={rmse_V:.2f} m/s")
print(f"Wind speed: bias={bias_speed:.2f} m/s, RMSE={rmse_speed:.2f} m/s")

# ----------------------------
# 7. Plot profiles
# ----------------------------
plt.figure(figsize=(6,8))
plt.scatter(df['wind_speed'], df['Altitude [m]'], label='Aircraft', color='blue', s=20)
plt.scatter(wrf_speed, wrf_altitudes, label='WRF', color='red', s=20)
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Altitude (m)')
plt.title('Wind Speed Profile Comparison')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(6,8))
plt.scatter(df['U_aircraft'], df['Altitude [m]'], label='Aircraft U', color='blue', s=20)
plt.scatter(wrf_U, wrf_altitudes, label='WRF U', color='red', s=20)
plt.xlabel('U component (m/s)')
plt.ylabel('Altitude (m)')
plt.title('U Component Profile Comparison')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(6,8))
plt.scatter(df['V_aircraft'], df['Altitude [m]'], label='Aircraft V', color='blue', s=20)
plt.scatter(wrf_V, wrf_altitudes, label='WRF V', color='red', s=20)
plt.xlabel('V component (m/s)')
plt.ylabel('Altitude (m)')
plt.title('V Component Profile Comparison')
plt.legend()
plt.grid(True)
plt.show()
