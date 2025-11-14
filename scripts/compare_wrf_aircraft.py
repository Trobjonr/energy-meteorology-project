#!/usr/bin/env python3
"""
compare_wrf_aircraft_full.py

Reads one WRF output file and one aircraft .tab file (AWAKEN_...) and:
 - unstagggers U/V
 - computes heights from PH + PHB
 - matches in time
 - bilinearly interpolates horizontally to aircraft lat/lon
 - linearly interpolates vertically to aircraft altitude (m)
 - computes bias/RMSE and plots/saves results and CSV

Edit file paths below if needed.
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ------------------------
# User inputs / filenames
# ------------------------
wrf_file = 'wrfout_d01_2023-08-31_00%3A00%3A00.nc'
plane_file = 'AWAKEN_20230831_1204.tab'
out_csv = 'wrf_vs_aircraft_matched.csv'
out_prefix = 'wrf_vs_aircraft'  # plots will be saved with this prefix

# ------------------------
# Helper functions
# ------------------------
def unstagger_U(U):
    """Unstagger U along the last axis (x) by averaging adjacent points.
    Works for arrays with shape (..., ny, nx_stag) -> returns (..., ny, nx)
    """
    return 0.5 * (U[..., :-1] + U[..., 1:])

def unstagger_V(V):
    """Unstagger V along the second-to-last axis (y) by averaging adjacent points.
    Works for arrays with shape (..., ny_stag, nx) -> returns (..., ny, nx)
    """
    return 0.5 * (V[..., :-1, :] + V[..., 1:, :])

def bilinear_on_grid(var2d, lat2d, lon2d, lat_pt, lon_pt):
    """
    Simple bilinear interpolation on the structured lat2d/lon2d grid.
    - lat2d, lon2d are 2D arrays of shape (ny, nx)
    - var2d is 2D array shape (ny, nx)
    - returns interpolated value at (lat_pt, lon_pt)
    This assumes the grid is reasonably smooth (typical WRF grid).
    If the point is outside the grid, returns nearest neighbor.
    """
    ny, nx = lat2d.shape

    # find nearest grid indices by absolute difference in lat and lon separately
    j = np.argmin(np.abs(lat2d[:, 0] - lat_pt))  # approx row
    i = np.argmin(np.abs(lon2d[0, :] - lon_pt))  # approx col

    # refine: use global nearest point as start
    flat_idx = np.argmin((lat2d - lat_pt)**2 + (lon2d - lon_pt)**2)
    r0, c0 = np.unravel_index(flat_idx, lat2d.shape)

    # pick neighbor indices for bilinear: r0/r1 and c0/c1
    r0 = int(r0); c0 = int(c0)
    r1 = r0 + 1 if r0 + 1 < ny else r0 - 1
    c1 = c0 + 1 if c0 + 1 < nx else c0 - 1

    # if we ended up picking same indices (grid size 1 or edges), fallback to nearest
    if r1 == r0 or c1 == c0:
        return var2d[r0, c0]

    # corners lat/lon (not used in weight calc here, but useful for robust weighting)
    Q11 = var2d[r0, c0]
    Q21 = var2d[r0, c1]
    Q12 = var2d[r1, c0]
    Q22 = var2d[r1, c1]

    # build local coordinates for interpolation using lat/lon rectangle
    x1 = lon2d[r0, c0]; x2 = lon2d[r0, c1]
    y1 = lat2d[r0, c0]; y2 = lat2d[r1, c0]

    # deal with degenerate cases
    if x2 == x1 or y2 == y1:
        return var2d[r0, c0]

    # bilinear interpolation weights
    # transform point to (x,y) in rectangle
    x = lon_pt; y = lat_pt
    # normalize
    tx = (x - x1) / (x2 - x1)
    ty = (y - y1) / (y2 - y1)

    # clamp
    tx = np.clip(tx, 0.0, 1.0)
    ty = np.clip(ty, 0.0, 1.0)

    # bilinear
    val = (Q11 * (1 - tx) * (1 - ty) +
           Q21 * tx * (1 - ty) +
           Q12 * (1 - tx) * ty +
           Q22 * tx * ty)
    return val

def compute_stats(obs, mod):
    mask = np.isfinite(obs) & np.isfinite(mod)
    if np.count_nonzero(mask) == 0:
        return np.nan, np.nan
    bias = np.mean(mod[mask] - obs[mask])
    rmse = np.sqrt(np.mean((mod[mask] - obs[mask])**2))
    return bias, rmse

# ------------------------
# 1) Read WRF
# ------------------------
print("Loading WRF file:", wrf_file)
ds = xr.open_dataset(wrf_file)

# grab variables (assume present)
if 'U' not in ds or 'V' not in ds:
    raise RuntimeError("U or V not found in WRF file.")

U_raw = ds['U'].values      # shape (time, zstagger?, ny, nx_stag?) depending on file
V_raw = ds['V'].values
# PH and PHB required for heights
if ('PH' not in ds) or ('PHB' not in ds):
    raise RuntimeError("PH and/or PHB not found in WRF file; needed for heights.")

PH = ds['PH'].values
PHB = ds['PHB'].values

# XLAT/XLONG - detect 2D lat/lon
if 'XLAT' in ds and 'XLONG' in ds:
    xlats = ds['XLAT'].values
    xlongs = ds['XLONG'].values
    # XLAT/XLONG can be 3D (time, ny, nx) or 2D (ny,nx)
    if xlats.ndim == 3:
        lat2d = xlats[0, :, :]
    elif xlats.ndim == 2:
        lat2d = xlats
    else:
        raise RuntimeError("Unexpected XLAT dimension.")
    if xlongs.ndim == 3:
        lon2d = xlongs[0, :, :]
    elif xlongs.ndim == 2:
        lon2d = xlongs
    else:
        raise RuntimeError("Unexpected XLONG dimension.")
else:
    raise RuntimeError("XLAT/XLONG not found in WRF file.")

# WRF time array
if 'XTIME' in ds:
    wrf_times = pd.to_datetime(ds['XTIME'].values)
else:
    # fallback to time coordinate if available
    wrf_times = pd.to_datetime(ds['time'].values)

# ------------------------
# Unstagger U & V to mass points
# ------------------------
# U is staggered in x (last axis), V staggered in y (second-to-last axis)
# We'll compute unstaggered arrays: U_mass(t, z, y, x), V_mass(t, z, y, x)
print("Unstaggering U and V ...")
# Determine axes: assume last two axes are (south_north, west_east) ordering
# We'll treat U_raw and V_raw generically: last axis is x-like, second-last is y-like.
U_mass = unstagger_U(U_raw)   # reduces last axis by 1
V_mass = unstagger_V(V_raw)   # reduces second-last axis by 1

# Now U_mass and V_mass should share the same horizontal shape as lat2d/lon2d
# Verify shapes agree on (ny,nx)
_, _, ny_u, nx_u = U_mass.shape
_, _, ny_v, nx_v = V_mass.shape
ny, nx = lat2d.shape
if not (ny == ny_u == ny_v and nx == nx_u == nx_v):
    # try transposing axes if shapes mismatch (rare); otherwise warn and continue with nearest-grid fallback
    print("Warning: unstaggered U/V horizontal shapes do not match XLAT/XLONG.")
    print(f"lat2d: {lat2d.shape}, U_mass: {U_mass.shape}, V_mass: {V_mass.shape}")
    # We'll proceed but bilinear may fallback to nearest in some cases.

# ------------------------
# Compute geometric height z (m) from PH + PHB
# PH has one extra vertical level (staggered). Average to mass levels.
# ------------------------
print("Computing height from PH and PHB ...")
g = 9.81
# PH shape: (time, z_stag, ny, nx)
# average adjacent z levels to mass levels: z_mass = 0.5*(PH[:, :-1,...] + PH[:, 1:,...]) + same for PHB, then / g
PH_total = PH + PHB
Z_mass = 0.5 * (PH_total[:, :-1, :, :] + PH_total[:, 1:, :, :]) / g  # dims (time, z, ny, nx)
# Ensure Z_mass vertical dimension matches U_mass/V_mass vertical dimension
# U_mass shape: (time, z, ny, nx) hopefully matching

# ------------------------
# 2) Read aircraft file
# ------------------------
print("Loading aircraft file:", plane_file)
df = pd.read_csv(plane_file, sep='\t')
# Trim whitespace on column names
df.columns = [c.strip() for c in df.columns]
print("Columns found:", df.columns.tolist())

# Convert Date/Time to pandas datetime
df['Date_Time'] = pd.to_datetime(df['Date/Time'])

# Use earth-referenced U/V columns (short names)
# confirm presence
if 'U [m/s] (east component)' in df.columns and 'V [m/s] (north component)' in df.columns:
    df['U_aircraft'] = df['U [m/s] (east component)']
    df['V_aircraft'] = df['V [m/s] (north component)']
else:
    # fallback to the long names if short ones absent (but you indicated short exist)
    df['U_aircraft'] = df['Aircraft vel U [m/s] (aircraft east)']
    df['V_aircraft'] = df['Aircraft vel V [m/s] (aircraft north)']

# altitude column
if 'Altitude [m]' in df.columns:
    alt_col = 'Altitude [m]'
else:
    raise RuntimeError("Altitude column not found in aircraft file.")

# compute observed wind speed
df['wind_obs'] = np.sqrt(df['U_aircraft']**2 + df['V_aircraft']**2)

# ------------------------
# 3) Time match: find nearest WRF time index per aircraft row
# ------------------------
print("Matching times to WRF timesteps ...")
df['wrf_time_idx'] = df['Date_Time'].apply(lambda t: np.argmin(np.abs(wrf_times - t)))

# ------------------------
# 4) Horizontal indices (we will not precompute indices for bilinear; we will search per point)
# ------------------------

# Prepare arrays to store results
N = len(df)
U_model = np.full(N, np.nan)
V_model = np.full(N, np.nan)
wind_model = np.full(N, np.nan)
wrf_time_for_row = np.full(N, None)

print("Interpolating (horizontal bilinear + vertical linear) to each aircraft point ...")
for idx, row in tqdm(df.iterrows(), total=N, desc="Interpolating"):
    t_idx = int(row['wrf_time_idx'])
    lat_pt = float(row['Latitude'])
    lon_pt = float(row['Longitude'])
    z_pt = float(row[alt_col])  # meters

    # choose the appropriate 2D fields for this time
    # U_mass[t_idx, z, y, x] and V_mass[t_idx, z, y, x]
    # Z_mass[t_idx, z, y, x]
    try:
        U_t = U_mass[t_idx]   # shape (z, ny, nx)
        V_t = V_mass[t_idx]
        Z_t = Z_mass[t_idx]
    except Exception:
        # fallback: if time dimension mismatch, try index 0
        U_t = U_mass[0]
        V_t = V_mass[0]
        Z_t = Z_mass[0]

    nz = Z_t.shape[0]

    # First: for each vertical level, bilinearly interpolate horizontal U_t[level], V_t[level], and Z_t[level]
    u_profile = np.full(nz, np.nan)
    v_profile = np.full(nz, np.nan)
    z_profile = np.full(nz, np.nan)

    for k in range(nz):
        try:
            u_profile[k] = bilinear_on_grid(U_t[k], lat2d, lon2d, lat_pt, lon_pt)
            v_profile[k] = bilinear_on_grid(V_t[k], lat2d, lon2d, lat_pt, lon_pt)
            z_profile[k] = bilinear_on_grid(Z_t[k], lat2d, lon2d, lat_pt, lon_pt)
        except Exception:
            # fallback to nearest neighbor
            flat_idx = np.argmin((lat2d - lat_pt)**2 + (lon2d - lon_pt)**2)
            r, c = np.unravel_index(flat_idx, lat2d.shape)
            u_profile[k] = U_t[k, r, c]
            v_profile[k] = V_t[k, r, c]
            z_profile[k] = Z_t[k, r, c]

    # Now we have vertical profiles u_profile(k), v_profile(k), z_profile(k)
    # Check monotonicity of z_profile for interpolation; if not monotonic, sort by z
    # Also remove nan levels
    valid = np.isfinite(z_profile) & np.isfinite(u_profile) & np.isfinite(v_profile)
    if np.count_nonzero(valid) < 2:
        # insufficient data
        continue

    z_valid = z_profile[valid]
    u_valid = u_profile[valid]
    v_valid = v_profile[valid]

    # sort by height (ascending)
    order = np.argsort(z_valid)
    z_valid = z_valid[order]
    u_valid = u_valid[order]
    v_valid = v_valid[order]

    # If requested altitude z_pt is outside the profile range, we will extrapolate using nearest level
    if z_pt <= z_valid[0]:
        u_at = u_valid[0]
        v_at = v_valid[0]
    elif z_pt >= z_valid[-1]:
        u_at = u_valid[-1]
        v_at = v_valid[-1]
    else:
        # linear interpolation in z
        u_at = np.interp(z_pt, z_valid, u_valid)
        v_at = np.interp(z_pt, z_valid, v_valid)

    U_model[idx] = u_at
    V_model[idx] = v_at
    wind_model[idx] = np.sqrt(u_at**2 + v_at**2)
    wrf_time_for_row[idx] = str(wrf_times[t_idx])

# attach model results to df
df['U_model'] = U_model
df['V_model'] = V_model
df['wind_model'] = wind_model
df['wrf_time_for_row'] = wrf_time_for_row

# ------------------------
# 5) Statistics
# ------------------------
print("Computing statistics ...")
bias_U, rmse_U = compute_stats(df['U_aircraft'].values, df['U_model'].values)
bias_V, rmse_V = compute_stats(df['V_aircraft'].values, df['V_model'].values)
bias_speed, rmse_speed = compute_stats(df['wind_obs'].values, df['wind_model'].values)

print(f"U bias = {bias_U:.3f} m/s, U RMSE = {rmse_U:.3f} m/s")
print(f"V bias = {bias_V:.3f} m/s, V RMSE = {rmse_V:.3f} m/s")
print(f"Speed bias = {bias_speed:.3f} m/s, Speed RMSE = {rmse_speed:.3f} m/s")

# ------------------------
# 6) Save CSV of matched results
# ------------------------
print("Saving CSV to", out_csv)
out_df = df[[
    'Date_Time', 'Latitude', 'Longitude', alt_col,
    'U_aircraft', 'V_aircraft', 'wind_obs',
    'U_model', 'V_model', 'wind_model', 'wrf_time_for_row'
]].copy()
out_df.to_csv(out_csv, index=False)

# ------------------------
# 7) Plot and save figures
# ------------------------
print("Saving plots ...")
plt.figure(figsize=(6,8))
plt.scatter(df['wind_obs'], df[alt_col], label='Aircraft', s=10)
plt.scatter(df['wind_model'], df[alt_col], label='WRF (interp)', s=10)
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Altitude (m)')
plt.legend()
plt.grid(True)
plt.title('Wind speed profile (obs vs WRF)')
plt.savefig(out_prefix + '_speed_profile.png', dpi=200)
plt.close()

plt.figure(figsize=(6,8))
plt.scatter(df['U_aircraft'], df[alt_col], label='Aircraft U', s=10)
plt.scatter(df['U_model'], df[alt_col], label='WRF U', s=10)
plt.xlabel('U (m/s)')
plt.ylabel('Altitude (m)')
plt.legend()
plt.grid(True)
plt.title('U profile (obs vs WRF)')
plt.savefig(out_prefix + '_U_profile.png', dpi=200)
plt.close()

plt.figure(figsize=(6,8))
plt.scatter(df['V_aircraft'], df[alt_col], label='Aircraft V', s=10)
plt.scatter(df['V_model'], df[alt_col], label='WRF V', s=10)
plt.xlabel('V (m/s)')
plt.ylabel('Altitude (m)')
plt.legend()
plt.grid(True)
plt.title('V profile (obs vs WRF)')
plt.savefig(out_prefix + '_V_profile.png', dpi=200)
plt.close()

print("Done. CSV and plots saved with prefix:", out_prefix)
