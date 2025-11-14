# Information

.py scripts were created to be run on Python 3.10 or above. The following prerequisite libraries are required for several, if not all, of the scripts:

xarray 

netcdf4 

pandas 

numpy 

matplotlib 

tqdm

I run my scripts on Anaconda (conda), but whatever floats your boat.

# Scripts

compare_wrf_aircraft.py : attempts to directly analyze a .tab and .nc file for a given day and spit out usable plots of their wind profiles. Some optimization clearly needs to be done, as the process takes a few hours to complete.

compare_wrf_aircraft.py : same as above, but incorrectly used aircraft U and V rather than wind U and V.

extract_wrf.py : deprecated file. Attempted to make an excel version of .nc files, but replaced by extract_wrf_to_excel.py .

extract_wrf_to_excel.py : converts WRF .nc files to more human-friendly Excel formats. 
