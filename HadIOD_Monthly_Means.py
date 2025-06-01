import numpy as np
import netCDF4
import os
import shutil
import glob
from netCDF4 import Dataset
import pandas as pd

'''
This program assumes that each year's .nc files are all stored in a single nc_dir, as expected from a direct zip download from the HadIOD download page at https://www.metoffice.gov.uk/hadobs/hadiod/download-hadiod1-2-0-0.html.
'''

path = "~" # <- Edit this yourself

grid_res = 2  # degrees
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
for year in range(1850, 2025):
    nc_dir = f"{path}/HadIOD1200.data.{year}/"
    
    for month in months:
        file_list = sorted(glob.glob(os.path.join(nc_dir, f"hadiod1200_{year}{month}*.nc")))
        all_records = []

        for file in file_list:
            print(f"Processing {file[-22:]}...")
            nc = Dataset(file)

            basename = os.path.basename(file)
            day = int(basename[-5:-3])

            row_size = np.ma.filled(nc.variables["rowSize"])
            lat = np.ma.filled(nc.variables["lat"])
            lon = np.ma.filled(nc.variables["lon"])

            depth = np.ma.filled(nc.variables["depth"])
            depth_corr = np.ma.filled(nc.variables["depth_corr"], 0)
            depth_corr = [0 if x == 99999 else x for x in depth_corr]

            temp = np.ma.filled(nc.variables["temp"], 99999)
            temp_type_corr = np.ma.filled(nc.variables["temp_type_corr"], 0)
            temp_plat_corr = np.ma.filled(nc.variables["temp_plat_corr"], 0)
            temp_type_corr = [0 if x == 99999 else x for x in temp_type_corr]
            temp_plat_corr = [0 if x == 99999 else x for x in temp_plat_corr]

            potemp = np.ma.filled(nc.variables["potemp"], np.nan)

            sal = np.ma.filled(nc.variables["sal"], 99999)
            sal_type_corr = np.ma.filled(nc.variables["sal_type_corr"], 0)
            sal_plat_corr = np.ma.filled(nc.variables["sal_plat_corr"], 0)
            sal_type_corr = [0 if x == 99999 else x for x in sal_type_corr]
            sal_plat_corr = [0 if x == 99999 else x for x in sal_plat_corr]

            start_idx = np.cumsum(np.insert(row_size[:-1], 0, 0))
            end_idx = np.cumsum(row_size)

            for i in range(len(row_size)):
                s = start_idx[i]
                e = end_idx[i]

                if e > len(depth):  # in case
                    continue

                profile_depths = depth[s:e] + depth_corr[s:e]
                profile_temps = temp[s:e] + temp_type_corr[s:e] + temp_plat_corr[s:e]
                profile_potemps = potemp[s:e]
                profile_sals = sal[s:e] + sal_type_corr[s:e] + sal_plat_corr[s:e]

                # Grid bin (unfortunately, I did not include the option to save the offsets for each measurement)
                grid_lon = grid_res * round(lon[i] / grid_res)
                grid_lat = grid_res * round(lat[i] / grid_res)

                for d, temp_val, potemp_val, sal_val in zip(profile_depths, profile_temps, profile_potemps, profile_sals):
                    all_records.append((year, month, day, grid_lon, grid_lat, d, temp_val, potemp_val, sal_val))
            nc.close()

        df = pd.DataFrame(all_records, columns=["Year", "Month", "Day", "Longitude", "Latitude", "Depth", "T", "θ", "H"])
        df = df.replace(99999, np.nan)
        grouped = df.groupby(["Year", "Month", "Longitude", "Latitude", "Depth"], as_index=False).mean()
        grouped = grouped.fillna("")
        grouped.to_csv(f"{path}/csvs/hadiod1200_{year}{month}.csv", index=False)
        print(f"Exported to hadiod1200_{year}{month}.csv")
