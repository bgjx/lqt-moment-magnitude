import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import csv


hypo_path = Path("hypo_catalog.xlsx")
picking_path = Path("picking_catalog.xlsx")

hypo_df = pd.read_excel(hypo_path, index_col=None)
picking_df = pd.read_excel(picking_path, index_col=None)


# build csv heaer 
csv_file = open("new_catalog_build.csv", 'w', newline='')
csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
header = [
    "network", "event_id", "source_lat", "source_lon", "source_depth_m", "source_origin_time", "station", "p_arr_time", "p_onset", "p_polarity", "s_arr_time", "coda_time", "remarks"
]
csv_writer.writerow(header)
csv_file.flush()


network = "SE"

for id in hypo_df.get("ID"):
    pick_data = picking_df[picking_df["Event ID"] == id]
    hypo_info = hypo_df[hypo_df.ID == id].iloc[0]
    _id, source_lat, source_lon, source_depth_m, year, month, day, hour, minute, t0 = hypo_info.ID, hypo_info.Lat, hypo_info.Lon, hypo_info.Depth, hypo_info.Year, hypo_info.Month, hypo_info.Day, hypo_info.Hour, hypo_info.Minute, hypo_info.T0
    int_t0 = int(t0)
    microsecond = int((t0 - int_t0)*1e6)
    source_origin_time =datetime(int(year), int(month), int(day), int(hour), int(minute), int_t0, microsecond)
    for station in pick_data.get("Station"):
        station_info = pick_data[pick_data.Station == station].iloc[0]
        station_name = station_info.Station
        year, month, day, hour, minute_p, second_p, p_onset, p_polarity, minute_s, second_s= station_info.Year, station_info.Month, station_info.Day, station_info.Hour, station_info.Minutes_P, station_info.P_Arr_Sec, station_info.P_Onset, station_info.P_Polarity, station_info.Minutes_S, station_info.S_Arr_Sec
        int_p_second = int(second_p)
        microsecond_p = int((second_p - int_p_second)*1e6)
        int_s_second = int(second_s)
        microsecond_s = int((second_s - int_s_second)*1e6)

        p_arr_time = datetime(year, month, day, hour, minute_p, int_p_second, microsecond_p)
        s_arr_time = datetime(year, month, day, hour, minute_s, int_s_second, microsecond_s)

        csv_writer.writerow([
            network, id, source_lat, source_lon, source_depth_m, source_origin_time, station_name, p_arr_time, p_onset, p_polarity, s_arr_time, "", ""
        ])
csv_file.close()
