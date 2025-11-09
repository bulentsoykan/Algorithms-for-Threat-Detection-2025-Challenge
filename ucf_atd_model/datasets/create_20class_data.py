import pandas as pd
from time import time
import pyarrow.parquet as pq
import numpy as np
from ucf_atd_model.data import data_loc, new_data_loc
from ..c20_consts import *
from ucf_atd_model.datasets.create_link_data import calculate_link_features, haversine_distance_m, project_forward
from datetime import datetime
import time
import traceback

# Pad with this when necessary
const_data = {key: -1.0 for key in link_features}

def subset(lastpts, i):
    columns = ["time", "lat", "lon", "speed", "course", "track_id_true"]
    return {key: lastpts[key][:i] for key in columns}

def setidx(lastpts, i, val):
    lastpts["time"][i] = val["time"].to_numpy()
    lastpts["lat"][i] = val["lat"]
    lastpts["lon"][i] = val["lon"]
    lastpts["speed"][i] = val["speed"]
    lastpts["course"][i] = val["course"]
    lastpts["track_id_true"][i] = val["track_id_true"]

def boolfilter(lastpts, bools):
    columns = ["time", "lat", "lon", "speed", "course", "track_id_true"]
    return {key: lastpts[key][bools] for key in columns}

def paddata(link_features):
    if link_features.shape[0] < n_norm_classes:
        to_add = n_norm_classes - link_features.shape[0]
        extra_rows = pd.DataFrame([const_data] * to_add)
        return pd.concat([link_features, extra_rows], ignore_index=True)
    else:
        return link_features

def create_data(df):
    """Implements the final ML-Enhanced Tracking algorithm."""
    df = df.sort_values('time').reset_index(drop=True)
    df['track_id'] = -1

    next_track_id = 0
    
    n = df.shape[0]
    lastPtInTrack = {
        "time": np.repeat(pd.Timestamp(year=1970, month=1, day=1, hour=0, minute=0, second=0).to_numpy(), n), 
        "lat": np.repeat(-1.0, n), 
        "lon": np.repeat(-1.0, n), 
        "speed": np.repeat(-1.0, n), 
        "course": np.repeat(-1.0, n),
        "track_id_true": np.repeat(-1, n)
    }

    xdata = []
    ydata = []

    for i in range(len(df)):
        p_current = df.iloc[i]

        if next_track_id == 0:
            df.loc[i, 'track_id'] = next_track_id
            setidx(lastPtInTrack, i, p_current)
            next_track_id += 1
            continue


        active_tracks_df = subset(lastPtInTrack, next_track_id)
        
        time_diff = (p_current["time"].to_numpy() - active_tracks_df["time"]).astype("timedelta64[s]").astype("int")

        max_dist_m = time_diff * 30 * 0.5144
        real_dist = haversine_distance_m(active_tracks_df["lat"], active_tracks_df["lon"], p_current["lat"], p_current["lon"])
        
        kinematic_errors = haversine_distance_m(p_current["lat"], p_current["lon"], *project_forward(active_tracks_df['lat'], active_tracks_df['lon'], active_tracks_df['speed'], active_tracks_df['course'], time_diff))
        error_cutoff = np.sort(kinematic_errors)[:n_norm_classes].max()
        kinematic_filter = kinematic_errors < error_cutoff
        
        loc_filter = real_dist < max_dist_m
        timeCorrect: np.ndarray = (0 < time_diff)
        idxs = np.arange(len(timeCorrect))

        big_filter = loc_filter & timeCorrect & kinematic_filter 

        # Create data if we find data points within the filters
        if np.any(big_filter):
            all_data = paddata(calculate_link_features(boolfilter(active_tracks_df, big_filter), p_current, eval=True))
            maindata = all_data[normal_features].to_numpy()
            otherdata = all_data[currpt_features].iloc[0].to_numpy()

            label = np.zeros(n_norm_classes + 1)
            id_correct = active_tracks_df["track_id_true"][big_filter] == p_current["track_id_true"]
            any_correct = np.any(id_correct)
            label[:id_correct.shape[0]] = id_correct
            if not any_correct:
                label[n_norm_classes] = 1
            
            toappend = np.zeros(n_norm_classes * len(normal_features) + len(currpt_features))

            raveled = np.ravel(maindata)
            toappend[:raveled.shape[0]] = raveled
            toappend[raveled.shape[0]:] = otherdata

            xdata.append(toappend)
            ydata.append(label)
            
            for perm in range(perms):
                idxs = np.random.permutation(n_norm_classes)
                toappend = np.zeros(n_norm_classes * len(normal_features) + len(currpt_features))
                
                raveled = np.ravel(maindata[idxs])
                toappend[:raveled.shape[0]] = raveled
                toappend[raveled.shape[0]:] = otherdata
                
                xdata.append(toappend)
                labelp = np.zeros(n_norm_classes + 1)
                if not any_correct:
                    labelp[n_norm_classes] = 1
                else:
                    labelp[:n_norm_classes] = label[idxs]
                ydata.append(labelp)

            # Assignment with a confidence threshold
            if any_correct:
                best_match_track_id = np.argmax(active_tracks_df["track_id_true"] == p_current["track_id_true"])
                df.loc[i, 'track_id'] = best_match_track_id
                setidx(lastPtInTrack, best_match_track_id, p_current)
            else:
                df.loc[i, 'track_id'] = next_track_id
                setidx(lastPtInTrack, next_track_id, p_current)
                next_track_id += 1
        else:
            df.loc[i, 'track_id'] = next_track_id
            setidx(lastPtInTrack, next_track_id, p_current)
            next_track_id += 1

    return df[['point_id', 'track_id']], xdata, ydata

def get_data(i):
    # Read in historical data
    with pq.ParquetFile(data_loc("historical.parquet")) as fulldata:
        rowgroup = fulldata.read_row_group(i)
        data: pd.DataFrame = rowgroup.to_pandas()
        data["time"] = data["time"].apply(lambda x: datetime.combine(datetime.fromtimestamp(0).date(), x))
    
    data["track_id_true"] = data["track_id"]
    return data

def run(i):
    print(f"Processing dataset {i}", flush=True)
    start = time.time()
    try:
        oracle, xdata, ydata = create_data(get_data(i))
        data_path = data_loc("class20")
        xdata = pd.DataFrame(np.array(xdata), columns=colnames)
        xdata.to_csv(f"{data_path}/xdata_{i}.csv")
        np.save(f"{data_path}/ydata_{i}.npy", np.array(ydata))
    except Exception:
        traceback.print_exc()
        return traceback.format_exc()
    end = time.time()
    print(f"Finished dataset {i}: Took {end - start:.2f} seconds")
    return ""
