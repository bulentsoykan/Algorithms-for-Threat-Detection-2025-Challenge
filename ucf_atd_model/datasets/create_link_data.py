import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from datetime import *

import pyproj as pp
import pyarrow.parquet as pq
import pandas as pd
from ..data import data_loc
from sys import argv
import traceback
import time

wgs84 = pp.CRS.from_epsg(4326)
utm = pp.CRS.from_epsg(32616)
to_utm = pp.Transformer.from_crs(wgs84, utm)

columns = ["distance_m", "implied_speed_knots", "delta_speed", "delta_course", "bearing_diff", "kinematic_error", "delta_time", "y1", "y2", "x1", "x2", "t1", "t2", "speed1", "speed2", "course1", "course2"]
eval_columns = [
    "distance_m",
    "implied_speed_knots",
    "delta_speed",
    "delta_course",
    "bearing_diff",
    "kinematic_error",
    "delta_time",
    "y1", 
    "y2", 
    "x1",
    "x2",
    "t1",
    "t2",
    "speed1",
    "speed2",
    "course1",
    "course2",
    "dx1",
    "dy1",
    "dx2",
    "dy2"
]

def get_data(i):
    # Read in historical data
    with pq.ParquetFile(data_loc("historical.parquet")) as fulldata:
        rowgroup = fulldata.read_row_group(i)
        data: pd.DataFrame = rowgroup.to_pandas()
        data["time"] = data["time"].apply(lambda x: datetime.combine(datetime.fromtimestamp(0).date(), x))
    
    return data

# --- Helper Functions  ---
def haversine_distance_m(lat1, lon1, lat2, lon2):
    R = 6371000; phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    delta_phi, delta_lambda = np.deg2rad(lat2 - lat1), np.deg2rad(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) 
    
    return R * c

def project_forward(lat, lon, speed, course, dt_seconds):
    knots_to_deg_per_sec = 1.852 / 3600 / 111.32; course_rad = np.deg2rad(course)
    proj_lat = lat + (speed * knots_to_deg_per_sec * np.cos(course_rad) * dt_seconds)
    proj_lon = lon + (speed * knots_to_deg_per_sec * np.sin(course_rad) * dt_seconds) / np.cos(np.deg2rad(lat))
    
    return proj_lat, proj_lon

def get_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dLon = lon2 - lon1
    x = np.sin(dLon) * np.cos(lat2); y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    bearing = np.rad2deg(np.arctan2(x, y)) 
    
    return (bearing + 360) % 360

def calculate_link_features(p1: pd.DataFrame, p2: pd.DataFrame, eval=False):
    """Calculates a feature vector for a potential link between two points."""
    features = None
    if type(p1) is pd.DataFrame:
        features = p1.copy()
        features['delta_time'] = (p2['time'] - p1['time']).dt.total_seconds()
    else:
        features = pd.DataFrame(p1)
        features["delta_time"] = (p2["time"].to_numpy() - p1["time"]).astype("timedelta64[s]").astype("int")
    filter = (0 < features["delta_time"])

    features['distance_m'] = haversine_distance_m(p1['lat'], p1['lon'], p2['lat'], p2['lon'])
    features['implied_speed_knots'] = (features['distance_m'] / features['delta_time']) * 1.94384
    features['delta_speed'] = abs(p2['speed'] - p1['speed'])
    features['delta_course'] = 180 - abs(180 - abs(p2['course'] - p1['course']))
    bearing = get_bearing(p1['lat'], p1['lon'], p2['lat'], p2['lon'])
    features['bearing_diff'] = 180 - abs(180 - abs(bearing - p1['course']))
    proj_lat, proj_lon = project_forward(p1['lat'], p1['lon'], p1['speed'], p1['course'], features['delta_time'])
    features['kinematic_error'] = haversine_distance_m(p2['lat'], p2['lon'], proj_lat, proj_lon)
    
    if eval:
        x1 = y1 = x2 = y2 = None
        if type(p1) is pd.DataFrame:
            x1, y1 = to_utm.transform(p1["lat"].to_numpy(), p1["lon"].to_numpy())
            x2, y2 = to_utm.transform(p2["lat"].to_numpy(), p2["lon"].to_numpy())
        else:
            x1, y1 = to_utm.transform(p1["lat"], p1["lon"])
            x2, y2 = to_utm.transform(p2["lat"], p2["lon"])

        dx1 = 0.000514444 *  p1["speed"] * np.sin(p1["course"] * (np.pi) / 180)
        dy1 = 0.000514444 * p1["speed"] * np.cos(p1["course"] * (np.pi) / 180)
        dx2 = 0.000514444 * p2["speed"] * np.sin(p2["course"] * (np.pi) / 180)
        dy2 = 0.000514444 * p2["speed"] * np.cos(p2["course"] * (np.pi) / 180)

        features["y1"] = y1 / 1000
        features["x1"] = x1 / 1000
        features["y2"] = y2 / 1000
        features["x2"] = x2 / 1000

        features["dx1"] = dx1
        features["dx2"] = dx2
        features["dy1"] = dy1
        features["dy2"] = dy2

        features["t1"] = (pd.to_datetime(p1["time"]) - pd.to_datetime("1970-01-01 00:00:00")) / pd.offsets.Second(1)
        features["t2"] = (pd.to_datetime(p2["time"]) - pd.to_datetime("1970-01-01 00:00:00")) / pd.offsets.Second(1)
    
    else:
        features["y1"] = p1["lat"]
        features["y2"] = p2["lat"]
        features["x1"] = p1["lon"]
        features["x2"] = p2["lon"]
        features["t1"] = p1["time"]
        features["t2"] = p2["time"]

    features["speed1"] = p1["speed"]
    features["speed2"] = p2["speed"]
    features["course1"] = p1["course"]
    features["course2"] = p2["course"]

    if eval:
        return features[eval_columns][filter]
    else:
        return features[columns][filter]

def create_link_feature_dataset(df, n_distractors=7):
    """Processes historical data to create a training set for link prediction."""
    all_features = []
    df = df.sort_values(['track_id', 'time'])

    points_rad = np.deg2rad(df[['lat', 'lon']].values)
    tree = BallTree(points_rad, metric='haversine')

    for track_id, group in df.groupby('track_id'):
        if len(group) < 2: 
            continue
        
        p1 = group[:-1].reset_index(drop=True)
        p2 = group[1:].reset_index(drop=True)
        
        # Positive example
        features = calculate_link_features(p1, p2)
        if not features.empty: 
            features["label"] = 1
            all_features.append(features)

        # Negative example
        max_dist_m = (p2['time'] - p1['time']).dt.total_seconds() * 30 * 0.5144 # Max 30 knots travel
        candidate_idxs = tree.query_radius(np.deg2rad([p1['lat'], p1['lon']]).T, r=max_dist_m/6371000)
        shuffled = np.array([np.pad(np.random.permutation(x)[:n_distractors], (0, n_distractors - min(n_distractors, len(x)))) for x in candidate_idxs])
        for distractor_num in range(n_distractors):
            candidate_idx = shuffled[:, distractor_num]
            p_distractor = df.iloc[candidate_idx].reset_index(drop=True)
            filter = (p_distractor['point_id'] != p2['point_id']) & (p_distractor['time'] > p1['time'])
            features = calculate_link_features(p1[filter].reset_index(drop=True), p_distractor[filter].reset_index(drop=True))
            if not features.empty: 
                features['label'] = 0
                all_features.append(features)
    return pd.concat(all_features)

def run(i):
    print(f"Processing dataset {i}", flush=True)
    start = time.time()
    try:
        link_df = create_link_feature_dataset(get_data(i))
        data_path = data_loc("lgbm_data")
        link_df.to_csv(f"{data_path}/data_{i}.csv")
    except Exception:
        return traceback.format_exc()
    end = time.time()
    print(f"Finished dataset {i}: Took {end - start:.2f} seconds")
    return ""

if __name__ == "__main__":
    if len(argv) > 1:
        i = int(argv[1])
        link_df = create_link_feature_dataset(get_data(i))
        data_path = data_loc("lgbm_data")
        link_df.to_csv(f"{data_path}/data_{i}.csv")