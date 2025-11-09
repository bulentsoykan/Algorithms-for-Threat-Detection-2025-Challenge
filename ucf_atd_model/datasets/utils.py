import pyarrow as pa
import pandas as pd
from ..data import data_loc

schema = pa.schema([
    pa.field("point_id", pa.int64()),
    pa.field("track_id", pa.int64()),
    pa.field("time", pa.time32("s")),
    pa.field("lat", pa.float64()),
    pa.field("lon", pa.float64()),
    pa.field("speed", pa.float64()),
    pa.field("course", pa.float64())
])

timelookup = pd.read_csv(data_loc("timeCodes.csv"))

# Cuts dataset into 5 minute intervals, enforces uniqueness of points within each interval
def discrete_time(data: pd.DataFrame, time_col = "time", unique_cols = ["batch", "track_id"]) -> pd.DataFrame:
    data = data.copy()
    
    # Do the cut
    data[time_col] = pd.cut(data["time"], 288)
    
    # Enforce uniqueness
    shouldUnique = data.groupby(by=[*unique_cols, "time"], observed=True).groups
    idxs_to_keep = [x[0] for x in shouldUnique.values()]
    
    # Convert back to original data type
    outData = data.iloc[idxs_to_keep].sort_index().reset_index(drop=True)
    outData["time"] = outData["time"].cat.codes
    
    return outData
