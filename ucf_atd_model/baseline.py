import numpy as np
import pandas as pd
from .data import data_loc, ResultCache
from typing import *
import atd2025

cache = ResultCache("baseline")

def run(file: str) -> Tuple[pd.DataFrame, str]:
    in_cache, path = cache.test_cache(file)
    if in_cache:
        return pd.read_csv(path), path
    
    data = pd.read_csv(data_loc(file))
    data["time"] = pd.to_datetime(data["time"])
    points = data.apply(lambda x: atd2025.Point(x["point_id"], x["time"], x["lat"], x["lon"], x["course"], x["speed"]), axis=1)
    base_results = atd2025.baseline([x for x in points])
    atd2025.predictions_to_csv(path, base_results)
    basePreds = pd.read_csv(path)

    return basePreds, path

if __name__ == "__main__":
    print(data_loc("dataset1_truth.csv"))