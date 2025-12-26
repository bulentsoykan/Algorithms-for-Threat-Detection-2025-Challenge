from pathlib import Path

import pandas as pd

from atd2025 import data as ais

##### Data arguments #####

# Which date to collect and process data for.
DATE = "01/01/2023"

# Where to store intermediate data files.
DATA_LOCATION = Path("data/")


##### Data region arguments #####

# Which region to restrict posits to.
REGION = "GC"

# Whether rows with invalid speed or course should be removed.
TRIM_INVALID = True

# When true, keep all posits of tracks that have at least one posit within the defined region,
# allowing tracks to extend out of the defined region.
# When false, only keep posits within the defined region, and remove all others.
AVOID_SPLITTING_TRACKS = True

# The minimum amount of time apart posits should be, when possible.
TIME_DELTA = pd.Timedelta("0.5h")


##### Trimming arguments #####

# Whether to keep the first stationary point from a consecutive group of stationary points.
KEEP_FIRST = True

# Whether to keep the last stationary point from a consecutive group of stationary points.
KEEP_LAST = True

# Whether to trim the beginning and end of the dataframe such that the dataframe
# starts and ends with a non-stationary point.
REMOVE_EDGE = False


##### Error arguments #####

# The covariance for the gausian distribution to apply error to posits spatially.
COV_MATRIX = ((0.0000001, 0.0000001), (0.0000001, 0.0000009))

# Range for how many seconds should the time error for each posit should vary by.
SECONDS = 3


##### Output arguments #####

# Where to output the finalized csvs.
OUT_LOCATION = Path("out/")

# Name for the output csvs.
FILENAME = "output_data.csv"


##### Process data #####

print("This may take several minutes...")
date = ais.validate_date(DATE)
csv_location = ais.query_ais(date, DATA_LOCATION)

df = ais.ais_to_df(csv_location, REGION, TRIM_INVALID, AVOID_SPLITTING_TRACKS)
df = ais.subsample_ais_df(df, delta=TIME_DELTA)
df = ais.trim_stationary(df, KEEP_FIRST, KEEP_LAST, REMOVE_EDGE)
df = ais.apply_spatial_error(df, COV_MATRIX)
df = ais.apply_temporal_error(df, SECONDS)
ais.df_to_csv(df, OUT_LOCATION, FILENAME)
