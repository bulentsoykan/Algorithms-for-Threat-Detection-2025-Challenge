import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from ucf_atd_model.c20_consts import *
from ucf_atd_model.data import data_loc, new_data_loc
import os
import gc
from time import sleep
from multiprocessing.connection import Connection
from time import sleep
from typing import List

def run_collater(which: int, mybatch: List[int], connection: Connection):
    pwriter = pq.ParquetWriter(new_data_loc(f"c20_data/class20_{which}.parquet"), schema=schema)

    shouldBreak = False

    while True:
        if connection.poll():
            shouldBreak = True
        sleep(30)
        
        folder = data_loc("class20")
        filesToCompress = None
        try:
            filesToCompress = os.listdir(folder)
        except:
            continue
        xfiles = [x for x in filesToCompress if "xdata" in x]
        yfiles = [x for x in filesToCompress if "ydata" in x]

        # Ensure both files are present
        xnums = [int(x.split("_")[1].removesuffix(".csv")) for x in xfiles]
        ynums = [int(x.split("_")[1].removesuffix(".npy")) for x in yfiles]

        nums_to_process = [x for x in ynums if x in xnums]

        nums_to_process = [x for x in nums_to_process if x in mybatch]

        # Wait for any file writes to finish
        sleep(60)

        print(f"Found {len(nums_to_process)} files to process")
        for i in nums_to_process:
            xdata_path = os.path.join(folder, f"xdata_{i}.csv")
            ydata_path = os.path.join(folder, f"ydata_{i}.npy")
            
            # Read in data
            xdata = pd.read_csv(xdata_path)
            if "Unnamed: 0" in xdata.keys():
                xdata = xdata.drop(["Unnamed: 0"], axis=1)
            ydata = np.load(ydata_path)

            for j, new_yname in enumerate(ynames):
                xdata[new_yname] = ydata[:, j]        

            xdata = pd.DataFrame(xdata)

            # Convert to parquet recordbatch
            batch = pa.RecordBatch.from_pandas(xdata, schema = schema, preserve_index = False)
            pwriter.write_batch(batch)

            # Remove files from disk
            try:
                os.remove(xdata_path)
                os.remove(ydata_path)
            except:
                continue
        print(f"Finished processing {len(nums_to_process)} files.")
        gc.collect()
        if shouldBreak:
            break
    
    pwriter.close()