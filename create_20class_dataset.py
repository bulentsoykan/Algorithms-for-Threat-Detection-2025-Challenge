import pyarrow.parquet as pq
from multiprocessing import Pool
from ucf_atd_model.data import data_loc
from ucf_atd_model.datasets.create_20class_data import run
from sys import argv
import itertools as it
from collate_20class_dataset import run_collater

from multiprocessing import Process, Pipe

if __name__ == "__main__":
    rowgroups = None
    if len(argv) < 3:
        print("Missing argument 2 and 3.")
        exit()

    batch_num = int(argv[1])
    num_batches = int(argv[2])

    # Read number of row groups
    with pq.ParquetFile(data_loc("historical.parquet")) as fulldata:
        rowgroups = fulldata.num_row_groups

    batches = list(it.batched(range(rowgroups), (rowgroups // num_batches) + 1))
    mybatch = batches[batch_num]

    pconn, cconn = Pipe()
    collater = Process(target=run_collater, args=(batch_num, mybatch, cconn))
    collater.start()

    with Pool(processes=34) as pool:
        results = pool.map(run, mybatch)
        for i, result in enumerate(results):
            if result != "":
                print(i)
                print(result)
                print()

    # Tell collater to finish up                
    pconn.send({"message": "end"})
    collater.join()
    pconn.close()
    cconn.close()