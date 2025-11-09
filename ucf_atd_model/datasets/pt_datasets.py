import pandas as pd
import torch as pt
import pyarrow.parquet as pq
import itertools as it
import numpy as np

from ..c20_consts import *


class C20data(pt.utils.data.IterableDataset):
    def __init__(self, validation, validation_file, data_files, badnames, ynames, xstd, xmean, batches_per_day):
        super(C20data).__init__()
        self.validation_file = validation_file
        self.validation = validation

        # Figure out how many rowgroups there are overall
        self.num_loops = 0
        for data_file in data_files:
            with pq.ParquetFile(data_file) as fulldata:
                self.num_loops += len(list(range(fulldata.num_row_groups))) * batches_per_day
        self.num_loops -= len(validation) * batches_per_day
        self.data_files = data_files
        self.badnames = badnames
        self.ynames = ynames
        self.xstd = xstd
        self.xmean = xmean
        self.batches_per_day = batches_per_day

    # Return batches that fit in GPU memory
    def rebatch(self, x, y, y_mask):
        """
        Return data in batches that fit in GPU memory
        
        Parameters
        ----------
            x (pt.tensor): x data
                
            y (pt.tensor): y data

            y_mask (pt.tensor): y mask
            
            n (int): Number of batches to divide x/y into
        """
        currPlace = 0
        batchlen = (x.shape[0] // self.batches_per_day) + 1
        while currPlace < x.shape[0]:
            nextPlace = currPlace + batchlen
            yield x[currPlace:nextPlace], y[currPlace:nextPlace], y_mask[currPlace:nextPlace]
            currPlace = nextPlace

    def gen(self, num_workers = None, worker_id = None):
        if num_workers is None or (num_workers == 1):
            for data_file in self.data_files:
                with pq.ParquetFile(data_file) as fulldata:
                    n_rowgroups = fulldata.num_row_groups
                    all_data = set(range(n_rowgroups))
                    if data_file == self.validation_file:
                        all_data = all_data.difference(self.validation)
                    
                    all_data = sorted(list(all_data))

                    for idx in all_data:
                        table: pd.DataFrame = fulldata.read_row_group(idx).to_pandas(self_destruct = True).replace([np.inf, -np.inf], np.nan).dropna()

                        train_X = pt.from_numpy(table.drop(self.ynames + self.badnames, axis=1).to_numpy()).float()
                        train_y = pt.from_numpy(table[self.ynames].to_numpy()).float()
                        
                        train_y_mask = pt.zeros_like(train_y, dtype=pt.float32)
                        train_y_mask[:, -1] = 0
                        train_y_mask[:, 0] = 0

                        num_ft_sets = n_norm_classes - 1
                        for i in range(1, num_ft_sets):
                            ft_names = getNormFeatures(i)
                            feats = table[ft_names]
                            train_y_mask[:, i] = pt.from_numpy(((feats == -1).all(axis=1) * -1e8).to_numpy()).float()

                        train_X = (train_X - self.xmean) / self.xstd

                        for stuff in self.rebatch(train_X, train_y, train_y_mask):
                            yield stuff

        else:
            for data_file in self.data_files:
                with pq.ParquetFile(data_file) as fulldata:
                    n_rowgroups = fulldata.num_row_groups
                    all_data = set(range(n_rowgroups))
                    if data_file == self.validation_file:
                        all_data = all_data.difference(self.validation)
                    
                    all_data = sorted(list(all_data))
                    batches = list(it.batched(all_data, num_workers))
                    mybatch = [x[worker_id] for x in batches if worker_id < len(x)]

                    for idx in mybatch:
                        table: pd.DataFrame = fulldata.read_row_group(idx).to_pandas(self_destruct = True).replace([np.inf, -np.inf], np.nan).dropna()

                        train_X = pt.from_numpy(table.drop(self.ynames + self.badnames, axis=1).to_numpy()).float()
                        train_y = pt.from_numpy(table[self.ynames].to_numpy()).float()
                        train_y_mask = pt.zeros_like(train_y, dtype=pt.float32)
                        train_y_mask[:, -1] = 0
                        train_y_mask[:, 0] = 0

                        num_ft_sets = n_norm_classes - 1
                        for i in range(1, num_ft_sets):
                            ft_names = getNormFeatures(i)
                            feats = table[ft_names]
                            train_y_mask[:, i] = pt.from_numpy(((feats == -1).all(axis=1) * -1e8).to_numpy()).float()

                        train_X = (train_X - self.xmean) / self.xstd
                        
                        for stuff in self.rebatch(train_X, train_y, train_y_mask):
                            yield stuff

    def __iter__(self):
        worker_info = pt.utils.data.get_worker_info()
        if worker_info is None:
            return self.gen()
        else:
            return self.gen(num_workers=worker_info.num_workers, worker_id=worker_info.id)