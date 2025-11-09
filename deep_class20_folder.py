import pandas as pd
import torch as pt
import torch.nn as nn
from time import time
import pyarrow.parquet as pq
import itertools as it
import gc
import tqdm
import numpy as np
from pickle import dump
import os

from ucf_atd_model.c20_consts import *
import ucf_atd_model.c20_consts as const
from ucf_atd_model.data import data_loc, ResultCache
from ucf_atd_model.datasets.pt_datasets import C20data


def main():
    badnames = [x for x in full_names if x.endswith("_16")]

    ynames = [x for x in const.ynames if not x.endswith("_16")]

    data_files = [os.path.join(data_loc("c20_data"), x) for x in os.listdir(data_loc("c20_data"))]
    validation_file = data_files[0]
    validation = [0, 1, 2]

    folder = "checkpoints/small_deep20"

    inp_dim = len(colnames) - len(badnames) + 1
    print(inp_dim)
    h_dim = 3000
    out_dim = n_norm_classes + 1 - 1

    device = pt.device("cuda:0")

    model = nn.Sequential(
        nn.Linear(inp_dim, h_dim),
        nn.ReLU(),
        nn.Linear(h_dim, h_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(h_dim, h_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.5),
        # nn.Linear(h_dim // 2, h_dim // 2),
        # nn.ReLU(),
        # nn.Dropout(0.2),
        nn.Linear(h_dim // 2, out_dim),
    ).to(device)


    # Read the validation data in
    X_test = None
    y_test = None
    y_test_mask = None
    xmean = None
    xstd = None

    with pq.ParquetFile(validation_file) as fulldata:
        testData = fulldata.read_row_groups(validation).to_pandas()
        X_test = pt.from_numpy(testData.drop(ynames + badnames, axis=1).to_numpy()).float()
        y_test = pt.from_numpy(testData[ynames].to_numpy()).float()
        y_test_mask = pt.zeros_like(y_test, dtype=pt.float32)
        y_test_mask[:, -1] = 0
        y_test_mask[:, 0] = 0

        num_ft_sets = n_norm_classes - 1
        for i in range(1, num_ft_sets):
            ft_names = getNormFeatures(i)
            feats = testData[ft_names]
            y_test_mask[:, i] = pt.from_numpy(((feats == -1).all(axis=1) * -1e8).to_numpy()).float()
        
        xmean = pt.mean(X_test, 0)
        xstd = pt.std(X_test, 0)
        pt.save(xmean, "xmean.pt")
        pt.save(xstd, "xstd.pt")

        X_test = (X_test - xmean) / xstd


    num_epochs = 1000
    num_batches = 80

    # model.load_state_dict(pt.load(f"{folder}/epoch_38.pt"))

    optimizer = pt.optim.AdamW(model.parameters(), lr=0.000002, weight_decay=0.002)
    # optimizer.load_state_dict(pt.load(f"{folder}/optim_38.pt"))

    scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler.load_state_dict(pt.load(f"{folder}/sched_38.pt"))
    # scheduler.step()
    start_epoch = 39

    lossfn = nn.CrossEntropyLoss(reduction="sum")

    dataset = C20data(validation, validation_file, data_files, badnames, ynames, xstd, xmean, num_batches)
    dataloader = pt.utils.data.DataLoader(dataset, num_workers=5, prefetch_factor=300, pin_memory=True)

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch} / {num_epochs}")
        startTime = time()
        
        epoch_train_loss = 0

        # Evaluate model
        model.eval()
        with pt.no_grad():
            tot_loss = 0
            n_loss = 0
            for xb, yb, ym in dataset.rebatch(X_test, y_test, y_test_mask):
                xb = xb.to(device)
                yb = yb.to(device)
                ym = ym.to(device)
                output = model(xb)
                output = output + ym
                loss = lossfn(output, yb)
                n_loss += xb.shape[0]
                
                tot_loss += loss.item()
                
            print(f"    Test Loss: {tot_loss / n_loss:.3f}")
        

        # Train the model
        model.train()
        for X_train, y_train, y_train_mask in tqdm.tqdm(dataloader, total=dataset.num_loops):
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            y_train_mask = y_train_mask.to(device)
            optimizer.zero_grad()
            
            output = None
            loss = None
            with pt.set_grad_enabled(True):
                output = model(X_train)
                output = output + y_train_mask
                loss = lossfn(output, y_train)
                loss.backward()
                optimizer.step()
                # del output

            epoch_train_loss += loss.item()
        
        print(f"    Train Loss: {epoch_train_loss:.2f}")
        endTime = time()
        print(f"    Time Elapsed: {(endTime - startTime):.3f} seconds")

        if epoch % 10 == 0:
            pt.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pt")
            pt.save(optimizer.state_dict(), f"checkpoints/optim_{epoch}.pt")
            pt.save(scheduler.state_dict(), f"checkpoints/sched_{epoch}.pt")
        scheduler.step()

if __name__ == "__main__":
    print("Running")
    main()