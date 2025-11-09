## UCF ATD 2025 Experiments
We briefly describe what each file here does.

1. cbtr_closest: Output from and code needed to run memory intensive part of the CBTR algorithm.
2. checkpoints: Trained classification model
3. clustering_output: Output from unsupservised clustering models
4. oracle_out: Oracle output for various screen sizes k.
5. ucf_atd_model: A collection of helper functions, mostly needed to run our final hybrid model.
6. collate_20class_dataset.py: Defines a process that is used to collate multiple days of the ATD2025 dataset into 1 large parquet file, intended to be run on something like an NFS, triggered by create_20class_dataset.py
7. create_20class_dataset.py: Runs many processes (on potentially multiple nodes) that read in the historical preprocessed dataset and generate the 17 class supervised classification dataset. Historical data available here: https://figshare.com/articles/dataset/Preprocessed_AIS_Dataset/29975875, must be placed in the ucf_atd_model/datasets folder.
8. deep_class20_folder.py: Runs the training process for the hybrid model on the folder of data created by create_20class_dataset.py
9. dist_matrix_clustering.ipynb: Code for creating the CBTR distance matrix without ellipsoidal gating, and running various clustering algorithms on it. CBTR itself does usee ellipsoidal gating, which is controlled in cbtr_closest.
10. kalman_const_turn.ipynb: Code for Kalman filter with constant turn rate, custom pytorch implementation for speed
11. kalman_const_velocity.ipynb: Code for Kalman filter with constant velocity, custom pytorch implementation for speed
12. kalman_half_maneuver.ipynb: Code for Adaptive Half-Maneuver Kalman filter with constant velocity, custom pytorch implementation for speed
13. ml_out_ds1.csv: Output from our hybrid model on dataset1 (same dataset as used everywhere else)
14. oracle_analysis.ipynb: Graphs oracle accuracy for various screen sizes
15. oracle.ipynb: Generates oracle output for various screen sizes
16. xmean/xstd: Normalizing constants needed to run the hybrid model

If you clone this repo, make sure to switch to this branch via `git checkout export_book`. The main branch is much messier and contains many files needed to run things via SLURM, and some unused models that are not necessary here.

To run grading code, the ATD2025 package must be installed, which is available here: https://gitlab.com/algorithms-for-threat-detection/2025/atd2025.
