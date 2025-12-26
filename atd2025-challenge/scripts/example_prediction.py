"""
example_prediction.py

This script demonstrates how you can the atd2025 package to participate in the
challenge. It shows how to read in a .csv with unlabeled AIS data, how to run one of
the provided baseline algorithms, how to save predictions in the format we require,
how you can plot tracks, and how we evaluate predictions against ground truth.
"""

from __future__ import annotations

import time
from pathlib import Path

import atd2025

# ************************************************
# All filepaths used in this script. CHANGE AS NEEDED.

# YOU NEED TO PROVIDE THIS FILE: it's the unlabeled data
UNLABELED_DATA_FILEPATH = atd2025.EXAMPLE_UNLABELLED
# UNLABELED_DATA_FILEPATH = Path(r"/path/to/your/unlabelled_data.csv")  # Linux Path
# UNLABELED_DATA_FILEPATH = Path(r"C:\path\to\your\unlabelled_data.csv")  # Windows Path

# YOU NEED TO PROVIDE THIS FILE: it's the labeled data
LABELED_DATA_FILEPATH = atd2025.EXAMPLE_LABELLED
# LABELED_DATA_FILEPATH = Path(r"/path/to/your/labelled_data.csv")  # Linux Path
# LABELED_DATA_FILEPATH = Path(r"C:\path\to\your\labelled_data.csv")  # Windows Path

# Provide filepath for where you want to save your predictions
PREDICTIONS_FILEPATH = Path("predictions.csv")

# Provide a filepath for where you want to save a plot of your predictions.
PREDICTIONS_PLOT_FILEPATH = Path("predictions_plot.html")

# *************************************************
# SELECT ONE ALGORITHM BELOW

# ALGORITHM = atd2025.baseline # Slow but more accurate than the naive examples.
# ALGORITHM = atd2025.assign_most_similar_speed_in_foc # Most accurate naive method.
# ALGORITHM = atd2025.assign_to_one_track # Assigns all posits to a single track.
# ALGORITHM = atd2025.assign_to_unique_track #Assigns each posit to its own track.
ALGORITHM = atd2025.assign_randomly  # Random assignments to tracks.


# *************************************************
# Example of how to load unlabeled data, predict the tracks, and save your predictions.

# Load unlabeled AIS data as a list of posits (ship observations)
posits = atd2025.read_points(UNLABELED_DATA_FILEPATH)

# Make predictions using a baseline algorithm

start_time = time.time()
predictions = ALGORITHM(posits)
print(f"Time to calculate: {time.time() - start_time} seconds")

# Save predictions to a file. You will submit that file for evaluation.
atd2025.predictions_to_csv(PREDICTIONS_FILEPATH, predictions)

# **********************************
# Example of how to plot your predictions and save to an .html
prediction_figure = atd2025.aisvis.show_tracks.make_line_plot(
    atd2025.to_pandas(predictions), show_lines=False, show_velocity=True
)
prediction_figure.write_html(PREDICTIONS_PLOT_FILEPATH)

# ******************************************
# This shows how we will evaluate your predictions

# We will load your predictions file and join it with our labeled data.
score = atd2025.evaluate_predictions(PREDICTIONS_FILEPATH, LABELED_DATA_FILEPATH)
print(f"Average accuracy per node: {score}")
