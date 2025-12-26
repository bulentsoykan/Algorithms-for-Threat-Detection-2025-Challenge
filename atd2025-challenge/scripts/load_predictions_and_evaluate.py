"""
load_predictions_and_evaluate.py

This script shows how we will evaluate submitted predictions. We will load the
predictions file, combine it with our ground truth file, and calculate average
accuracy per posit.
"""

from __future__ import annotations

from pathlib import Path  # noqa

import atd2025

# ************************************************
# All filepaths used in this script. CHANGE AS NEEDED.

# YOU NEED TO PROVIDE THIS FILE: it's the labeled data.
LABELED_DATA_FILEPATH = atd2025.EXAMPLE_LABELLED
# LABELED_DATA_FILEPATH = Path(r"/path/to/your/labelled_data.csv")

# YOU NEED TO PROVIDE THIS FILE: it's a set of predictions in the required format.
PREDICTIONS_FILEPATH = atd2025.EXAMPLE_PREDICTIONS
# PREDICTIONS_FILEPATH = Path(r"/path/to/your/predictions.csv")

# ************************************************
# Load and evaluate predictions.

score = atd2025.evaluate_predictions(PREDICTIONS_FILEPATH, LABELED_DATA_FILEPATH)
print(f"Average accuracy per posit: {score}")
