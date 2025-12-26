import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import atd2025
from atd2025.io import AISColumns

SHARED_DATADIR = Path(__file__).parent / "data"


def data_filepath(filename: str) -> str:
    return os.path.join(SHARED_DATADIR, filename)


@pytest.fixture
def ground_truth() -> pd.DataFrame:
    return atd2025.read_points_as_df(data_filepath("single_track_example_with_ID.csv"))


test_files = [f"single_track_predictions0{n}.csv" for n in range(1, 5)]
correct_accuracy_per_node = [0.979591837, 0.020408163, 0.673469388, 1]


@pytest.mark.parametrize(
    "testdata_file, ans",
    [(f, a) for f, a in zip(test_files, correct_accuracy_per_node)],
)
def test_per_node_accuracy(testdata_file: str, ans: float, ground_truth: pd.DataFrame) -> None:
    predictions = atd2025.to_pandas(
        atd2025.read_predictions(data_filepath(testdata_file), atd2025.to_points(ground_truth))
    )
    assert np.isclose(
        atd2025.per_posit_accuracy(predictions, ground_truth),  # type: ignore[arg-type]
        ans,  # type: ignore[arg-type]
    )


def test_per_node_accuracy_array_return(ground_truth: pd.DataFrame) -> None:
    testdata_file = "single_track_predictions01.csv"
    predictions = atd2025.to_pandas(
        atd2025.read_predictions(data_filepath(testdata_file), atd2025.to_points(ground_truth))
    )
    _, all_posits = atd2025.per_posit_accuracy(predictions, ground_truth, return_array=True)  # type: ignore[misc]
    correct_ans_col_1 = 49 * [True]
    correct_ans_col_2 = 24 * [True] + [False] + 24 * [True]
    correct_ans_col_3 = 23 * [True] + [False] + 25 * [True]
    correct_ans = np.array([correct_ans_col_1, correct_ans_col_2, correct_ans_col_3]).T
    # Next, sort correct_ans rows by pid
    pids = ground_truth[AISColumns.pid].values
    pid_order = np.argsort(pids)  # type: ignore[arg-type]
    correct_ans = correct_ans[pid_order, :]

    assert np.allclose(all_posits, correct_ans)
