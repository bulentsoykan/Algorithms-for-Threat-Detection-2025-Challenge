import os
from typing import Union

import numpy as np
import pandas as pd

import atd2025.io as aio
from atd2025.aisvis.base import AISColumns

StrPath = Union[str, os.PathLike[str]]


def evaluate_predictions(
    predictions_filepath: StrPath,
    ground_truth_filepath: StrPath,
    return_array: bool = False,
) -> Union[float, tuple[float, np.ndarray]]:  # type: ignore[type-arg]
    """Evaluate predictions in predictions_filepath against data in
    ground_truth_filepath.

    Args:
        predictions_filepath: path to .csv file of predictions as a string or
            PathLike.
        ground_truth_filepath: path to .csv file of ground truth AIS data to
            compare with predictions as a string or PathLike.
        return_array: An optional bool. Default is True. If True, will return the
            overall accuracy and a 2-D array showing the accuracy of each predicted
            track segment. If False, will return only the overall accuracy.

    Returns:
        float: Average accuracy per posit
        np.ndarray: Only if return_array is True. N x 3 array, showing for each
            posit whether the preceding and succeeding posits are correct.
    """
    ground_truth = aio.read_points(ground_truth_filepath)
    predictions = aio.read_predictions(predictions_filepath, ground_truth)

    result = per_posit_accuracy(
        aio.to_pandas(predictions),
        aio.to_pandas(ground_truth),
        return_array=return_array,
    )

    return result


def per_posit_accuracy(
    proposed_tracks: pd.DataFrame, true_tracks: pd.DataFrame, return_array: bool = False
) -> Union[float, tuple[float, np.ndarray]]:  # type: ignore[type-arg]
    """Fraction of correctly identified track segments.

    This function calculates accuracy of the track segments associated with each
    node. Since every posit X is part of a track (possibly the only posit in a track),
    we can assign each posit X an immediately preceding posit Y (the posit Y that
    occurs immediately before X on the same track) and an immediately succeeding
    posit Z (the posit Z that occurs next after X on the same track), allowing Y=None
    for posits X that are first on a track, and Z=None for posits X that are last on
    a track.

    Calling the pairs (Y,X)  and (X, Z) track segments, this function returns the
    fraction of correctly determined track segments in proposed_tracks.

    Args:
        proposed_tracks: Proposed tracks stored in pandas DataFrame format.
        true_tracks: True tracks stored in pandas DataFrame format.
        return_array: An optional bool. Default is True. If True, will return the
            overall accuracy and a 2-D array showing the accuracy of each predicted
            track segment. If False, will return only the overall accuracy.

    Returns:
        float: Average accuracy per posit
        np.ndarray: Only if return_array is True. N x 3 array, showing for each
            posit whether the preceding and succeeding posits are correct.

    """
    proposed_segments = _tracks_to_segments(proposed_tracks, force_str_type=True)
    true_segments = _tracks_to_segments(true_tracks, force_str_type=True)
    test_eq = proposed_segments == true_segments
    test_eq_sum = test_eq.sum(axis=1)
    if test_eq_sum[0] != proposed_segments.shape[1]:
        raise ValueError("proposed_tracks and true_tracks do not have matching pids.")
    metric = float(test_eq_sum[1:].sum() / (2 * true_segments.shape[1]))
    if return_array:
        return metric, test_eq.T
    else:
        return metric


def _tracks_to_segments(track_df: pd.DataFrame, force_str_type: bool = True) -> np.ndarray:  # type: ignore[type-arg]
    """Returns 2-D array showing the preceding and succeeding posit for each posit.

    This function looks at each posit and identifies its preceding posit on the same
    track, and each succeeding posit on the same track. If there is no preceding posit,
    meaning the posit is the first one on a track, that value is listed as None.
    Similary, if there is no succeeding posit, that value is listed as None. The
    returned array has rows ordered with pid increasing.

    Args:
        track_df: AIS Posits stored in pandas dataframe format.
        force_str_type: If True, forces output array to contain str values.
        Optional. Defaults to True.

    Returns:
        Nx3 numpy array, where N is the number of rows in track_df. First column has
        each pid from track_df. Each row is [pid, preceding pid or None, succeeding
        pi or None]. Rows ordered so that pid increases.

    """
    track_ids = np.unique(track_df[AISColumns.tid])
    all_posits = np.concatenate(
        [
            _single_track_to_segment(track_df[track_df[AISColumns.tid] == track_id])
            for track_id in track_ids
        ],
        axis=1,
    )
    #     sort all_posits by posit tid to make it easier for comparing other track
    #     splits
    sorted_by_pid = np.argsort(all_posits[0, :])
    result = all_posits[:, sorted_by_pid]
    if force_str_type:
        return result.astype(str)  # type: ignore[no-any-return]
    else:
        return result  # type: ignore[no-any-return]


def _single_track_to_segment(
    track: pd.DataFrame,
) -> np.ndarray:  # type: ignore[type-arg]
    """Returns representation of track segments for the given track.

    If (x_k, y_k, t_k) is a posit on the track, and (x_{k+1}, y_{k+1}, t_{k+1}) is the
    next posit on the track, then the pair corresponds to a track segment. A track of
    with M posits should have M-1 track segments.

    Returns: 3X(M-1) array A where
    A[i, :] = [posit, preceding posit, following posit]
    """
    sorted_track = track.sort_values(AISColumns.time)
    leading = [None] + list(sorted_track[AISColumns.pid])[:-1]
    trailing = list(sorted_track[AISColumns.pid])[1:] + [None]
    return np.array([sorted_track[AISColumns.pid].values, leading, trailing])
