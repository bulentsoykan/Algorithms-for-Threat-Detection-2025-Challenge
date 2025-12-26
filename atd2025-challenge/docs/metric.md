# Evaluation of Predictions

To evaluate predictions, we will look at every posit `X` and check whether the
prediction has the correct preceding posit `Y` (the posit that occurred immediately
prior to `X` on the
same track) and the correct succeeding posit `Z` (the posit that occurred immediately
after `X` on the same track). This includes checking whether the
first posit on each track has no preceding posit, and whether the last posit on each
track has no succeeding posit. We will give each posit an accuracy value
between 0 and 1. A value of 0 means the predicted preceding and succeeding posits
are wrong. A value of 0.5 means exactly one of the predicted preceding and
succeeding posits is correct, and an accuracy of 1 means the predicted
preceding and succeeding posits are both correct. Our evaluation metric for a full
set of posits is the average of those values over all posits. See below for an example.

## Example

We’ll use the notation `X` &rarr; `Y` to mean that `X` and `Y` are on the same track,
with `Y`
being the next posit to occur after `X` on that track.
Suppose we have 8 posits that lie on 2 tracks:

- Track A: 1 &rarr; 2 &rarr; 4 &rarr; 6
- Track B: 3 &rarr; 5 &rarr; 7 &rarr; 8

Our prediction for the same set of posits has them lying on 3 tracks that look like
this:

- Predicted Track C: 1 &rarr; 2 &rarr; 4
- Predicted Track D: 3 &rarr; 6 &rarr; 8
- Predicted Track E: 5 &rarr; 7

For each posit, we compare the predicted preceding posit with the true preceding
posit, then do the same with succeeding posits.

### Ground Truth

This table shows all the correct preceding and succeeding posits, including
appropriate values of None for each track's first and last posits.

| Posit | Preceding Posit | Succeeding Posit |
|-------|-----------------|-----------------|
| 1     | None            | 2               |
| 2     | 1               | 4               |
| 3     | None            | 5               |
| 4     | 2               | 6               |
| 5     | 3               | 7               |
| 6     | 4               | None            |
| 7     | 5               | 8               |
| 8     | 7               | None            |

### Prediction

This table shows all the *predicted* preceding and succeeding posits, including
appropriate values of None for each *predicted* track's first and last posits.

| Posit | Preceding Posit | Succeeding Posit |
|-------|-----------------|-----------------|
| 1     | None            | 2               |
| 2     | 1               | 4               |
| 3     | None            | 6               |
| 4     | 2               | None            |
| 5     | None            | 7               |
| 6     | 3               | 8               |
| 7     | 5               | None            |
| 8     | 3               | None            |

### Evaluation of Prediction

This shows how we can use the above tables to calculate an accuracy for each posit.
We use the average of those values for the final scoring metric.

| Posit | Preceding Posit is Correct | Succeeding Posit is Correct | Posit Accuracy |
|-------|----------------------------|----------------------------|----------------|
| 1     | Yes                        | Yes                        | 1.0            |
| 2     | Yes                        | Yes                        | 1.0            |
| 3     | Yes                        | No                         | 0.5            |
| 4     | Yes                        | No                         | 0.5            |
| 5     | No                         | Yes                        | 0.5            |
| 6     | No                         | No                         | 0.0            |
| 7     | Yes                        | No                         | 0.5            |
| 8     | No                         | Yes                        | 0.5            |

For this example the average posit accuracy is `4.5/8 = 0.5625`.