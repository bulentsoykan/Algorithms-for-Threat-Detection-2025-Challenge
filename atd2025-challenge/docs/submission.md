# Submitting Predictions

## Formatting Predictions

Your predictions should be in a CSV format with two columns. Each row has:

* `point_id` --- one of the posit IDs provided in the dataset. Every row should have a unique 
  point_id.
* `track_id` --- the ID you choose for the track that you believe `point_id` lies on

This formatting process can be done with the `atd2025` package.

Let's say you have the following data:

| `point_id` | `time`              | `lat`      | `lon`       | `speed` | `course` | 
|------------|---------------------|------------|-------------|---------|----------|
| 0          | 2024-01-01T23:58:01 | 30.0518251 | -90.5403474 | 0       | 187.4    |
| 1          | 2024-01-01T06:36:24 | 29.8305799 | -91.9901608 | 0       | 234.7    |
| 2          | 2024-01-01T00:00:01 | 29.3029124 | -92.6439051 | 2.8     | 192.3    |
| 3          | 2024-01-01T00:00:28 | 29.2822950 | -92.6494236 | 2.7     | 202.2    |

Then, you make your predictions in a new column called `track_id`:

| `point_id` | `time`              | `lat`      | `lon`       | `speed` | `course` | `track_id` |
|------------|---------------------|------------|-------------|---------|----------|------------|
| 0          | 2024-01-01T23:58:01 | 30.0518251 | -90.5403474 | 0       | 187.4    | 1          |
| 1          | 2024-01-01T06:36:24 | 29.8305799 | -91.9901608 | 0       | 234.7    | 2          |
| 2          | 2024-01-01T00:00:01 | 29.3029124 | -92.6439051 | 2.8     | 192.3    | 3          |
| 3          | 2024-01-01T00:00:28 | 29.2822950 | -92.6494236 | 2.7     | 202.2    | 3          |

In a `pandas` DataFrame format, you can convert your predictions into a csv with the
`io.predictions_to_csv` command:

```python
import atd2025
import atd2025.io

points_pred = atd2025.io.to_points(pred_df)  # Here, pred_df is the DataFrame of your predictions
output_path = "path/to/store/your/predictions.csv"

atd2025.io.predictions_to_csv(output_path, points_pred)
```

The output CSV would look like this:

```
point_id, track_id
0,1
1,2
2,3
3,3
```

## Submitting Predictions

CSV files of your predictions should be sent over email to
both [wsk5068@psu.edu](mailto:wsk5068@psu.edu)
and [bgc5088@psu.edu](mailto:bgc5088@psu.edu) **before 8am EDT** for each submission
deadline.