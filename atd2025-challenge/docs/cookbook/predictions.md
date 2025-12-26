# Making Predictions

This script demonstrates how you can the atd2025 package to participate in the
challenge. It shows how to read in a .csv with unlabeled AIS data, how to run one of
the provided baseline algorithms, how to save predictions in the format we require,
how you can plot tracks, and how we evaluate predictions against ground truth.

To run the script, copy and paste it into a local file, then 
type the following in a terminal or console:
```console
python /path/to/script
```
As currently written, the script uses data loaded from the installed package. If 
you want to try it on different data, 
just change the paths as explained in the code's comments.


```python title="making_predictions.py"
--8<-- "scripts/example_prediction.py"
```