# Metrics Evaluation

This script shows how we will evaluate submitted predictions. We will load the
predictions file, combine it with our ground truth file, and calculate average
accuracy per posit. To run the script, copy and paste it into a local file, then 
type the following in a terminal or console:

```console
python /path/to/script
```

The script uses data installed with the package. To run it on 
other data, just change the file paths as explained in the code comments.

```python title="evaluate_predictions.py"
--8<-- "scripts/load_predictions_and_evaluate.py"
```