# Pickling in Python: Saving and Loading Data and Models

Course practice covering lists, dictionaries, a custom grade tracker, a scikit-learn linear regression model, and joblib serialization.

## Run

Use Python 3.12 and the versions in `requirements.txt` to reproduce the included binary artifacts.

```bash
python -m pip install -r requirements.txt
python run_examples.py
python load_saved.py
```

`run_examples.py` runs all seven Python code blocks from the course exercise in order. The `GradeTracker` class is placed in `grade_tracker.py` so that objects can be loaded in a separate process. Matplotlib uses the noninteractive Agg backend. Outputs are written next to the scripts regardless of the current working directory.

`load_saved.py` independently loads the saved objects, checks the original and updated grades, makes predictions without retraining, compares pickle and joblib predictions, and writes `verification.json`.

## Artifacts

| File | Contents |
| --- | --- |
| `fruits.pkl` | List of favorite fruits |
| `scores.pkl` | Dictionary of example student scores |
| `complex_data.pkl` | Nested data containing lists, a dictionary, and a Boolean |
| `ml_course_grades.pkl` | Original GradeTracker with five students |
| `ml_course_grades_updated.pkl` | Updated GradeTracker with eight students |
| `study_hours_model.pkl` | Fitted linear regression model saved with pickle |
| `study_hours_model.joblib` | The same fitted model saved with joblib |
| `study_hours_model.png` | Original course example plot |
| `run_output.txt` | Actual output of running the examples |
| `verification.json` | Actual independent reload verification |

The model uses the exercise's **synthetic** data with seed 42. Reported MSE and R-squared are calculated on the training data, as in the original example; they are not held-out evaluation results. The example predictions are not validated educational advice.

Only load pickle and joblib files from trusted sources. Both formats can execute code during loading. The custom class module and compatible dependency versions are required when loading saved objects.

## Course source and delivery

[Original course exercise](https://gitee.com/lundechen/data_engineering_master_2026_fall/blob/master/session-3/practice-pickling-of-a-model.md)

The course asks for code and pickled files in a GitHub repository, followed by submission of that repository URL to the WeChat group. This local folder is prepared for that upload. A local run alone does not complete the GitHub submission.
