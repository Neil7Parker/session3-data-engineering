"""Load the artifacts in a fresh process without training a model."""
from pathlib import Path
import json
import pickle
import sys

import joblib
import numpy as np
import sklearn

from grade_tracker import GradeTracker


def main():
    root = Path(__file__).resolve().parent

    def load(name):
        with (root / name).open("rb") as stream:
            return pickle.load(stream)

    fruits = load("fruits.pkl")
    scores = load("scores.pkl")
    complex_data = load("complex_data.pkl")
    original = load("ml_course_grades.pkl")
    updated = load("ml_course_grades_updated.pkl")
    model = load("study_hours_model.pkl")
    joblib_model = joblib.load(root / "study_hours_model.joblib")
    assert fruits == ["apple", "banana", "mango", "strawberry"]
    assert scores == {"Alice": 92, "Bob": 85, "Charlie": 78, "Diana": 95}
    assert complex_data == {
        "fruits": fruits, "scores": scores,
        "metrics": [1.2, 3.4, 5.6, 7.8], "active": True,
    }
    assert isinstance(original, GradeTracker)
    assert isinstance(updated, GradeTracker)
    assert len(original.student_grades) == 5
    assert len(updated.student_grades) == 8
    assert original.get_top_student() == updated.get_top_student() == ("Alice", 95)
    assert original.get_average() == 87
    assert updated.get_average() == 86.875
    hours = np.array([[5], [10], [15], [17.5], [20]])
    predictions = model.predict(hours)
    np.testing.assert_array_equal(predictions, joblib_model.predict(hours))
    assert np.isfinite(predictions).all()
    assert 0 < model.coef_[0, 0] < 5
    assert (root / "study_hours_model.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    result = {
        "status": "passed",
        "verification": "fresh-process loading; no training performed by this script",
        "data": "synthetic study-hours data from the course example; not real student records",
        "python": sys.version.split()[0],
        "scikit_learn": sklearn.__version__,
        "original_grade_count": len(original.student_grades),
        "updated_grade_count": len(updated.student_grades),
        "original_average": original.get_average(),
        "updated_average": updated.get_average(),
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
        "hours": hours.ravel().tolist(),
        "predicted_scores": predictions.ravel().tolist(),
        "pickle_joblib_predictions_equal": True,
    }
    (root / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
