import os
import pickle
import pandas as pd
from flask import Flask, render_template, request

class PredictorException(Exception):
    pass

class ModelLoadException(PredictorException):
    pass

class InputValidationException(PredictorException):
    pass

class StudentPerformancePredictor:
    def __init__(self, model_path="model.pkl"):
        self.model_path = model_path
        self.model = None
        self.feature_columns = [
            "Hours Studied",
            "Previous Scores",
            "Extracurricular Activities",
            "Sleep Hours",
            "Sample Question Papers Practiced"
        ]
        self.load_model()

    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file '{self.model_path}' not found.")
            with open(self.model_path, "rb") as model_file:
                self.model = pickle.load(model_file)
        except Exception as exc:
            raise ModelLoadException(f"Failed to load model: {exc}")

    def extract_and_validate(self, form_data):
        try:
            raw_hours = form_data.get("hours_studied")
            raw_scores = form_data.get("previous_scores")
            raw_extra = form_data.get("extracurricular_activities")
            raw_sleep = form_data.get("sleep_hours")
            raw_papers = form_data.get("sample_papers")

            if None in (raw_hours, raw_scores, raw_extra, raw_sleep, raw_papers):
                raise ValueError("All fields are required.")

            hours_studied = float(raw_hours)
            previous_scores = float(raw_scores)
            sleep_hours = float(raw_sleep)
            sample_papers = float(raw_papers)

            if str(raw_extra).strip().lower() in ["1", "yes"]:
                extracurricular_val = 1
            elif str(raw_extra).strip().lower() in ["0", "no"]:
                extracurricular_val = 0
            else:
                extracurricular_val = int(raw_extra)

            if hours_studied < 0 or hours_studied > 24:
                raise ValueError("Hours Studied must be between 0 and 24.")
            if previous_scores < 0 or previous_scores > 100:
                raise ValueError("Previous Scores must be between 0 and 100.")
            if sleep_hours < 0 or sleep_hours > 24:
                raise ValueError("Sleep Hours must be between 0 and 24.")
            if sample_papers < 0:
                raise ValueError("Sample Papers Practiced cannot be negative.")

            input_data = pd.DataFrame(
                [[hours_studied, previous_scores, extracurricular_val, sleep_hours, sample_papers]],
                columns=self.feature_columns
            )
            return input_data

        except (ValueError, TypeError) as exc:
            raise InputValidationException(str(exc))
        except Exception as exc:
            raise InputValidationException(f"Invalid input: {exc}")

    def predict(self, feature_dataframe):
        try:
            if self.model is None:
                self.load_model()
            prediction_result = self.model.predict(feature_dataframe)[0]
            return round(float(prediction_result), 2)
        except Exception as exc:
            raise PredictorException(f"Prediction failed: {exc}")

app = Flask(__name__)
predictor = StudentPerformancePredictor("model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    error_text = None

    if request.method == "POST":
        try:
            input_df = predictor.extract_and_validate(request.form)
            result = predictor.predict(input_df)
            prediction_text = f"{result}"
        except PredictorException as exc:
            error_text = str(exc)
        except Exception as exc:
            error_text = f"Unexpected error: {exc}"

    return render_template(
        "index.html",
        prediction_text=prediction_text,
        error_text=error_text
    )

@app.route("/predict", methods=["POST"])
def predict():
    prediction_text = None
    error_text = None

    try:
        input_df = predictor.extract_and_validate(request.form)
        result = predictor.predict(input_df)
        prediction_text = f"{result}"
    except PredictorException as exc:
        error_text = str(exc)
    except Exception as exc:
        error_text = f"Unexpected error: {exc}"

    return render_template(
        "index.html",
        prediction_text=prediction_text,
        error_text=error_text
    )

if __name__ == "__main__":
    app.run(debug=True)
