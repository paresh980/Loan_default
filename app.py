"""Flask app for the loan default risk model."""
import logging
import os

from flask import Flask, jsonify, render_template, request

from predictor import (CATEGORICAL_FIELDS, NUMERIC_FIELDS, LoanDefaultPredictor,
                       ValidationError, validate)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("loan-default")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # payloads are tiny; reject anything large

# Load once at start-up so every request reuses the same model objects.
predictor = LoanDefaultPredictor()


@app.get("/")
def index():
    return render_template(
        "index.html",
        numeric=NUMERIC_FIELDS,
        categorical=CATEGORICAL_FIELDS,
        threshold=predictor.threshold,
    )


@app.get("/health")
def health():
    return jsonify(status="ok", threshold=predictor.threshold)


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send the applicant details as a JSON object."), 400
    try:
        applicant = validate(payload)
        return jsonify(predictor.predict(applicant))
    except ValidationError as exc:
        return jsonify(error=str(exc), fields=exc.fields), 422
    except Exception:
        log.exception("Prediction failed")
        return jsonify(error="Something went wrong while scoring this application."), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
