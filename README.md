# Loan default risk scorer

Flask app that serves a trained loan-default model (logistic regression, review threshold 0.63) with a browser front end.

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py                   # http://localhost:5000
pytest                          # optional: run the tests
```

## Deploy on Render (free)

1. Create a GitHub repository and push this folder to it (`git init`, `git add .`, `git commit`, `git push`).
2. On render.com choose **New > Blueprint** (or **New > Web Service**) and connect the repository.
3. Render reads `render.yaml`. For a manual Web Service use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --workers 2 --timeout 60`
   - Health check path: `/health`
4. Open the URL Render gives you. The free tier sleeps after inactivity, so the first request can take about 30 seconds.

The `Procfile` also works on Railway and Heroku, and the `Dockerfile` on any container host.

## Project layout

```
app.py            Flask routes (/, /api/predict, /health)
predictor.py      Validation, feature engineering and model inference
artifacts/        best_model.joblib and preprocessor.joblib
templates/        index.html
static/           style.css and app.js
tests/            pytest suite
```

## Important

Keep the library versions in `requirements.txt` the same as the environment where you trained and saved the model. If you retrain, replace the two files in `artifacts/` and rerun `pytest`.

## API

`POST /api/predict` with a JSON body containing the 16 applicant fields returns `probability`, `threshold`, `high_risk`, `factors`, `monthly_payment` and `payment_to_income`. Invalid input returns status 422 with a `fields` object.
