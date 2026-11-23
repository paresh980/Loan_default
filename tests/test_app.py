import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app import app

VALID = dict(Age=30, Income=50000, LoanAmount=15000, CreditScore=700, MonthsEmployed=36,
             NumCreditLines=3, InterestRate=10.5, LoanTerm=36, DTIRatio=0.3,
             Education="Bachelor's", EmploymentType="Full-time", MaritalStatus="Single",
             HasMortgage="Yes", HasDependents="No", LoanPurpose="Education", HasCoSigner="No")


@pytest.fixture
def client():
    return app.test_client()


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Score application" in r.data


def test_health(client):
    assert client.get("/health").get_json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/api/predict", json=VALID)
    body = r.get_json()
    assert r.status_code == 200
    assert 0 <= body["probability"] <= 1
    assert body["threshold"] == pytest.approx(0.63)
    assert len(body["factors"]) == 6


def test_risky_applicant_scores_higher(client):
    risky = {**VALID, "Age": 22, "Income": 28000, "LoanAmount": 180000, "CreditScore": 420,
             "MonthsEmployed": 4, "InterestRate": 22, "LoanTerm": 60, "DTIRatio": 0.8,
             "EmploymentType": "Unemployed", "HasCoSigner": "No", "HasMortgage": "No"}
    safe = {**VALID, "Age": 45, "Income": 110000, "LoanAmount": 20000, "CreditScore": 760,
            "MonthsEmployed": 100, "InterestRate": 6, "LoanTerm": 24, "DTIRatio": 0.2}
    p_risky = client.post("/api/predict", json=risky).get_json()["probability"]
    p_safe = client.post("/api/predict", json=safe).get_json()["probability"]
    assert p_risky > p_safe


def test_out_of_range_rejected(client):
    r = client.post("/api/predict", json={**VALID, "CreditScore": 9000, "Education": "Wizard"})
    assert r.status_code == 422
    assert set(r.get_json()["fields"]) == {"CreditScore", "Education"}


def test_missing_and_bad_body(client):
    assert client.post("/api/predict", json={}).status_code == 422
    assert client.post("/api/predict", data="nope", content_type="text/plain").status_code == 400
