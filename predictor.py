"""Inference logic: raw applicant -> engineered features -> preprocessor -> model."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

# ---- Input schema (ranges taken from the training dataset) -------------------
NUMERIC_FIELDS = {
    "Age": (18, 69),
    "Income": (15000, 149999),
    "LoanAmount": (5000, 249999),
    "CreditScore": (300, 849),
    "MonthsEmployed": (0, 119),
    "NumCreditLines": (1, 4),
    "InterestRate": (2.0, 25.0),
    "LoanTerm": (12, 60),
    "DTIRatio": (0.1, 0.9),
}
CATEGORICAL_FIELDS = {
    "Education": ["High School", "Bachelor's", "Master's", "PhD"],
    "EmploymentType": ["Full-time", "Part-time", "Self-employed", "Unemployed"],
    "MaritalStatus": ["Single", "Married", "Divorced"],
    "HasMortgage": ["No", "Yes"],
    "HasDependents": ["No", "Yes"],
    "LoanPurpose": ["Auto", "Business", "Education", "Home", "Other"],
    "HasCoSigner": ["No", "Yes"],
}

# Quartile edges of Income from the training data
INCOME_BINS = [-np.inf, 48825.5, 82466.0, 116219.0, np.inf]
INCOME_LABELS = ["Low", "MidLow", "MidHigh", "High"]

# Training-time medians, used when a ratio cannot be computed
MEDIANS = {
    "LoanToIncome": 1.546720133161837,
    "Monthly_Income": 6872.166666666667,
    "MonthlyPayment": 4442.059862803908,
    "PaymentToIncome": 0.6825416644296394,
    "TotalInterestCost": 18586.13452726009,
    "DebtBurden": 51399.6,
    "IncomePerCreditLine": 32980.0,
    "Rate_x_Term": 397.92,
    "CreditScore_x_Income": 43521296.0,
    "EmploymentYears": 5.0,
    "EmploymentToAge": 0.11410256410256411,
    "IsUnstableJob": 1.0,
    "Log_Income": 11.320153492301433,
    "Log_LoanAmount": 11.756318602503125,
    "Log_TotalInterestCost": 9.830224928289166,
}

# Readable names for the "key factors" panel
FEATURE_LABELS = {
    "Age": "Age",
    "Income": "Annual income",
    "LoanAmount": "Loan amount",
    "CreditScore": "Credit score",
    "MonthsEmployed": "Employment length",
    "NumCreditLines": "Number of credit lines",
    "InterestRate": "Interest rate",
    "LoanTerm": "Loan term",
    "DTIRatio": "Debt-to-income ratio",
    "LoanToIncome": "Loan relative to income",
    "Monthly_Income": "Annual income",
    "MonthlyPayment": "Monthly payment",
    "PaymentToIncome": "Payment relative to income",
    "TotalInterestCost": "Total interest cost",
    "DebtBurden": "Debt burden",
    "IncomePerCreditLine": "Income per credit line",
    "Rate_x_Term": "Interest rate over the loan term",
    "CreditScore_x_Income": "Credit score and income combined",
    "EmploymentYears": "Employment length",
    "EmploymentToAge": "Employment length relative to age",
    "Log_Income": "Annual income",
    "Log_LoanAmount": "Loan amount",
    "Log_TotalInterestCost": "Total interest cost",
    "HasMortgage": "Has a mortgage",
    "HasDependents": "Has dependents",
    "HasCoSigner": "Has a co-signer",
    "Education": "Education level",
    "AgeGroup": "Age group",
    "CreditBand": "Credit score band",
    "IncomeBand": "Income band",
    "EmploymentType_Part-time": "Part-time employment",
    "EmploymentType_Self-employed": "Self-employment",
    "EmploymentType_Unemployed": "Unemployed",
    "MaritalStatus_Married": "Married",
    "MaritalStatus_Single": "Single",
    "LoanPurpose_Business": "Business loan",
    "LoanPurpose_Education": "Education loan",
    "LoanPurpose_Home": "Home loan",
    "LoanPurpose_Other": "Other loan purpose",
    "IsUnstableJob": "Unstable employment",
}


class ValidationError(ValueError):
    """Raised when the submitted applicant data is invalid."""

    def __init__(self, message, fields=None):
        super().__init__(message)
        self.fields = fields or {}


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    fe = data.copy()

    fe["LoanToIncome"] = fe["LoanAmount"] / fe["Income"]
    fe["Monthly_Income"] = fe["Income"] / 12

    r = fe["InterestRate"] / 100 / 12
    n = fe["LoanTerm"]
    fe["MonthlyPayment"] = fe["LoanAmount"] * r / (1 - (1 + r) ** (-n))
    fe["PaymentToIncome"] = fe["MonthlyPayment"] / fe["Monthly_Income"]
    fe["TotalInterestCost"] = fe["MonthlyPayment"] * n - fe["LoanAmount"]

    fe["DebtBurden"] = fe["DTIRatio"] * fe["LoanAmount"]
    fe["IncomePerCreditLine"] = fe["Income"] / fe["NumCreditLines"]
    fe["Rate_x_Term"] = fe["InterestRate"] * fe["LoanTerm"]
    fe["CreditScore_x_Income"] = fe["CreditScore"] * fe["Income"]

    fe["EmploymentYears"] = fe["MonthsEmployed"] / 12
    fe["EmploymentToAge"] = fe["EmploymentYears"] / fe["Age"]
    fe["IsUnstableJob"] = fe["EmploymentType"].isin(["Unemployed", "Part-time"]).astype(int)

    fe["AgeGroup"] = pd.cut(
        fe["Age"], bins=[17, 25, 35, 45, 55, 70],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"],
    )
    fe["CreditBand"] = pd.cut(
        fe["CreditScore"], bins=[299, 579, 669, 739, 799, 850],
        labels=["Poor", "Fair", "Good", "VeryGood", "Exceptional"],
    )
    fe["IncomeBand"] = pd.cut(fe["Income"], bins=INCOME_BINS, labels=INCOME_LABELS)

    for col in ["Income", "LoanAmount", "TotalInterestCost"]:
        fe[f"Log_{col}"] = np.log1p(fe[col])

    fe = fe.replace([np.inf, -np.inf], np.nan)
    fe = fe.fillna(MEDIANS)

    for col in ["HasMortgage", "HasDependents", "HasCoSigner", "Education", "AgeGroup",
                "CreditBand", "IncomeBand", "EmploymentType", "MaritalStatus", "LoanPurpose"]:
        fe[col] = fe[col].astype(str)
    return fe


def validate(payload: dict) -> dict:
    """Return a clean applicant dict or raise ValidationError with per-field messages."""
    errors, clean = {}, {}
    for name, (lo, hi) in NUMERIC_FIELDS.items():
        raw = payload.get(name)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors[name] = "Enter a number."
            continue
        if not np.isfinite(value) or value < lo or value > hi:
            errors[name] = f"Enter a value between {lo:g} and {hi:g}."
            continue
        clean[name] = value
    for name, options in CATEGORICAL_FIELDS.items():
        value = payload.get(name)
        if value not in options:
            errors[name] = "Choose one of the listed options."
        else:
            clean[name] = value
    if "LoanTerm" in clean and clean["LoanTerm"] not in (12, 24, 36, 48, 60):
        errors["LoanTerm"] = "Choose 12, 24, 36, 48 or 60 months."
    if errors:
        raise ValidationError("Some fields need attention.", errors)
    for k in ("Age", "MonthsEmployed", "NumCreditLines", "LoanTerm", "CreditScore"):
        clean[k] = int(clean[k])
    return clean


class LoanDefaultPredictor:
    def __init__(self, artifact_dir: Path = ARTIFACT_DIR):
        self.preprocessor = joblib.load(artifact_dir / "preprocessor.joblib")
        bundle = joblib.load(artifact_dir / "best_model.joblib")
        self.model = bundle["model"]
        self.threshold = float(bundle["threshold"])
        self.features = list(bundle["features"])
        self.input_columns = list(self.preprocessor.feature_names_in_)

    def _transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        fe = create_features(raw)[self.input_columns]
        processed = pd.DataFrame(
            self.preprocessor.transform(fe),
            columns=self.preprocessor.get_feature_names_out(),
        )
        return processed.reindex(columns=self.features)

    def predict_proba_frame(self, raw: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._transform(raw))[:, 1]

    def predict(self, applicant: dict, top_n: int = 6) -> dict:
        raw = pd.DataFrame([applicant])
        X = self._transform(raw)
        probability = float(self.model.predict_proba(X)[0, 1])

        # Logistic regression: each feature's pull on the log-odds = coefficient x value.
        # Features that share a label (e.g. Income and Log_Income) are summed.
        contrib = X.iloc[0].to_numpy() * self.model.coef_[0]
        merged = {}
        for feat, c in zip(self.features, contrib):
            label = FEATURE_LABELS.get(feat, feat)
            merged[label] = merged.get(label, 0.0) + float(c)
        ranked = sorted(merged.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
        factors = [
            {"label": k, "effect": round(v, 4), "direction": "raises" if v > 0 else "lowers"}
            for k, v in ranked
        ]

        fe = create_features(raw)
        return {
            "probability": probability,
            "threshold": self.threshold,
            "high_risk": probability >= self.threshold,
            "factors": factors,
            "monthly_payment": float(fe["MonthlyPayment"].iloc[0]),
            "payment_to_income": float(fe["PaymentToIncome"].iloc[0]),
        }
