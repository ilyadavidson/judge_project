"""
Fit a simple logistic regression: promoted ~ overturn_rate + covariates.
Writes a text summary and a CSV of the modeling table.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def main(args):
    feats = pd.read_parquet(ARTIFACTS_DIR / "judges_features.parquet").copy()

    # Target variable
    if "is_promoted" not in feats.columns:
        raise RuntimeError("judges_features.parquet must include 'is_promoted' (0/1)")
    feats["is_promoted"] = pd.to_numeric(feats["is_promoted"], errors="coerce")

    # Predictors
    X_cols = ["overturn_rate"]
    # add optional controls if present
    for c in ["n_cases","gender","ethnicity"]:
        if c in feats.columns:
            X_cols.append(c)

    # Encode categoricals
    for c in ["gender","ethnicity"]:
        if c in feats.columns:
            feats[c] = feats[c].astype("category")
    X = pd.get_dummies(feats[X_cols], drop_first=True)

    # Drop rows with missing target or predictors
    y = feats["is_promoted"]
    mask = (~y.isna()) & (~X.isna().any(axis=1))
    X, y = X.loc[mask], y.loc[mask]

    # Handle constant / singular matrix
    X = sm.add_constant(X, has_constant="add")
    # Drop zero-variance columns
    nunique = X.nunique()
    keep_cols = nunique[nunique > 1].index
    X = X[keep_cols]

    print(f"[Model] rows: {len(y):,} | features: {X.shape[1]}")
    if X.shape[0] == 0 or X.shape[1] < 2:
        raise RuntimeError("Not enough data to fit the model.")

    try:
        model = sm.Logit(y, X).fit(maxiter=200, disp=False)
    except Exception as e:
        # fallback to penalized to reduce singularity issues
        model = sm.Logit(y, X).fit_regularized(alpha=1e-4, L1_wt=0.0, maxiter=500)

    # Save summary + coefs
    summary_txt = ARTIFACTS_DIR / "logit_summary.txt"
    with summary_txt.open("w") as f:
        f.write(model.summary().as_text())
    print(f"[Model] wrote {summary_txt}")

    coefs = pd.DataFrame({
        "term": X.columns,
        "coef": np.asarray(model.params),
    })
    coefs.to_csv(ARTIFACTS_DIR / "logit_coefs.csv", index=False)
    print(f"[Model] wrote artifacts/logit_coefs.csv")

if __name__ == "__main__":
    _ = argparse.ArgumentParser().parse_args()
    main(_)