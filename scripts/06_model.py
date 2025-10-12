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

# 1. Target and base numeric features
####################################################################################
    feature_dataset = pd.read_csv(ARTIFACTS_DIR / "features.csv")
    y       = feature_dataset["is_promoted"].astype(float)  
    X       = pd.DataFrame({
        "overturnrate": pd.to_numeric(feature_dataset["overturnrate"], errors="coerce"),
        "average_politicality": pd.to_numeric(feature_dataset["avg_politicality"], errors="coerce"),
    })

    # 2. Categorical features (one-hot encoded)
    ####################################################################################
    gender_dummies = pd.get_dummies(feature_dataset["gender"], prefix="gender", drop_first=True, dtype=float)
    X = pd.concat([X, gender_dummies], axis=1)

    aba_num = pd.to_numeric(feature_dataset['aba rating'], errors="coerce").fillna(2)
    aba_dummies = pd.get_dummies(
        aba_num.astype("category"),
        prefix="ABA",
        drop_first=True,   # omits one category (baseline)
        dtype=float
    )
    X = pd.concat([X, aba_dummies], axis=1)

    X = X.apply(pd.to_numeric, errors="coerce")
    mask = ~(y.isna() | X.isna().any(axis=1))
    y_clean = y.loc[mask]
    X_clean = X.loc[mask]

    X_clean = sm.add_constant(X_clean, has_constant="add")


    # 3. Regression models
    ####################################################################################
    model = sm.Logit(y_clean, X_clean).fit(maxiter=100)
    print("=== Logistic Regression: is promoted ~ overturnrate + gender + ethnicity ===")
    print(model.summary())

    ols_model = sm.OLS(y_clean, X_clean).fit(maxiter=100)
    print("\n=== OLS Regression: is promoted ~ overturnrate + gender + ethnicity ===")
    print(ols_model.summary())
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