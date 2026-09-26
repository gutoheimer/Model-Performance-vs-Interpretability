# SPDX-FileCopyrightText: 2026 Augusto Rheinheimer
#
# SPDX-License-Identifier: MIT
"""
Fetches and preprocesses the 8 medical datasets used to extend the Kruschel et al.
(2026) benchmark to the medical domain, and persists each as a static CSV under
`data/`, matching the original benchmark's convention of reading pre-downloaded
CSVs rather than fetching data live during experiment runs (see load_datasets.py).

This mirrors the exploratory preprocessing already done in medical_datasets.ipynb,
with one addition: the Diabetes 130-US Hospitals diagnosis codes (diag_1/2/3) are
grouped into the 9 ICD-9 categories from Strack et al. (2014, Table 2) so they
survive the shared _preprocess_columns() cardinality filter (>25 unique values
are dropped) instead of being discarded outright.

Run once: `python prepare_medical_data.py`. Requires: pandas, numpy, scikit-learn,
ucimlrepo.
"""

import warnings

warnings.filterwarnings("ignore")

import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    raise ImportError("Install with: pip install ucimlrepo")

DATA_DIR = "data"


def icd9_group(code):
    """
    Map a raw ICD-9 diagnosis code to one of the 9 categories used by
    Strack et al. (2014, Table 2): Circulatory, Respiratory, Digestive,
    Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other.
    V- and E-coded (supplementary classification) and missing codes fall
    into 'Other'.
    """
    if pd.isna(code):
        return "Other"
    code = str(code)
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    if code.startswith("250"):
        return "Diabetes"
    try:
        num = float(code)
    except ValueError:
        return "Other"
    if 390 <= num <= 459 or num == 785:
        return "Circulatory"
    if 460 <= num <= 519 or num == 786:
        return "Respiratory"
    if 520 <= num <= 579 or num == 787:
        return "Digestive"
    if 800 <= num <= 999:
        return "Injury"
    if 710 <= num <= 739:
        return "Musculoskeletal"
    if 580 <= num <= 629 or num == 788:
        return "Genitourinary"
    if 140 <= num <= 239:
        return "Neoplasms"
    return "Other"


def classify_columns(X):
    """
    Numerical vs categorical split for the manifest handed to load_datasets.py
    loaders. Follows the convention already used across the original loaders
    (e.g. load_stroke_data): binary flags are treated as categorical (routed
    through OneHotEncoder), everything else non-numeric is categorical, and
    remaining numeric columns with more than 2 distinct values are numerical.
    """
    categorical_cols, numerical_cols = [], []
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            categorical_cols.append(col)
        elif X[col].nunique(dropna=True) <= 2:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    return numerical_cols, categorical_cols


def write_dataset(alias, X, y, target_name, numerical_cols=None, categorical_cols=None, groups=None):
    """
    groups: optional Series of a repeated-measures identifier (e.g. patient ID),
    written alongside the features/target so grouped CV (GroupKFold) can key on
    it downstream without re-fetching the source data.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if numerical_cols is None or categorical_cols is None:
        numerical_cols, categorical_cols = classify_columns(X)
    df = X.copy()
    df[target_name] = y.values if hasattr(y, "values") else y
    if groups is not None:
        df["group_id"] = groups.values if hasattr(groups, "values") else groups
    path = os.path.join(DATA_DIR, f"{alias}.csv")
    df.to_csv(path, index=False)
    print(f"\n{'=' * 60}")
    print(f"  Wrote {path}  ({df.shape[0]:,} rows x {df.shape[1]} cols)")
    print(f"  numerical_cols   = {numerical_cols}")
    print(f"  categorical_cols = {categorical_cols}")
    if groups is not None:
        print(f"  group_id         = present ({groups.nunique()} unique groups)")
    print(f"{'=' * 60}")


# ── 1. Diabetes 130-US Hospitals (UCI 296) — classification ─────────────────
def prepare_readmission():
    ds = fetch_ucirepo(id=296)
    X = ds.data.features.copy()
    y_raw = ds.data.targets.copy()
    y = (y_raw.iloc[:, 0] == "<30").astype(int)

    drop_cols = [
        c
        for c in X.columns
        if c in ("weight", "encounter_id", "patient_nbr") or X[c].isnull().mean() > 0.5
    ]
    X = X.drop(columns=drop_cols)

    # Group ICD-9 diagnosis codes into 9 categories (Strack et al., 2014, Table 2)
    # so they survive the >25-distinct-value categorical cutoff downstream.
    for diag_col in ("diag_1", "diag_2", "diag_3"):
        if diag_col in X.columns:
            X[diag_col] = X[diag_col].apply(icd9_group)

    # admission_type_id / discharge_disposition_id / admission_source_id are
    # coded integer IDs, not continuous quantities — force categorical.
    id_like_cols = [
        c
        for c in ("admission_type_id", "discharge_disposition_id", "admission_source_id")
        if c in X.columns
    ]
    numerical_cols, categorical_cols = classify_columns(X)
    for c in id_like_cols:
        if c in numerical_cols:
            numerical_cols.remove(c)
            categorical_cols.append(c)

    write_dataset("readmission", X, y, "readmitted_lt30", numerical_cols, categorical_cols)


# ── 2. Thyroid Disease (UCI allhypo.data) — classification ──────────────────
def prepare_thyroid():
    THYROID_URL = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "thyroid-disease/allhypo.data"
    )
    THYROID_COLS = [
        "age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_meds",
        "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",
        "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych",
        "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4",
        "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured", "TBG",
        "referral_source", "target_raw",
    ]

    df = pd.read_csv(THYROID_URL, header=None, names=THYROID_COLS, na_values="?")

    df["target_raw"] = df["target_raw"].str.split("|").str[0].str.rstrip(".")
    y = (df["target_raw"] != "negative").astype(int)

    feature_cols = [c for c in THYROID_COLS if c != "target_raw"]
    X = df[feature_cols].copy()

    binary_cols = [
        "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_meds",
        "sick", "pregnant", "thyroid_surgery", "I131_treatment",
        "query_hypothyroid", "query_hyperthyroid", "lithium", "goitre",
        "tumor", "hypopituitary", "psych",
        "TSH_measured", "T3_measured", "TT4_measured",
        "T4U_measured", "FTI_measured", "TBG_measured",
    ]
    for col in binary_cols:
        if col in X.columns:
            X[col] = X[col].map({"t": 1, "f": 0, "M": 1, "F": 0}).fillna(np.nan)

    num_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
    for col in num_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    write_dataset("thyroid", X, y, "hypothyroid")


# ── 3. Wisconsin Breast Cancer Diagnostic (UCI 17) — classification ─────────
def prepare_breastcancer():
    ds = fetch_ucirepo(id=17)
    X = ds.data.features.copy()
    y = (ds.data.targets.iloc[:, 0] == "M").astype(int)
    write_dataset("breastcancer", X, y, "malignant")


# ── 4. Heart Failure Clinical Records (UCI 519) — classification ────────────
def prepare_heartfailure():
    ds = fetch_ucirepo(id=519)
    X = ds.data.features.copy()
    y = ds.data.targets.iloc[:, 0].astype(int)
    write_dataset("heartfailure", X, y, "death_event")


# ── 5. Pima Indians Diabetes — classification ────────────────────────────────
def prepare_pima():
    PIMA_URL = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "pima-indians-diabetes.csv"
    )
    PIMA_COLS = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
    ]
    df = pd.read_csv(PIMA_URL, header=None, names=PIMA_COLS)
    y = df["Outcome"].astype(int)
    X = df.drop(columns="Outcome").copy()

    zero_coded = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_coded:
        X.loc[X[col] == 0, col] = np.nan

    write_dataset("pima", X, y, "Outcome")


# ── 6. Diabetes Progression (Efron et al., 2004) — regression ───────────────
def prepare_progression():
    raw = load_diabetes(as_frame=True)
    X = raw.data.copy()
    y = raw.target.copy()
    X.columns = ["age", "sex", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"]
    write_dataset("progression", X, y, "Progression")


# ── 7. Chronic Kidney Disease (UCI 336) — regression (serum creatinine) ─────
def prepare_kidney():
    ds = fetch_ucirepo(id=336)
    X_raw = ds.data.features.copy()

    sc_candidates = [
        c for c in X_raw.columns if "sc" in c.lower() or "creatinine" in c.lower()
    ]
    sc_col = sc_candidates[0] if sc_candidates else X_raw.select_dtypes(include="number").columns[0]

    y = pd.to_numeric(X_raw[sc_col], errors="coerce")
    X = X_raw.drop(columns=[sc_col])

    mask = y.notna()
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    write_dataset("kidney", X, y, "serum_creatinine")


# ── 8. Parkinsons Telemonitoring (UCI 189) — regression ─────────────────────
def prepare_parkinsons():
    ds = fetch_ucirepo(id=189)
    X = ds.data.features.copy()
    y_raw = ds.data.targets.copy()

    if "total_UPDRS" in y_raw.columns:
        y = y_raw["total_UPDRS"].astype(float)
    else:
        y = y_raw.iloc[:, -1].astype(float)

    # The patient identifier is not in .features at all — ucimlrepo keeps it
    # separate in .ids ('subject#', 42 unique patients across 5,875 recordings).
    # It must be carried through as a grouping key (not a predictive feature)
    # so downstream CV can be patient-level (GroupKFold) rather than
    # recording-level, avoiding leakage from a patient's other recordings.
    groups = ds.data.ids["subject#"]

    write_dataset("parkinsons", X, y, "total_UPDRS", groups=groups)


if __name__ == "__main__":
    prepare_readmission()
    prepare_thyroid()
    prepare_breastcancer()
    prepare_heartfailure()
    prepare_pima()
    prepare_progression()
    prepare_kidney()
    prepare_parkinsons()
    print("\nAll 8 medical datasets written to ./data/")
