"""
OASIS Cross-Sectional MRI Dataset Parser
Extracts structured clinical + volumetric features from individual patient .txt files.
"""

import os
import re
import pandas as pd
import numpy as np


def parse_patient_file(filepath: str) -> dict | None:
    """
    Parse a single OASIS patient .txt file and return a feature dictionary.
    Fields: SESSION_ID, AGE, SEX, HAND, EDUC, SES, CDR, MMSE, eTIV, ASF, nWBV
    """
    data = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        patterns = {
            "SESSION_ID": r"SESSION ID:\s+(\S+)",
            "AGE":        r"AGE:\s+([\d.]+)",
            "SEX":        r"M/F:\s+(\w+)",
            "HAND":       r"HAND:\s+(\w+)",
            "EDUC":       r"EDUC:\s+([\d.]+)",
            "SES":        r"SES:\s+([\d.]+)",
            "CDR":        r"CDR:\s+([\d.]+)",
            "MMSE":       r"MMSE:\s+([\d.]+)",
            "eTIV":       r"eTIV:\s+([\d.]+)",
            "ASF":        r"ASF:\s+([\d.]+)",
            "nWBV":       r"nWBV:\s+([\d.]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                data[key] = match.group(1)
            else:
                data[key] = None

    except Exception as e:
        print(f"  [WARN] Could not parse {filepath}: {e}")
        return None

    return data


def load_oasis_dataset(disc_root: str) -> pd.DataFrame:
    """
    Walk all patient sub-folders under disc_root and aggregate into a DataFrame.
    Automatically handles multiple discs if disc_root contains multiple disc* folders.
    """
    records = []

    # Support both single disc and multi-disc layouts
    for root, dirs, files in os.walk(disc_root):
        for fname in files:
            if re.match(r"^OAS1_\d+_MR\d+\.txt$", fname):
                fpath = os.path.join(root, fname)
                rec = parse_patient_file(fpath)
                if rec:
                    records.append(rec)

    if not records:
        raise ValueError(f"No patient .txt files found under: {disc_root}")

    df = pd.DataFrame(records)

    # Type coercions
    numeric_cols = ["AGE", "EDUC", "SES", "CDR", "MMSE", "eTIV", "ASF", "nWBV"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Encode categorical
    df["SEX_ENC"]  = df["SEX"].map({"Male": 1, "Female": 0, "M": 1, "F": 0}).fillna(0).astype(int)
    df["HAND_ENC"] = df["HAND"].map({"Right": 1, "Left": 0}).fillna(1).astype(int)

    print(f"[INFO] Loaded {len(df)} patient records from {disc_root}")
    print(f"[INFO] CDR distribution:\n{df['CDR'].value_counts().sort_index()}")

    return df


def preprocess(df: pd.DataFrame, binary: bool = True, include_young_controls: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and label y.

    Includes clinical imputation per OASIS study protocol:
      - Missing SES and EDUC imputed with median values.
      - Young subjects (age < 50) without cognitive complaints are non-demented
        controls (CDR = 0.0, MMSE = 30.0).

    Args:
        df:     Raw OASIS DataFrame
        binary: If True, collapse CDR > 0 -> 1 (Dementia vs Non-Dementia).
                If False, keep multi-class (0, 0.5, 1, 2).
        include_young_controls: Whether to include young healthy controls.

    Returns:
        X (DataFrame), y (Series)
    """
    df_proc = df.copy()

    # Impute SES and EDUC with median of available records
    if "SES" in df_proc.columns:
        df_proc["SES"] = df_proc["SES"].fillna(df_proc["SES"].median())
    if "EDUC" in df_proc.columns:
        df_proc["EDUC"] = df_proc["EDUC"].fillna(df_proc["EDUC"].median())

    if include_young_controls:
        # OASIS protocol: Young subjects (age < 50) are clinically normal non-demented controls
        young_mask = df_proc["CDR"].isnull() & (df_proc["AGE"] < 50)
        df_proc.loc[young_mask, "CDR"] = 0.0
        df_proc.loc[young_mask, "MMSE"] = 30.0

    feature_cols = ["AGE", "SEX_ENC", "HAND_ENC", "EDUC", "SES", "MMSE", "eTIV", "ASF", "nWBV"]

    # Drop rows with missing core features
    df_clean = df_proc.dropna(subset=feature_cols + ["CDR"]).copy()

    X = df_clean[feature_cols].copy()
    y_raw = df_clean["CDR"].copy()

    if binary:
        y = (y_raw > 0).astype(int)
        y.name = "DEMENTIA"
    else:
        # Map 0 -> 0, 0.5 -> 1, 1 -> 2, 2 -> 3
        y = y_raw.map({0: 0, 0.5: 1, 1.0: 2, 2.0: 3}).fillna(0).astype(int)
        y.name = "CDR_CLASS"

    print(f"[INFO] Features: {list(X.columns)}")
    print(f"[INFO] Samples available: {len(X)} (Healthy: {(y==0).sum()}, Dementia: {(y==1).sum()})")

    return X, y

    return X, y


if __name__ == "__main__":
    disc_root = r"e:\New folder\Downloads\QuantumX\oasis_cross-sectional_disc1"
    df = load_oasis_dataset(disc_root)
    X, y = preprocess(df, binary=True)
    print("\nSample features:\n", X.head())
    print("\nSample labels:\n", y.head())
