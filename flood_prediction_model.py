"""
Flood prediction for Polish monthly precipitation data
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns

log = logging.getLogger("flood_pipeline")

# Configuration
RANDOM_STATE = 42
TARGET = "FLOOD_OCCURRENCE"
GROUP_COL = "STATION_CODE"

# Status mapping: 8 = measurement missing, 9 = phenomenon did not occur
STATUS_TO_VALUE = {
    "STATUS_PRECIP": "MONTHLY_PRECIP_MM",
    "STATUS_SNOW_DAYS": "DAYS_SNOW_PRECIP",
    "STATUS_MAX_PRECIP": "MAX_PRECIP_MM",
    "STATUS_SNOW_COVER": "DAYS_SNOW_COVER",
}

MAP_STATUS_9_TO_ZERO = True

NUMERIC_COLS = [
    "YEAR", "MONTH",
    "MONTHLY_PRECIP_MM", "STATUS_PRECIP",
    "DAYS_SNOW_PRECIP", "STATUS_SNOW_DAYS",
    "MAX_PRECIP_MM", "STATUS_MAX_PRECIP",
    "DAY_MAX_PRECIP_START", "DAY_MAX_PRECIP_END",
    "DAYS_SNOW_COVER", "STATUS_SNOW_COVER",
    TARGET,
]

# Transliteration table for Polish diacritics
PL_DIACRITICS = str.maketrans(
    "\u0105\u0107\u0119\u0142\u0144\u00f3\u015b\u017a\u017c"
    "\u0104\u0106\u0118\u0141\u0143\u00d3\u015a\u0179\u017b",
    "acelnoszzACELNOSZZ",
)

LAGGED_COLS = ("MONTHLY_PRECIP_MM", "MAX_PRECIP_MM")

# Final model inputs
FEATURE_COLS = [
    "MONTHLY_PRECIP_MM", "MAX_PRECIP_MM", "DAYS_SNOW_PRECIP", "DAYS_SNOW_COVER",
    "MONTH_SIN", "MONTH_COS",
    "MONTHLY_PRECIP_MM_LAG1", "MONTHLY_PRECIP_MM_LAG2",
    "MAX_PRECIP_MM_LAG1", "MAX_PRECIP_MM_LAG2",
    "PRECIP_ROLL3_SUM",
    "SNOWMELT_RISK", "MAX_EVENT_DURATION_DAYS", "PRECIP_CONCENTRATION",
]

# 0. Loading
def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    last_err: Exception | None = None
    for enc in ("utf-8", "cp1250", "iso-8859-2"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            log.info("Loaded %d rows from %s (encoding=%s)", len(df), csv_path, enc)
            break
        except UnicodeDecodeError as err:
            last_err = err
    else:
        raise UnicodeDecodeError(*last_err.args)

    df.columns = [str(c).strip().upper() for c in df.columns]
    return df

def _coerce_numeric(s: pd.Series) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(s):
        s = (
            s.astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
            .replace({"": None, "nan": None, "None": None})
        )
    return pd.to_numeric(s, errors="coerce")

# 1. Cleaning
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["STATION_NAME"] = (
        df["STATION_NAME"]
        .astype(str)
        .str.translate(PL_DIACRITICS)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    if not pd.api.types.is_numeric_dtype(df[GROUP_COL]):
        df[GROUP_COL] = df[GROUP_COL].astype(str).str.strip()

    for col in NUMERIC_COLS:
        df[col] = _coerce_numeric(df[col])

    for status_col, value_col in STATUS_TO_VALUE.items():
        df.loc[df[status_col].eq(8), value_col] = np.nan
        if MAP_STATUS_9_TO_ZERO:
            df.loc[df[status_col].eq(9), value_col] = 0.0
        else:
            df.loc[df[status_col].eq(9), value_col] = np.nan

    n0 = len(df)
    df = df[df["YEAR"].notna() & df["MONTH"].between(1, 12)]
    df = df.dropna(subset=[TARGET, GROUP_COL])
    df = df.drop_duplicates(subset=[GROUP_COL, "YEAR", "MONTH"], keep="first")
    
    if (dropped := n0 - len(df)):
        log.info("Dropped %d invalid/duplicate rows", dropped)

    df["YEAR"] = df["YEAR"].astype(int)
    df["MONTH"] = df["MONTH"].astype(int)
    df[TARGET] = (df[TARGET] > 0).astype(int)

    return df.reset_index(drop=True)

# 2. Feature engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([GROUP_COL, "YEAR", "MONTH"], kind="mergesort").reset_index(drop=True)

    df["T_INDEX"] = df["YEAR"] * 12 + df["MONTH"]
    g = df.groupby(GROUP_COL, sort=False)

    df["MONTH_SIN"] = np.sin(2.0 * np.pi * df["MONTH"] / 12.0)
    df["MONTH_COS"] = np.cos(2.0 * np.pi * df["MONTH"] / 12.0)

    for col in LAGGED_COLS:
        for k in (1, 2):
            consecutive = g["T_INDEX"].diff(k).eq(k)
            df[f"{col}_LAG{k}"] = g[col].shift(k).where(consecutive)

    roll3 = g["MONTHLY_PRECIP_MM"].transform(
        lambda s: s.rolling(window=3, min_periods=3).sum()
    )
    df["PRECIP_ROLL3_SUM"] = roll3.where(g["T_INDEX"].diff(2).eq(2))

    df["SNOWMELT_RISK"] = df["DAYS_SNOW_COVER"] * df["DAYS_SNOW_PRECIP"]

    start, end = df["DAY_MAX_PRECIP_START"], df["DAY_MAX_PRECIP_END"]
    valid_event = start.ge(1) & end.ge(start) & end.le(31)
    df["MAX_EVENT_DURATION_DAYS"] = (end - start + 1).where(valid_event)

    denom = df["MONTHLY_PRECIP_MM"].where(df["MONTHLY_PRECIP_MM"] > 0)
    df["PRECIP_CONCENTRATION"] = df["MAX_PRECIP_MM"] / denom

    return df

# 3. Modelling
def station_split(df: pd.DataFrame, test_size: float, n_candidates: int = 25) -> tuple[np.ndarray, np.ndarray]:
    y = df[TARGET].to_numpy()
    groups = df[GROUP_COL].to_numpy()
    gss = GroupShuffleSplit(n_splits=n_candidates, test_size=test_size, random_state=RANDOM_STATE)

    fallback: tuple[np.ndarray, np.ndarray] | None = None
    for tr, te in gss.split(df, y, groups):
        fallback = fallback or (tr, te)
        if np.unique(y[tr]).size == 2 and np.unique(y[te]).size == 2:
            return tr, te
    
    log.warning("No candidate split had both classes; using fallback.")
    return fallback

def build_models() -> dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced", learning_rate=0.06, max_iter=400,
            l2_regularization=1.0, random_state=RANDOM_STATE,
        ),
    }

def make_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("clf", model),
        ]
    )

def plot_curves(results: dict, y_test: pd.Series, out_path: Path) -> None:
    prevalence = float(y_test.mean())
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5))

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["proba"])
        ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {res['roc_auc']:.3f})")
        prec, rec, _ = precision_recall_curve(y_test, res["proba"])
        ax_pr.plot(rec, prec, lw=2, label=f"{name} (AP = {res['pr_auc']:.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax_roc.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
               title="ROC curves", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))

    ax_pr.axhline(prevalence, color="k", ls="--", lw=1,
                  label=f"Chance (prevalence = {prevalence:.3f})")
    ax_pr.set(xlabel="Recall", ylabel="Precision",
              title="Precision-Recall curves", xlim=(-0.02, 1.02), ylim=(-0.02, 1.05))

    for ax in (ax_roc, ax_pr):
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right" if ax is ax_roc else "upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def run(args: argparse.Namespace) -> pd.DataFrame:
    df = load_data(Path(args.csv))
    df = clean_data(df)
    df = engineer_features(df)

    X, y = df[FEATURE_COLS], df[TARGET]
    tr, te = station_split(df, test_size=args.test_size)

    st_tr = set(df[GROUP_COL].iloc[tr])
    st_te = set(df[GROUP_COL].iloc[te])
    assert not (st_tr & st_te), "Station leakage between train and test!"

    results: dict[str, dict] = {}
    fitted_rf = None 
    
    for name, model in build_models().items():
        pipe = make_pipeline(model)
        pipe.fit(X.iloc[tr], y.iloc[tr])

        if name == "RandomForest":
            fitted_rf = pipe 
            
        proba_te = pipe.predict_proba(X.iloc[te])[:, 1]
        proba_tr = pipe.predict_proba(X.iloc[tr])[:, 1]
        results[name] = {
            "proba": proba_te,
            "roc_auc": roc_auc_score(y.iloc[te], proba_te),
            "pr_auc": average_precision_score(y.iloc[te], proba_te),
            "train_roc_auc": roc_auc_score(y.iloc[tr], proba_tr), 
        }

    summary = (
        pd.DataFrame(
            {n: {"ROC_AUC": r["roc_auc"], "PR_AUC": r["pr_auc"]} for n, r in results.items()}
        ).T.sort_values("PR_AUC", ascending=False).round(4)
    )
    
    plot_curves(results, y.iloc[te], Path(args.plot_out))
    
    # 4. Feature Importance Plot
    if fitted_rf is not None:
        imputer = fitted_rf.named_steps["imputer"]
        rf_model = fitted_rf.named_steps["clf"]
        
        feature_names = imputer.get_feature_names_out(FEATURE_COLS)
        importances = rf_model.feature_importances_
        
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(15)
        
        fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
        feat_imp.plot(kind="barh", ax=ax_fi, color="#457B9D", edgecolor="white", width=0.7)
        ax_fi.set_title("Top 15 Feature Importances", fontsize=14, fontweight="bold")
        ax_fi.set_xlabel("Importance (Gini)", fontsize=12)
        ax_fi.grid(axis="x", linestyle="--", alpha=0.5)
        ax_fi.spines['top'].set_visible(False)
        ax_fi.spines['right'].set_visible(False)
        
        fi_path = Path("feature_importance.png")
        fig_fi.tight_layout()
        fig_fi.savefig(fi_path, dpi=150, facecolor="white")
        plt.close(fig_fi)

        # 5. Confusion Matrix Plot
        y_pred_rf = fitted_rf.predict(X.iloc[te])
        cm = confusion_matrix(y.iloc[te], y_pred_rf)
        
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=["No Flood", "Flood"],
                    yticklabels=["No Flood", "Flood"],
                    annot_kws={"size": 14, "weight": "bold"})
        ax_cm.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
        ax_cm.set_ylabel("Actual Class", fontsize=11)
        ax_cm.set_xlabel("Predicted Class", fontsize=11)
        
        cm_path = Path("confusion_matrix_rf.png")
        fig_cm.tight_layout()
        fig_cm.savefig(cm_path, dpi=150, facecolor="white")
        plt.close(fig_cm)

    return summary

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flood-occurrence prediction pipeline")
    p.add_argument("--csv", default="merged_precipitation_1996_2010.csv")
    p.add_argument("--plot-out", default="flood_models_roc_pr.png")
    p.add_argument("--test-size", type=float, default=0.25)
    return p.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    run(parse_args(argv))
    return 0

if __name__ == "__main__":
    sys.exit(main())