"""
Day 4/6 -- train, calibrate, and persist the final win-probability model.

Three-way split, not two: train_core (SynthEdge-augmented, for learning),
calib (real/unaugmented, held out, for probability calibration), test
(untouched, for evaluation). scale_pos_weight fixes decision quality at a
threshold but badly distorts predict_proba as an actual probability
(verified: raw Brier score was WORSE than predicting the base rate for
everyone) -- isotonic calibration on a real, unaugmented, held-out split
fixes this and is required before any per-dispute cost math is trustworthy.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier
from synthedge import SynthEdge

ARTIFACT_DIR = Path('model_artifacts')
ARTIFACT_DIR.mkdir(exist_ok=True)

DATA_PATH = 'disputes_10k.csv'
DROP_COLS = ['dispute_id', 'transaction_id', 'win_probability_true']
CATEGORICAL_COLS = ['dispute_reason_code', 'merchant_category']
TARGET_COL = 'won'


def main():
    df_raw = pd.read_csv(DATA_PATH)
    reason = df_raw['dispute_reason_code']
    df_model = df_raw.drop(columns=DROP_COLS)
    df_model = pd.get_dummies(df_model, columns=CATEGORICAL_COLS)

    y = df_model[TARGET_COL]
    X = df_model.drop(columns=[TARGET_COL])
    strat_key = reason.astype(str) + "_" + y.astype(str)

    X_trainval, X_test, y_trainval, y_test, strat_trainval, _ = train_test_split(
        X, y, strat_key, test_size=0.2, random_state=42, stratify=strat_key, )
    # Held out from training/augmentation entirely -- real data only, used
    # solely to check whether predicted probabilities match reality.
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=strat_trainval)

    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values
    se = SynthEdge(train_df, target_col=TARGET_COL, verbose=False)
    se.analyze(top_k=10)
    aug_df = se.fill(n_top=10, ctgan_epochs=100, use_ctgan=True)
    X_aug = aug_df.drop(columns=[TARGET_COL])
    y_aug = aug_df[TARGET_COL]

    pos_w = (y_aug == 0).sum() / (y_aug == 1).sum()
    model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                           scale_pos_weight=pos_w, eval_metric='logloss',
                           random_state=42, verbosity=0)
    model.fit(X_aug, y_aug)

    # ---- calibration: fit isotonic regression on REAL, unaugmented,
    # held-out data the model has never seen in any form
    raw_proba_calib = model.predict_proba(X_calib)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_proba_calib, y_calib)

    # ---- verify on the untouched test set (not the calibration set --
    # that would just confirm the fit worked, not that it generalizes)
    raw_proba_test = model.predict_proba(X_test)[:, 1]
    calibrated_proba_test = calibrator.predict(raw_proba_test)

    brier_raw = brier_score_loss(y_test, raw_proba_test)
    brier_cal = brier_score_loss(y_test, calibrated_proba_test)
    brier_base_rate = brier_score_loss(y_test, [y_train.mean()] * len(y_test))

    model.save_model(str(ARTIFACT_DIR / 'win_probability_model.json'))
    joblib.dump(calibrator, ARTIFACT_DIR / 'probability_calibrator.joblib')
    with open(ARTIFACT_DIR / 'feature_columns.json', 'w') as f:
        json.dump(list(X.columns), f, indent=2)
    with open(ARTIFACT_DIR / 'training_meta.json', 'w') as f:
        json.dump({
            'synthedge_severity': se.severity['severity'],
            'rows_added_by_synthedge': int(len(aug_df) - len(train_df)),
            'training_rows': int(len(X_train)),
            'calibration_rows': int(len(X_calib)),
            'test_rows': int(len(X_test)),
            'reason_codes': sorted(reason.unique().tolist()),
            'brier_score_raw_uncalibrated': round(float(brier_raw), 4),
            'brier_score_calibrated': round(float(brier_cal), 4),
            'brier_score_base_rate_baseline': round(float(brier_base_rate), 4),
            'calibration_note': (
                'Raw XGBoost probabilities (scale_pos_weight-adjusted) were '
                'WORSE than the base-rate baseline -- badly overconfident, not '
                'usable as true probabilities. Isotonic calibration on a real, '
                'unaugmented, held-out split fixes this; use the calibrator, '
                'not raw predict_proba, for any cost-based decision.'
            ),
        }, f, indent=2)

    print(f"Brier (raw, uncalibrated):  {brier_raw:.4f}")
    print(f"Brier (base-rate baseline): {brier_base_rate:.4f}")
    print(f"Brier (calibrated):         {brier_cal:.4f}")
    print(f"\nSaved model + calibrator + artifacts -> {ARTIFACT_DIR}/")


if __name__ == '__main__':
    main()
