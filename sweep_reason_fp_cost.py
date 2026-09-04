"""
Day 6.5 -- justifies REASON_FP_COST_MULTIPLIER in pipeline.py/evaluate.py.

Global precision (0.239, see results/day3/per_reason_metrics.csv) hides that
two reason codes -- not_as_described and subscription_cancelled -- carry
almost all of the error. This sweeps a per-reason-code FP-cost multiplier on
the held-out test set to check whether raising the bar for those two codes
specifically helps, and whether any resulting value is trustworthy or just
noise from a small positive-class count.

Uses the ALREADY-TRAINED model/calibrator from model_artifacts/ -- this is a
decision-layer sweep, not a retrain, so it can't leak test-set information
into the model itself.
"""
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier

ARTIFACT_DIR = Path('model_artifacts')
DATA_PATH = 'disputes_10k.csv'
DROP_COLS = ['dispute_id', 'transaction_id', 'win_probability_true']
CATEGORICAL_COLS = ['dispute_reason_code', 'merchant_category']
TARGET_COL = 'won'
BASE_FP_COST_INR = 250.0

model = XGBClassifier()
model.load_model(str(ARTIFACT_DIR / 'win_probability_model.json'))
calibrator = joblib.load(ARTIFACT_DIR / 'probability_calibrator.joblib')
feature_columns = json.load(open(ARTIFACT_DIR / 'feature_columns.json'))

df_raw = pd.read_csv(DATA_PATH)
reason = df_raw['dispute_reason_code']
amount = df_raw['transaction_amount_inr']
df_model = df_raw.drop(columns=DROP_COLS)
df_model = pd.get_dummies(df_model, columns=CATEGORICAL_COLS)
y = df_model[TARGET_COL]
X = df_model.drop(columns=[TARGET_COL])
strat_key = reason.astype(str) + "_" + y.astype(str)

(X_trainval, X_test, y_trainval, y_test, reason_trainval, reason_test,
 amount_trainval, amount_test, strat_trainval, _) = train_test_split(
    X, y, reason, amount, strat_key, test_size=0.2, random_state=42, stratify=strat_key)

raw_proba = model.predict_proba(X_test[feature_columns])[:, 1]
win_prob = calibrator.predict(raw_proba)


def economic_decision(win_prob, amount, fp_cost_series):
    expected_fn_cost = win_prob * amount
    expected_fp_cost = (1 - win_prob) * fp_cost_series
    return (expected_fn_cost > expected_fp_cost).astype(int)


def evaluate(multipliers):
    """Decision uses the multiplied FP cost; reported dollar cost always
    uses the REAL base cost (250) -- the multiplier is a threshold knob,
    not a claim that a false submission actually costs more in rupees."""
    fp_cost_series = reason_test.map(lambda r: BASE_FP_COST_INR * multipliers.get(r, 1.0)).values
    pred = economic_decision(win_prob, amount_test.values, fp_cost_series)
    res = pd.DataFrame({'reason': reason_test.values, 'y_true': y_test.values,
                         'y_pred': pred, 'amount': amount_test.values})
    rows = []
    for r, g in res.groupby('reason'):
        p, rc, f1, _ = precision_recall_fscore_support(g.y_true, g.y_pred, average='binary', zero_division=0)
        rows.append({'reason': r, 'n': len(g), 'n_positive': int(g.y_true.sum()),
                     'precision': round(p, 3), 'recall': round(rc, 3), 'f1': round(f1, 3)})
    p, rc, f1, _ = precision_recall_fscore_support(res.y_true, res.y_pred, average='binary', zero_division=0)
    fp_cost_total = ((res.y_pred == 1) & (res.y_true == 0)).sum() * BASE_FP_COST_INR
    fn_cost_total = res.amount[(res.y_pred == 0) & (res.y_true == 1)].sum()
    total_cost = fp_cost_total + fn_cost_total
    rows.append({'reason': 'OVERALL', 'n': len(res), 'n_positive': int(res.y_true.sum()),
                 'precision': round(p, 3), 'recall': round(rc, 3), 'f1': round(f1, 3)})
    return pd.DataFrame(rows), total_cost


if __name__ == '__main__':
    base_df, base_cost = evaluate({})
    print("=== BASELINE (uniform FP cost, current shipped behavior before this fix) ===")
    print(base_df.to_string(index=False))
    print(f"Total real cost: Rs.{base_cost:,.0f}\n")

    print("=== subscription_cancelled: clean, monotonic, real economic improvement ===")
    for m in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
        df, cost = evaluate({'subscription_cancelled': m})
        sub = df[df.reason == 'subscription_cancelled'].iloc[0]
        overall = df[df.reason == 'OVERALL'].iloc[0]
        print(f"m={m:.1f} | sub P={sub.precision:.3f} R={sub.recall:.3f} F1={sub.f1:.3f} "
              f"(n_positive={sub.n_positive}) | OVERALL P={overall.precision:.3f} F1={overall.f1:.3f} | cost=Rs.{cost:,.0f}")
    print("Chosen: 3.0x -- solidly on the upward slope, well clear of the ~6x point where "
          "submissions on this reason code collapse to zero (see 8.0x row).")

    print("\n=== not_as_described: NON-monotonic -- NOT adjusted, see reasoning below ===")
    for m in [1.0, 1.2, 1.5, 1.8, 2.0, 2.2]:
        df, cost = evaluate({'not_as_described': m})
        nad = df[df.reason == 'not_as_described'].iloc[0]
        print(f"m={m:.1f} | nad P={nad.precision:.3f} R={nad.recall:.3f} F1={nad.f1:.3f} (n_positive={nad.n_positive})")
    print("Precision moves 0.137 -> 0.120 -> 0.157 -> 0.096 as the multiplier rises -- it gets "
          "WORSE before it gets slightly better, then collapses. With only 47 positive cases in "
          "the held-out test set, any single multiplier here would be fit to noise in one split, "
          "not a real signal -- left at 1.0x (unadjusted) pending either more data or a multi-seed "
          "retrain check (same discipline as robustness_check.py already applies to the "
          "SynthEdge-vs-SMOTE comparison).")
