"""
Day 3/6 -- evaluation: per-reason precision/recall/F1 and rupee-denominated
cost, using the CALIBRATED model + the same per-dispute economic decision
rule pipeline.py actually runs (not a single fixed threshold -- see
train_model.py's calibration fix for why that changed).

Loads model_artifacts/ rather than retraining, so these numbers are
guaranteed to match production behavior, not a separately-trained
approximation of it.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier

ARTIFACT_DIR = Path('model_artifacts')
OUTDIR = Path('results/day3')
OUTDIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = 'disputes_10k.csv'
DROP_COLS = ['dispute_id', 'transaction_id', 'win_probability_true']
CATEGORICAL_COLS = ['dispute_reason_code', 'merchant_category']
TARGET_COL = 'won'

# Chosen default, not a placeholder: reasoned from industry dispute-handling
# time estimates (see write-up), adjusted down for what this pipeline
# automates away. Sensitivity is still checked below in case real ops data
# later suggests a different number -- the default doesn't stop being worth
# stress-testing just because it's no longer provisional.
FP_COST_INR = 250.0


def load_artifacts():
    model = XGBClassifier()
    model.load_model(str(ARTIFACT_DIR / 'win_probability_model.json'))
    calibrator = joblib.load(ARTIFACT_DIR / 'probability_calibrator.joblib')
    with open(ARTIFACT_DIR / 'feature_columns.json') as f:
        feature_columns = json.load(f)
    return model, calibrator, feature_columns


def rebuild_test_split():
    """Reconstructs the exact held-out test split train_model.py used (same
    random_state, same split logic) so this evaluates genuinely unseen data."""
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
    return X_test, y_test, reason_test, amount_test


def economic_decision(win_prob, amount, fp_cost):
    expected_fn_cost = win_prob * amount
    expected_fp_cost = (1 - win_prob) * fp_cost
    return (expected_fn_cost > expected_fp_cost).astype(int)


def main():
    model, calibrator, feature_columns = load_artifacts()
    X_test, y_test, reason_test, amount_test = rebuild_test_split()

    raw_proba = model.predict_proba(X_test[feature_columns])[:, 1]
    win_prob = calibrator.predict(raw_proba)  # calibrated -- see train_model.py

    pred = economic_decision(win_prob, amount_test.values, FP_COST_INR)

    fp_cost_total = ((pred == 1) & (y_test.values == 0)).sum() * FP_COST_INR
    fn_cost_total = amount_test.values[(pred == 0) & (y_test.values == 1)].sum()
    total_cost = fp_cost_total + fn_cost_total
    cost_nothing = amount_test.values[y_test.values == 1].sum()
    cost_everything = (y_test.values == 0).sum() * FP_COST_INR

    print(f"Per-dispute economic rule (FP_COST=Rs.{FP_COST_INR:.0f}, calibrated probabilities): "
          f"total cost Rs.{total_cost:,.0f} = FP Rs.{fp_cost_total:,.0f} + FN Rs.{fn_cost_total:,.0f}")
    print(f"Reference -- flag everything as insufficient: Rs.{cost_nothing:,.0f}")
    print(f"Reference -- submit everything automatically: Rs.{cost_everything:,.0f}")
    print(f"Gate saves Rs.{min(cost_nothing, cost_everything) - total_cost:,.0f} vs. the better naive policy\n")

    res = pd.DataFrame({'reason': reason_test.values, 'y_true': y_test.values,
                         'y_pred': pred, 'amount': amount_test.values})
    rows = []
    for r, g in res.groupby('reason'):
        p, rc, f1, _ = precision_recall_fscore_support(g.y_true, g.y_pred, average='binary', zero_division=0)
        fn_r = g.amount[(g.y_pred == 0) & (g.y_true == 1)].sum()
        fp_r = ((g.y_pred == 1) & (g.y_true == 0)).sum() * FP_COST_INR
        rows.append({'reason': r, 'n': len(g), 'n_positive': int(g.y_true.sum()),
                      'precision': round(p, 3), 'recall': round(rc, 3), 'f1': round(f1, 3),
                      'fn_cost_inr': fn_r, 'fp_cost_inr': fp_r})
    p, rc, f1, _ = precision_recall_fscore_support(res.y_true, res.y_pred, average='binary', zero_division=0)
    rows.append({'reason': 'OVERALL', 'n': len(res), 'n_positive': int(res.y_true.sum()),
                 'precision': round(p, 3), 'recall': round(rc, 3), 'f1': round(f1, 3),
                 'fn_cost_inr': res.amount[(res.y_pred == 0) & (res.y_true == 1)].sum(),
                 'fp_cost_inr': ((res.y_pred == 1) & (res.y_true == 0)).sum() * FP_COST_INR})
    report_df = pd.DataFrame(rows)
    report_df.to_csv(OUTDIR / 'per_reason_metrics.csv', index=False)
    print(report_df.to_string(index=False))

    # Sensitivity is still worth showing with a chosen default: it's now
    # "here's our default, and here's how gracefully the system degrades if
    # the real ops number differs" rather than "we don't know what to use."
    sens_rows = []
    for fp_candidate in [100, 150, 250, 400, 600, 1000]:
        pred_c = economic_decision(win_prob, amount_test.values, fp_candidate)
        p, rc, f1, _ = precision_recall_fscore_support(y_test, pred_c, average='binary', zero_division=0)
        sens_rows.append({'assumed_fp_cost_inr': fp_candidate, 'precision': round(p, 3),
                           'recall': round(rc, 3), 'f1': round(f1, 3)})
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(OUTDIR / 'fp_cost_sensitivity.csv', index=False)
    print(f"\nFP-cost sensitivity (Rs.{FP_COST_INR:.0f} is the chosen default):")
    print(sens_df.to_string(index=False))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sens_df.assumed_fp_cost_inr, sens_df.precision, marker='o', label='precision', color='#2980b9')
    ax.plot(sens_df.assumed_fp_cost_inr, sens_df.recall, marker='s', label='recall', color='#27ae60')
    ax.axvline(FP_COST_INR, linestyle='--', color='gray', alpha=0.7, label=f'chosen default (Rs.{FP_COST_INR:.0f})')
    ax.set_xlabel('Assumed FP cost (INR per wrongly submitted case)')
    ax.set_ylabel('Metric value')
    ax.set_title('Gate precision/recall vs. FP-cost assumption (calibrated probabilities)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / 'fp_cost_sensitivity.png', dpi=150)
    plt.close(fig)

    with open(OUTDIR / 'summary.json', 'w') as f:
        json.dump({
            'fp_cost_default_inr': FP_COST_INR,
            'decision_rule': 'per-dispute economic rule on calibrated probabilities '
                              '(win_prob vs FP_COST/(FP_COST+amount))',
            'total_cost_inr': float(total_cost),
            'cost_flag_everything_insufficient_inr': float(cost_nothing),
            'cost_submit_everything_inr': float(cost_everything),
            'per_reason': report_df.to_dict(orient='records'),
        }, f, indent=2)

    print(f"\nSaved -> {OUTDIR}/per_reason_metrics.csv, fp_cost_sensitivity.csv, "
          f"fp_cost_sensitivity.png, summary.json")


if __name__ == '__main__':
    main()
