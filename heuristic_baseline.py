"""
Day 3/6 -- heuristic baseline: what would a human analyst do with no ML at
all, just the same reason-relevant evidence fields the pipeline already
surfaces? Answers the obvious judge question -- "does the ML actually add
anything over a simple rule?" -- with a number instead of an assertion.

Rule: submit if at least half of the reason-relevant evidence fields are
present/true. Uses the exact same REASON_RELEVANT_FIELDS mapping and cost
function as pipeline.py/evaluate.py, so this is a fair, apples-to-apples
comparison, not a strawman.
"""
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from evaluate import rebuild_test_split, load_artifacts, economic_decision, FP_COST_INR
from pipeline import REASON_RELEVANT_FIELDS

df_raw = pd.read_csv('disputes_10k.csv')
X_test, y_test, reason_test, amount_test = rebuild_test_split()

# Need the raw (pre-dummy) evidence columns for the heuristic, not the
# one-hot X_test used by the model -- pull those rows straight from source.
test_raw = df_raw.loc[X_test.index]


def heuristic_decision_for_row(row) -> int:
    fields = REASON_RELEVANT_FIELDS.get(row['dispute_reason_code'], [])
    if not fields:
        return 0
    positive = sum(1 for f in fields if row.get(f) not in (0, False, None))
    return 1 if positive / len(fields) >= 0.5 else 0


heuristic_pred = test_raw.apply(heuristic_decision_for_row, axis=1).values

# ---- ML model, for direct comparison on the identical test set ----
model, calibrator, feature_columns = load_artifacts()
raw_proba = model.predict_proba(X_test[feature_columns])[:, 1]
ml_win_prob = calibrator.predict(raw_proba)
ml_pred = economic_decision(ml_win_prob, amount_test.values, FP_COST_INR)


def cost_of(pred):
    fp = ((pred == 1) & (y_test.values == 0)).sum() * FP_COST_INR
    fn = amount_test.values[(pred == 0) & (y_test.values == 1)].sum()
    return fp + fn, fp, fn


def report(pred, label):
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average='binary', zero_division=0)
    cost, fp_cost, fn_cost = cost_of(pred)
    print(f"{label:<20s} precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}  "
          f"total_cost=Rs.{cost:,.0f} (FP Rs.{fp_cost:,.0f} + FN Rs.{fn_cost:,.0f})")
    return {'method': label, 'precision': round(p, 3), 'recall': round(r, 3),
            'f1': round(f1, 3), 'total_cost_inr': cost}


rows = []
rows.append(report(heuristic_pred, 'heuristic (>=50% evidence)'))
rows.append(report(ml_pred, 'ML (calibrated + gate)'))

# per-reason head-to-head
res = pd.DataFrame({'reason': reason_test.values, 'y_true': y_test.values,
                     'heuristic': heuristic_pred, 'ml': ml_pred})
print("\nPer-reason recall, heuristic vs ML:")
per_reason_rows = []
for r, g in res.groupby('reason'):
    pos = g[g.y_true == 1]
    if len(pos) == 0:
        continue
    h_rec = (pos.heuristic == 1).mean()
    m_rec = (pos.ml == 1).mean()
    print(f"  {r:<25s} n_pos={len(pos):<4d} heuristic_recall={h_rec:.3f}  ml_recall={m_rec:.3f}")
    per_reason_rows.append({'reason': r, 'n_positive': len(pos),
                             'heuristic_recall': round(h_rec, 3), 'ml_recall': round(m_rec, 3)})

pd.DataFrame(rows).to_csv('results/day3/heuristic_vs_ml.csv', index=False)
pd.DataFrame(per_reason_rows).to_csv('results/day3/heuristic_vs_ml_per_reason.csv', index=False)
print("\nSaved -> results/day3/heuristic_vs_ml.csv, heuristic_vs_ml_per_reason.csv")
