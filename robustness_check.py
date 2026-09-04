"""
Robustness check: does the SMOTE-fails-on-two-reason-codes finding hold
across multiple random splits, or is it an artifact of random_state=42?
Runs the full baseline/SMOTE/SynthEdge comparison across 5 different seeds.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from synthedge import SynthEdge

df_raw = pd.read_csv('disputes_10k.csv')
reason_all = df_raw['dispute_reason_code']
df_model_all = df_raw.drop(columns=['dispute_id', 'transaction_id', 'win_probability_true'])
df_model_all = pd.get_dummies(df_model_all, columns=['dispute_reason_code', 'merchant_category'])
y_all = df_model_all['won']
X_all = df_model_all.drop(columns=['won'])

SEEDS = [42, 7, 123, 2024, 99]
results = []

for seed in SEEDS:
    strat_key = reason_all.astype(str) + "_" + y_all.astype(str)
    X_trainval, X_test, y_trainval, y_test, reason_trainval, reason_test, strat_trainval, _ = train_test_split(
        X_all, y_all, reason_all, strat_key, test_size=0.2, random_state=seed, stratify=strat_key)
    X_train, _, y_train, _, reason_train, _ = train_test_split(
        X_trainval, y_trainval, reason_trainval, test_size=0.2, random_state=seed, stratify=strat_trainval)

    def make_model(pos_w):
        return XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                              scale_pos_weight=pos_w, eval_metric='logloss', random_state=seed, verbosity=0)

    def recall_for_reasons(model):
        preds = model.predict(X_test)
        out = {}
        for r in ['not_as_described', 'subscription_cancelled']:
            mask = (reason_test.values == r) & (y_test.values == 1)
            if mask.sum() == 0:
                out[r] = None
                continue
            out[r] = recall_score(y_test.values[mask], preds[mask], zero_division=0) if False else \
                     (preds[mask] == 1).mean()
        overall_recall = recall_score(y_test, preds, zero_division=0)
        return out, overall_recall

    # baseline
    pos_w = (y_train == 0).sum() / (y_train == 1).sum()
    m_base = make_model(pos_w); m_base.fit(X_train, y_train)
    base_reasons, base_overall = recall_for_reasons(m_base)

    # SMOTE
    sm = SMOTE(random_state=seed)
    X_sm, y_sm = sm.fit_resample(X_train, y_train)
    pos_w_sm = (y_sm == 0).sum() / (y_sm == 1).sum()
    m_sm = make_model(pos_w_sm); m_sm.fit(X_sm, y_sm)
    smote_reasons, smote_overall = recall_for_reasons(m_sm)

    # SynthEdge
    train_df = X_train.copy(); train_df['won'] = y_train.values
    se = SynthEdge(train_df, target_col='won', verbose=False)
    se.analyze(top_k=10)
    aug_df = se.fill(n_top=10, ctgan_epochs=100, use_ctgan=True)
    X_se = aug_df.drop(columns=['won']); y_se = aug_df['won']
    pos_w_se = (y_se == 0).sum() / (y_se == 1).sum()
    m_se = make_model(pos_w_se); m_se.fit(X_se, y_se)
    se_reasons, se_overall = recall_for_reasons(m_se)

    row = {
        'seed': seed,
        'baseline_overall': round(base_overall, 3),
        'smote_overall': round(smote_overall, 3),
        'synthedge_overall': round(se_overall, 3),
        'baseline_not_as_described': round(base_reasons['not_as_described'], 3) if base_reasons['not_as_described'] is not None else None,
        'smote_not_as_described': round(smote_reasons['not_as_described'], 3) if smote_reasons['not_as_described'] is not None else None,
        'synthedge_not_as_described': round(se_reasons['not_as_described'], 3) if se_reasons['not_as_described'] is not None else None,
        'baseline_subscription_cancelled': round(base_reasons['subscription_cancelled'], 3) if base_reasons['subscription_cancelled'] is not None else None,
        'smote_subscription_cancelled': round(smote_reasons['subscription_cancelled'], 3) if smote_reasons['subscription_cancelled'] is not None else None,
        'synthedge_subscription_cancelled': round(se_reasons['subscription_cancelled'], 3) if se_reasons['subscription_cancelled'] is not None else None,
    }
    results.append(row)
    print(f"seed={seed}: baseline={base_overall:.3f} smote={smote_overall:.3f} synthedge={se_overall:.3f}  |  "
          f"not_as_described: base={row['baseline_not_as_described']} smote={row['smote_not_as_described']} se={row['synthedge_not_as_described']}  |  "
          f"subscription_cancelled: base={row['baseline_subscription_cancelled']} smote={row['smote_subscription_cancelled']} se={row['synthedge_subscription_cancelled']}")

results_df = pd.DataFrame(results)
results_df.to_csv('results/robustness_check.csv', index=False)
print("\n=== Summary across seeds ===")
for col in ['baseline_overall', 'smote_overall', 'synthedge_overall',
            'smote_not_as_described', 'synthedge_not_as_described',
            'smote_subscription_cancelled', 'synthedge_subscription_cancelled']:
    vals = results_df[col].dropna()
    print(f"{col:<32s} mean={vals.mean():.3f}  std={vals.std():.3f}  min={vals.min():.3f}  max={vals.max():.3f}")
