"""
Day 2 -- baseline vs SMOTE vs SynthEdge comparison, using the SAME
three-way split (train_core / calibration / test) as train_model.py, so
these numbers -- including rows_added_by_synthedge -- genuinely match
production rather than a similar-but-different standalone run.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from synthedge import SynthEdge

DATA_PATH = 'disputes_10k.csv'
DROP_COLS = ['dispute_id', 'transaction_id', 'win_probability_true']
CATEGORICAL_COLS = ['dispute_reason_code', 'merchant_category']
TARGET_COL = 'won'

df_raw = pd.read_csv(DATA_PATH)
reason = df_raw['dispute_reason_code']
df_model = df_raw.drop(columns=DROP_COLS)
df_model = pd.get_dummies(df_model, columns=CATEGORICAL_COLS)
y = df_model[TARGET_COL]
X = df_model.drop(columns=[TARGET_COL])
strat_key = reason.astype(str) + "_" + y.astype(str)

# Identical two-step split to train_model.py: trainval/test, then
# train_core/calibration out of trainval. Day 2 only needs train_core (the
# actual training data) and test (the actual held-out evaluation data) --
# the calibration slice belongs to train_model.py's job, not this comparison,
# but excluding it here is what makes train_core match production exactly.
X_trainval, X_test, y_trainval, y_test, reason_trainval, reason_test, strat_trainval, _ = train_test_split(
    X, y, reason, strat_key, test_size=0.2, random_state=42, stratify=strat_key)
X_train, X_calib, y_train, y_calib, reason_train, reason_calib = train_test_split(
    X_trainval, y_trainval, reason_trainval, test_size=0.2, random_state=42, stratify=strat_trainval)


def make_model(pos_w):
    return XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                          scale_pos_weight=pos_w, eval_metric='logloss', random_state=42, verbosity=0)


def evaluate(model, label):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    pr = average_precision_score(y_test, y_proba)
    print(f"\n=== {label} ===")
    print(f"recall={recall:.3f}  precision={precision:.3f}  f1={f1:.3f}  roc_auc={auc:.3f}  pr_auc={pr:.3f}")
    res = pd.DataFrame({'reason': reason_test.values, 'y_true': y_test.values, 'y_pred': y_pred})
    for r, g in res.groupby('reason'):
        pos = g[g.y_true == 1]
        if len(pos) == 0:
            continue
        rec = (pos.y_pred == 1).mean()
        print(f"  {r:<25s} n_pos={len(pos):<4d} recall={rec:.3f}")
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
        print("Top 5 features:")
        print(imp.head(5).to_string())
    return {'label': label, 'recall': recall, 'precision': precision, 'f1': f1, 'roc_auc': auc, 'pr_auc': pr}


results = []

pos_w = (y_train == 0).sum() / (y_train == 1).sum()
m = make_model(pos_w)
m.fit(X_train, y_train)
results.append(evaluate(m, 'baseline (scale_pos_weight only)'))

sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_resample(X_train, y_train)
pos_w_sm = (y_sm == 0).sum() / (y_sm == 1).sum()
m = make_model(pos_w_sm)
m.fit(X_sm, y_sm)
results.append(evaluate(m, f'smote (+{(y_sm==1).sum()-(y_train==1).sum()})'))

train_df = X_train.copy()
train_df[TARGET_COL] = y_train.values
se = SynthEdge(train_df, target_col=TARGET_COL, verbose=False)
se.analyze(top_k=10)
aug_df = se.fill(n_top=10, ctgan_epochs=100, use_ctgan=True)
X_se = aug_df.drop(columns=[TARGET_COL])
y_se = aug_df[TARGET_COL]
pos_w_se = (y_se == 0).sum() / (y_se == 1).sum()
added = (y_se == 1).sum() - (y_train == 1).sum()
m = make_model(pos_w_se)
m.fit(X_se, y_se)
results.append(evaluate(m, f'synthedge (severity={se.severity["severity"]}, +{added})'))

summary = pd.DataFrame(results)
summary.to_csv('day2_final_comparison.csv', index=False)
print("\n\nSaved -> day2_final_comparison.csv")
print(summary.to_string(index=False))
print(f"\nTrain rows: {len(X_train)}  (matches train_model.py's training_rows in training_meta.json)")
