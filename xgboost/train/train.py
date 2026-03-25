import pandas as pd
import numpy as np
import wandb

import xgboost as xgb
import lightgbm as lgb


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score


col_names = ["label"] + [f"int{i}" for i in range(1, 14)] + [f"cat{i}" for i in range(1, 27)]
df = pd.read_csv("dac/train.txt", sep="\t", names=col_names, nrows=1_000_000)

num_cols = [f"int{i}" for i in range(1, 14)]
cat_cols = [f"cat{i}" for i in range(1, 27)]

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna("missing")

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
    

X = df[num_cols + cat_cols]
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)


xgb_preds = xgb_model.predict_proba(X_val)[:, 1]
xgb_auc = roc_auc_score(y_val, xgb_preds)
print(f"XGBoost AUC: {xgb_auc:.4f}")

xgb_model.save_model("../serve/xgb_model.json")
print("XGBoost model saved.")

lgb_model = lgb.LGBMClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.log_evaluation(period=50)],
)

lgb_preds = lgb_model.predict_proba(X_val)[:, 1]
lgb_auc = roc_auc_score(y_val, lgb_preds)
print(f"LightGBM AUC: {lgb_auc:.4f}")

lgb_model.booster_.save_model("../serve/lgb_model.txt")
print("LightGBM model saved.")

# A/B comparison — compute full metrics for both models
xgb_labels = (xgb_preds >= 0.5).astype(int)
lgb_labels = (lgb_preds >= 0.5).astype(int)

run = wandb.init(project="ctr-ab-test", job_type="train")

wandb.log({
    "xgb/auc":       xgb_auc,
    "xgb/precision": precision_score(y_val, xgb_labels),
    "xgb/recall":    recall_score(y_val, xgb_labels),
    "xgb/f1":        f1_score(y_val, xgb_labels),
    "lgb/auc":       lgb_auc,
    "lgb/precision": precision_score(y_val, lgb_labels),
    "lgb/recall":    recall_score(y_val, lgb_labels),
    "lgb/f1":        f1_score(y_val, lgb_labels),
})
wandb.finish()
print("Logged to W&B.")