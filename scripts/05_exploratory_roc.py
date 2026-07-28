"""
05_exploratory_roc.py
=======================
Exploratory ROC/threshold analysis for baseline AMH and FSH as
discriminators of natural pregnancy. Baseline cortisol is deliberately
excluded, since it was already used as an inclusion criterion
(>18 mcg/dL) and re-examining it here would risk circular reasoning.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

df = pd.read_pickle("data/prepared_iptw.pkl")
herbal = df[df['group'] == '한약병용군']

for var, label, higher_is_better in [('amh', 'AMH', True), ('fsh', 'FSH', False)]:
    sub = herbal[[var, 'event']].dropna()
    y = sub['event'].values
    score = sub[var].values if higher_is_better else -sub[var].values

    auc = roc_auc_score(y, score)
    fpr, tpr, thresholds = roc_curve(y, score)
    youden_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[youden_idx] if higher_is_better else -thresholds[youden_idx]

    print(f"{label}: AUC={auc:.3f}, Youden cutoff={best_threshold:.3f}, "
          f"sensitivity={tpr[youden_idx]:.3f}, specificity={1 - fpr[youden_idx]:.3f}")
