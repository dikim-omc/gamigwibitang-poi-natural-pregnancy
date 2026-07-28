"""
04_secondary_predictors.py
============================
Secondary, exploratory analysis of baseline predictors of time to
natural pregnancy among herbal-treated patients: multicollinearity
screening (VIF), univariable and multivariable Cox regression,
proportional-hazards diagnostics, and logistic/Firth sensitivity checks.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from firth_impl import firth_logistic
from scipy.stats import norm

df = pd.read_pickle("data/prepared_iptw.pkl")
herbal = df[df['group'] == '한약병용군'].copy()

CANDIDATES = ['age', 'poi_dur', 'cortisol', 'amh', 'afc', 'fsh']
FINAL_VARS = ['age', 'poi_dur', 'cortisol', 'amh_per001']  # pre-specified, EPV-limited

# --- Variance Inflation Factor ---
X = herbal[CANDIDATES].dropna()
Xc = add_constant(X)
vif = pd.DataFrame({
    'var': Xc.columns,
    'VIF': [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]
})
print("VIF:\n", vif[vif['var'] != 'const'].to_string(index=False))

# --- Univariable Cox ---
print("\nUnivariable Cox regression:")
for v in CANDIDATES + ['amh_per001']:
    if v == 'amh':
        continue
    sub = herbal[[v, 'time_months', 'event']].dropna()
    cph = CoxPHFitter()
    cph.fit(sub, 'time_months', 'event', formula=v)
    s = cph.summary
    print(f"  {v}: HR={s.loc[v, 'exp(coef)']:.3f} "
          f"({s.loc[v, 'exp(coef) lower 95%']:.3f}-{s.loc[v, 'exp(coef) upper 95%']:.3f}) "
          f"P={s.loc[v, 'p']:.5f}")

# --- Multivariable Cox (pre-specified, retained regardless of individual significance) ---
sub = herbal[FINAL_VARS + ['time_months', 'event']].dropna()
cphm = CoxPHFitter()
cphm.fit(sub, 'time_months', 'event')
print(f"\nMultivariable Cox (N={len(sub)}, events={int(sub['event'].sum())}):")
print(cphm.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']])

# --- Proportional hazards diagnostics (Schoenfeld residuals) ---
ph_test = proportional_hazard_test(cphm, sub, time_transform='rank')
print("\nSchoenfeld residual test (rank transform):\n", ph_test.summary)

# --- Sensitivity: logistic regression (time-to-event ignored) ---
sub2 = herbal[FINAL_VARS + ['event']].dropna()
Xl = sm.add_constant(sub2[FINAL_VARS])
logit = sm.Logit(sub2['event'], Xl).fit(disp=0)
print("\nSensitivity: multivariable logistic regression (herbal only):")
for v in FINAL_VARS:
    or_ = np.exp(logit.params[v])
    lo, hi = np.exp(logit.conf_int().loc[v])
    print(f"  {v}: OR={or_:.3f} ({lo:.3f}-{hi:.3f}) P={logit.pvalues[v]:.4f}")

# --- Sensitivity: Firth-penalized regression (herbal only) ---
beta, cov, _ = firth_logistic(sub2[FINAL_VARS].values, sub2['event'].values)
se = np.sqrt(np.diag(cov))
print("\nSensitivity: Firth-penalized regression (herbal only):")
for i, v in enumerate(FINAL_VARS):
    b, s = beta[i + 1], se[i + 1]
    or_ = np.exp(b)
    lo, hi = np.exp(b - 1.96 * s), np.exp(b + 1.96 * s)
    p = 2 * (1 - norm.cdf(abs(b / s)))
    print(f"  {v}: OR={or_:.3f} ({lo:.3f}-{hi:.3f}) P={p:.4f}")
