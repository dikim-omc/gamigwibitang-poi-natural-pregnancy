"""
03_sensitivity_firth_evalue.py
================================
Handles complete separation (zero events in the control arm) via
Firth-penalized logistic regression (Heinze & Schemper, 2002),
unweighted and IPTW-weighted. Also computes the E-value
(VanderWeele & Ding, 2017) at the 95% CI bound closest to the null.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from firth_impl import firth_logistic

df = pd.read_pickle("data/prepared_iptw.pkl")
X = df[['group_herbal']].values
y = df['event'].values
w = df['iptw'].values


def fit_and_report(X, y, weights, label):
    beta, cov, n_iter = firth_logistic(X, y, weights=weights)
    se = np.sqrt(np.diag(cov))
    b, s = beta[1], se[1]
    z = b / s
    p = 2 * (1 - norm.cdf(abs(z)))
    or_, lo, hi = np.exp(b), np.exp(b - 1.96 * s), np.exp(b + 1.96 * s)
    print(f"{label}: OR={or_:.1f} (95% CI {lo:.2f}-{hi:.2f}), P={p:.4f}")
    return or_, lo, hi, p


print_unw = fit_and_report(X, y, None, "Unweighted Firth")
print_w = fit_and_report(X, y, w, "IPTW-weighted Firth")


def e_value(rr):
    if rr >= 1:
        return rr + np.sqrt(rr * (rr - 1))
    rr_inv = 1 / rr
    return rr_inv + np.sqrt(rr_inv * (rr_inv - 1))


lo_ci = print_unw[1]  # lower 95% CI bound of the unweighted Firth OR
print(f"\nE-value at 95% CI lower bound ({lo_ci:.2f}): {e_value(lo_ci):.2f}")

# Continuity-corrected point estimate (Haldane-Anscombe correction), reported
# in Supplementary Materials only, since it depends on an arbitrary correction
# for the zero-event control cell.
p_herbal = herbal_rate = y[df['group_herbal'] == 1].mean()
p_control_corrected = 0.5 / (df['group_herbal'] == 0).sum()
rr_corrected = p_herbal / p_control_corrected
print(f"Continuity-corrected point-estimate E-value: {e_value(rr_corrected):.2f}")
