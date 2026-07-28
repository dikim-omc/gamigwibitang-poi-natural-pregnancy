"""
02_primary_analysis.py
=======================
Primary endpoint: time to natural pregnancy, herbal co-treatment vs.
control. Kaplan-Meier cumulative incidence, unweighted and IPTW-weighted
log-rank test.
"""
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

df = pd.read_pickle("data/prepared_iptw.pkl")
herbal = df[df['group'] == '한약병용군']
control = df[df['group'] == '대조군']

kmf_h = KaplanMeierFitter().fit(herbal['time_months'], herbal['event'], label="Herbal")
kmf_c = KaplanMeierFitter().fit(control['time_months'], control['event'], label="Control")

print("Cumulative incidence, herbal group:")
for t in [3, 6, 12]:
    print(f"  {t} months: {(1 - kmf_h.predict(t)) * 100:.2f}%")

res_unweighted = logrank_test(
    herbal['time_months'], control['time_months'],
    herbal['event'], control['event']
)
print(f"\nUnweighted log-rank: chi2={res_unweighted.test_statistic:.2f}, "
      f"P={res_unweighted.p_value:.5f}")

res_weighted = logrank_test(
    herbal['time_months'], control['time_months'],
    herbal['event'], control['event'],
    weights_A=herbal['iptw'], weights_B=control['iptw']
)
print(f"IPTW-weighted log-rank: chi2={res_weighted.test_statistic:.2f}, "
      f"P={res_weighted.p_value:.5f}")

nat = df[df['outcome'] == 'natural_pregnancy']
print(f"\nMedian time to natural conception: {nat['time_months'].median():.2f} months "
      f"(IQR {nat['time_months'].quantile(.25):.2f}-{nat['time_months'].quantile(.75):.2f})")
print(f"Live birth: {(nat['lbr'] == 'Y').sum()}/{len(nat)}")
print(f"Miscarriage: {(nat['miscarriage'] == 'Y').sum()}/{len(nat)}")
