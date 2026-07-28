"""
01_data_preparation.py
=======================
Loads the raw source spreadsheet, cleans and audits it, derives the
time-to-event variables used throughout the analysis, and computes
stabilized inverse probability of treatment weights (IPTW).

Input:  data/raw_cohort.xlsx  (not included in this repository; see
        Data Availability Statement in the manuscript)
Output: data/prepared_iptw.pkl
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm

RAW_PATH = "data/raw_cohort.xlsx"
OUT_PATH = "data/prepared_iptw.pkl"

COLUMNS = ['id', 'age', 'poi_dur', 'group', 'cortisol', 'amh', 'afc', 'fsh',
           'adherence', 'start_date', 'embryos', 'cpr', 'miscarriage', 'lbr',
           'outcome', 'end_date', 'conception_date', 'ovul_recovery', 'notes']


def remove_blank_artifact_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows that are blank across EVERY column (leftover residual rows
    from a prior in-spreadsheet deletion of 10 duplicate patient records;
    Excel leaves the row shell behind when rows are deleted this way).

    This is NOT a missing-data handling step: a row is removed only if
    every substantive column is empty, i.e. it does not correspond to a
    real patient record at all. The free-text 'notes' column is excluded
    from this check: one artifact row was found to carry a stray editorial
    annotation there (e.g. a leftover data-source comment) while every
    other field, including patient ID, is empty; excluding 'notes' from
    the blankness test still correctly identifies it as an artifact rather
    than a real patient record with only 'notes' populated.

    A row with a genuine partial missing value (e.g. one covariate blank
    but an ID and other fields present) would NOT be removed here and
    would instead be caught and reported by run_full_audit() below. This
    distinction matters because the manuscript's Statistical Analysis
    section states there were no missing values among the analyzed
    patients; that claim is verified independently by run_full_audit(),
    not by this cleanup step.
    """
    substantive_cols = [c for c in df.columns if c != 'notes']
    is_fully_blank = df[substantive_cols].isna().all(axis=1)
    n_removed = int(is_fully_blank.sum())
    if n_removed > 0:
        print(f"Removed {n_removed} spreadsheet artifact row(s) "
              f"(all columns except 'notes' empty): index {list(df[is_fully_blank].index)}")
    return df[~is_fully_blank].reset_index(drop=True)


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Data", header=0)
    df.columns = COLUMNS
    df = remove_blank_artifact_rows(df)
    return df


def derive_time_to_event(df: pd.DataFrame) -> pd.DataFrame:
    df['event'] = (df['outcome'] == 'natural_pregnancy').astype(int)
    df['event_time_date'] = np.where(df['event'] == 1, df['conception_date'], df['end_date'])
    df['event_time_date'] = pd.to_datetime(df['event_time_date'])
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['time_days'] = (df['event_time_date'] - df['start_date']).dt.days
    df['time_months'] = df['time_days'] / 30.44
    df['group_herbal'] = (df['group'] == '한약병용군').astype(int)
    df['amh_per001'] = df['amh'] * 100
    return df


def run_full_audit(df: pd.DataFrame) -> list:
    """Logical-consistency audit. Returns a list of issue strings (empty if clean)."""
    issues = []
    if (df['event_time_date'] < df['start_date']).sum() > 0:
        issues.append("Event/censoring date precedes start date")
    if (df['cortisol'] <= 18).sum() > 0:
        issues.append("Baseline cortisol below inclusion threshold (>18 mcg/dL)")
    if (df['amh'] > 0.1).sum() > 0:
        issues.append("Baseline AMH above inclusion threshold (<=0.1 ng/mL)")
    if df['id'].duplicated().sum() > 0:
        issues.append("Duplicate patient ID")
    mismatch = df[(df['outcome'] == 'natural_pregnancy') & (df['event'] != 1)]
    if len(mismatch) > 0:
        issues.append("outcome/event flag mismatch")
    for col in ['age', 'poi_dur', 'cortisol', 'amh', 'afc', 'fsh',
                'start_date', 'event_time_date', 'outcome']:
        if df[col].isna().sum() > 0:
            issues.append(f"Missing values in {col}")
    return issues


def compute_iptw(df: pd.DataFrame) -> pd.DataFrame:
    """Stabilized IPTW for the average treatment effect, using six baseline covariates."""
    covs = ['age', 'poi_dur', 'cortisol', 'amh', 'afc', 'fsh']
    X = sm.add_constant(df[covs])
    y = df['group_herbal']
    ps_model = sm.Logit(y, X).fit(disp=0)
    df['ps'] = ps_model.predict(X)
    p_treat = y.mean()
    df['iptw'] = np.where(
        df['group_herbal'] == 1,
        p_treat / df['ps'],
        (1 - p_treat) / (1 - df['ps'])
    )
    return df


if __name__ == "__main__":
    df = load_and_clean(RAW_PATH)
    df = derive_time_to_event(df)

    issues = run_full_audit(df)
    print(f"N = {len(df)} | Herbal = {(df['group']=='한약병용군').sum()} "
          f"| Control = {(df['group']=='대조군').sum()} | Events = {df['event'].sum()}")
    print("Audit result:", issues if issues else "No issues found")

    # The manuscript's Statistical Analysis section states there were no
    # missing values among the analyzed baseline covariates or outcome
    # variables. This is enforced here, not merely reported: if a future
    # data update introduces missing values, this script will fail loudly
    # rather than silently proceeding with an inaccurate claim in the text.
    missing_data_issues = [i for i in issues if i.startswith("Missing values")]
    assert not missing_data_issues, (
        f"Missing data detected: {missing_data_issues}. "
        f"The manuscript's 'no missing data' statement no longer holds for "
        f"this dataset and must be revised before proceeding."
    )

    df = compute_iptw(df)
    df.to_pickle(OUT_PATH)
    print(f"Saved: {OUT_PATH}")
