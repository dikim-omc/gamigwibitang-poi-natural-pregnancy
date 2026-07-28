[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21633771.svg)](https://doi.org/10.5281/zenodo.21633771)

# Analysis Code — Natural pregnancy after adjunctive Gamigwibitang treatment in idiopathic POI

This repository contains the statistical analysis code supporting the
manuscript "Natural pregnancy after adjunctive Gamigwibitang treatment
in women with idiopathic premature ovarian insufficiency and elevated
cortisol: A retrospective cohort study" (submitted to PLOS ONE).

## Data availability

The de-identified source dataset (`data/raw_cohort.xlsx`) is **not**
included in this repository. It was accessed under an Institutional
Review Board exemption (Korea Ministry of Health and Welfare Designated
Public IRB, Exemption No. P01-202606-01-086) that restricts redistribution
of patient-level data. See the manuscript's Data Availability Statement
for information on requesting access.

## Requirements

```
pip install -r requirements.txt
```

Key package versions used for the reported results:

- Python 3.12
- lifelines 0.30.3
- statsmodels 0.14.6
- scikit-learn (for ROC analysis)
- pandas, numpy, scipy

## Pipeline (run in order)

| Script | Purpose | Corresponds to |
|---|---|---|
| `scripts/01_data_preparation.py` | Load, clean, audit raw data; compute IPTW | Methods: Study Design, Statistical Analysis (IPTW) |
| `scripts/02_primary_analysis.py` | Kaplan-Meier, unweighted/IPTW-weighted log-rank | Results: Natural Pregnancy Incidence; Robustness |
| `scripts/03_sensitivity_firth_evalue.py` | Firth-penalized regression, E-value | Results: Robustness; Table 2 |
| `scripts/04_secondary_predictors.py` | VIF, univariable/multivariable Cox, Schoenfeld test, logistic/Firth sensitivity | Results: Secondary Exploratory Analysis; Table 3; S1, S3, S7 Tables |
| `scripts/05_exploratory_roc.py` | ROC/threshold analysis for AMH, FSH | S5 Table |
| `scripts/firth_impl.py` | Custom Firth-penalized logistic regression (Heinze & Schemper, 2002), supporting case weights for IPTW combination | Used by scripts 03 and 04 |

All scripts read from and write to a local `data/` directory (not
tracked in version control except for derived, de-identified summary
outputs where applicable).

## Citation

If you use this code, please cite the manuscript (citation details to
be added upon publication) and this repository:

Kim DI. Analysis code for "Natural pregnancy after adjunctive
Gamigwibitang treatment in women with idiopathic premature ovarian
insufficiency and elevated cortisol: A retrospective cohort study."
Zenodo. 2026. https://doi.org/10.5281/zenodo.21633771
