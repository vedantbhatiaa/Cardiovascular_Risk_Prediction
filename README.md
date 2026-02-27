# Cardiovascular Risk Score Prediction
### MSIN0097 — Predictive Analytics | UCL MSc Business Analytics 2025–26

> **Task:** Predict a continuous cardiovascular risk score (0–100) from 15 patient features using an end-to-end ML pipeline. Secondary task: classify patients into risk bands (Low / Medium / High) via derived thresholds.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Dataset](#dataset)
4. [Environment Setup](#environment-setup)
5. [How to Run](#how-to-run)
6. [Notebook Structure](#notebook-structure)
7. [Final Results](#final-results)
8. [Key Design Decisions](#key-design-decisions)
9. [Agent Tooling](#agent-tooling)
10. [References](#references)

---

## Project Overview

| Item | Detail |
|------|--------|
| **Problem type** | Regression (primary) + derived classification |
| **Target variable** | `heart_disease_risk_score` (continuous, 0–100) |
| **Dataset** | Cardiovascular Risk Dataset — 5,500 patients, 15 features |
| **Final model** | MLP Neural Network (256→128→64, ReLU, Adam, early stopping) |
| **Test MAE** | ~2.38 risk-score points |
| **Test R²** | ~0.98 |
| **Macro F1 (bands)** | ~0.90 |
| **High-risk Recall** | ~0.93 |
| **Split** | 70% train / 15% val / 15% test — stratified by risk band |

---

## Repository Structure

```
📦 cardiovascular-risk-prediction/
├── cardiovascular_risk_prediction.ipynb   # Main notebook — Steps 1–6
├── cardiovascular_risk_dataset.csv        # Dataset (see Dataset section)
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
└── appendix/
    ├── Unified_Session_Log.docx           # Agent usage log + decision register
    └── Assessment_Information_Coursework_2025.docx  # Step-by-step decision log
```

---

## Dataset

**File:** `cardiovascular_risk_dataset.csv`  
**Source:** Kaggle — Cardiovascular Risk Dataset (synthetic)  
**Rows:** 5,500 | **Columns:** 17 (15 features + 1 target + 1 dropped)

| Column | Type | Notes |
|--------|------|-------|
| `heart_disease_risk_score` | float (0–100) | **Target variable** |
| `risk_category` | categorical | Dropped — algorithmic bucketing of target (leakage) |
| `Patient_ID` | integer | Dropped — identifier only |
| `systolic_bp`, `diastolic_bp` | numeric | Strongest predictors (r=0.90, r=0.81) |
| `cholesterol_mg_dl` | numeric | r=0.85 with target |
| `smoking_status` | ordinal | Never / Former / Current → encoded 0/1/2 |
| `family_history_heart_disease` | binary | No/Yes → encoded 0/1 |
| `alcohol_units_per_week` | numeric | Right-skewed — Winsorised at 99th pct for linear pipeline |
| `age_bmi` | engineered | age × bmi interaction (r=0.838) — tree pipeline only |

**No missing values. No duplicates.**

---

## Environment Setup

### Option 1 — pip (recommended)

```bash
# 1. Clone or download the repository
git clone <repo-url>
cd cardiovascular-risk-prediction

# 2. Create a virtual environment (Python 3.12)
python -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Option 2 — conda

```bash
conda create -n cv-risk python=3.12
conda activate cv-risk
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "import pandas, numpy, sklearn, xgboost, statsmodels; print('All packages OK')"
```

---

## How to Run

### Full notebook (recommended)

```bash
jupyter notebook cardiovascular_risk_prediction.ipynb
```

Then **run all cells in order** (`Kernel → Restart & Run All`).

> ⚠️ **Important:** The test set evaluation cell (Cell 106) is designed to be run **once only**. Re-running after seeing results would constitute implicit data leakage into model selection decisions.

### Expected runtime

| Section | Approximate time |
|---------|-----------------|
| Steps 1–3 (EDA + Preprocessing) | ~1–2 min |
| Step 4 (Model Exploration, 4 models + CV) | ~3–5 min |
| Step 5 — XGBoost tuning (60 iter × 5 folds) | ~4–6 min |
| Step 5 — RF tuning (40 iter × 5 folds) | ~4–6 min |
| Step 5 — Test evaluation + Step 6 | ~2–3 min |
| **Total** | **~15–20 min** |

---

## Notebook Structure

The notebook follows the six required steps from the brief:

| Step | Cells | Description |
|------|-------|-------------|
| **Step 1** | 1–2 | Problem framing: target definition, metrics, assumptions |
| **Step 2** | 3–28 | EDA: distributions, outliers, correlations, boundary region analysis |
| **Step 3** | 29–55 | Preprocessing: encoding, age_bmi engineering, VIF, stratified split, pipelines, leakage guards |
| **Step 4** | 56–77 | Model exploration: Linear Regression, Random Forest, XGBoost, MLP (+ scaling fix) |
| **Step 5** | 78–114 | Tuning (RandomizedSearchCV), error analysis (6 sub-sections), test evaluation, three-way comparison |
| **Step 6** | 115–121 | Final model selection rationale, limitations, model card |

---

## Final Results

### Model comparison (validation set)

| Model | Val MAE | CV MAE | CV Std | Val R² |
|-------|---------|--------|--------|--------|
| Linear Regression | 3.266 | 3.288 | 0.079 | 0.968 |
| Random Forest | 3.255 | 3.381 | 0.129 | 0.971 |
| XGBoost | 2.650 | 2.673 | 0.094 | 0.982 |
| **MLP (Neural Network)** ✓ | **2.400** | **2.358** | **0.046** | **0.984** |
| XGBoost (Tuned) | 2.663 | 2.597 | 0.059 | 0.982 |
| Random Forest (Tuned) | 3.246 | 3.197 | 0.064 | 0.971 |

### Final test set (held-out, evaluated once)

| Metric | Value |
|--------|-------|
| MAE | ~2.38 |
| RMSE | ~3.09 |
| R² | ~0.98 |
| Macro F1 (derived bands) | ~0.90 |
| High-risk Recall | ~0.93 |

### Three-way generalisation check

Val ≈ CV ≈ Test — confirms model generalises and validation set was not implicitly overfit during model selection.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Task type | Regression (primary) | `risk_category` is algorithmic bucketing of score — using it as target = if-else rule learning |
| `risk_category` | Dropped entirely | Direct leakage of target variable |
| Feature engineering | `age_bmi` interaction | r=0.838 with target, outperforms age (0.69) and bmi (0.71) individually |
| `age_bmi` scope | Tree pipeline only | VIF=61.3 — unsafe for linear models |
| `alcohol_units_per_week` | Winsorised at 99th pct | 3.9% outliers — real population values, not errors |
| CV strategy (tuning) | KFold(5) | StratifiedKFold is incompatible with continuous regression targets in RandomizedSearchCV |
| CV strategy (exploration) | StratifiedKFold(5) | Manual fold loop — stratification on risk bands for stable band distributions |
| Winner model | MLP (Neural Network) | Outperforms all candidates on Val MAE, CV MAE, CV Std, and Val R² |
| MLP inputs | `X_train_tree_scaled` | MLP requires scaled inputs — StandardScaler fitted on training data only (no leakage) |

---


## References

- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. Proceedings of the ACM Conference on Fairness, Accountability, and Transparency.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825–2830.
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD '16.

---

*MSIN0097 — Predictive Analytics | UCL MSc Business Analytics 2025–26 | Submission deadline: 03 March 2026*