# explainable-hybrid-rf-et-stroke-risk
Imbalance-aware stroke risk prediction on tabular clinical data using a soft-voting Random Forest + Extra Trees ensemble with SHAP/LIME interpretability.


# Explainable Hybrid RF+ET Framework for Imbalanced Clinical Stroke Risk Prediction

This repository contains the implementation of an **imbalance-aware** and **explainable** machine learning pipeline for **clinical stroke risk prediction** using structured/tabular data.  
The core model is a **soft-voting hybrid ensemble** of **Random Forest (RF)** and **Extra Trees (ET)**, with **SMOTE** for class imbalance handling and **SHAP + LIME** for interpretability.

---

## Key Features
- ✅ End-to-end pipeline: preprocessing → imbalance handling → training → evaluation
- ✅ **Hybrid RF+ET soft-voting ensemble**
- ✅ Handles class imbalance using **SMOTE**
- ✅ Baseline comparison (LR, SVM, MLP, NB, Boosting models, etc.)
- ✅ Explainability:
  - **SHAP** (global feature attribution)
  - **LIME** (instance-level explanations)
- ✅ Performance visualization:
  - Confusion matrix, ROC curve, PR curve, threshold vs F1

---

## Dataset
This notebook expects the CSV file:

**`Clinical Stroke Risk Prediction Dataset.csv`**

Download the dataset from Mendeley Data (as referenced in the paper).  
After downloading, place the CSV in the same directory as the notebook, or update the file path inside the notebook.

> ⚠️ Note: The dataset is not included in this repository.

---

## Repository Contents
- `Hybrid_RF+ET.ipynb` — Main notebook (full pipeline + experiments + XAI)
- `requirements.txt` — Python dependencies
- `assets/` — (optional) figures/images for README

---

## Installation

### 1) Create environment (recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
