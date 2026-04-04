# Model-T

**Training with Partially Known Hidden Variables**

> Rule-Primed Inversion · Calibrated Interval Prediction · No Full Labels Required

---

## Overview

Model-T is a machine learning framework for systems where a key variable is only measurable for a fraction of your data. Instead of imputing or ignoring it, Model-T **recovers it mathematically** using model inversion — then produces calibrated prediction intervals at test time.

Starting from less than 30% labeled rows, the framework fills in the hidden variable for the entire training set and achieves **84.5% interval coverage** on held-out test data.

---

## The Problem

In many real-world systems, the relationship between inputs and outputs is governed by a variable that is expensive, delayed, or impossible to measure consistently.

A standard model trained without it sees contradictory patterns in the data and learns no consistent relationship:

```
Baseline (without hidden variable):  R² =  0.0096
Oracle   (with true hidden variable): R² = ~0.50
```

This gap is caused entirely by one missing variable. Model-T is designed to close it — without requiring full measurement.

---

## The Core Idea

Most missing-data methods ask: *what value is statistically plausible here, given similar rows?*

Model-T asks a different question: **given what I observed and what the output was, what must `h` have been?**

This is **model inversion** — using the fitted model to back-calculate the hidden variable from residuals:

```
h_estimated = (y_observed − base_prediction) / weight_of_h
```

Because the model is linear, this inversion is exact. The recovered value is clipped to the valid range, and Phase 2 retrains on the full dataset with all `h` values filled in.

This is mechanistically different from imputation, KNN filling, or EM. It uses the output to infer the cause — which standard gap-filling methods do not do.

---

## How It Works

### Step 1 — Partial labeling via rules

Domain knowledge is encoded as rules that assign `h` for rows meeting certain conditions:

```python
if laser > 50 and m1 == 1  →  h = 3
if laser < 20 and m1 == 0  →  h = 0
```

Rules use pandas `eval`-compatible condition strings. A row is labeled only once — the first matching rule applies. **15–30% coverage is sufficient.**

### Step 2 — Rule quality check (5 automated gates)

| Check | What it verifies |
|---|---|
| 1. Coverage | ≥ 15% of training rows labeled |
| 2. Matrix rank | Rank equals number of features |
| 3. Condition number | < 100 (well-conditioned system) |
| 4. Feature variance | Every feature has std > 0 in labeled rows |
| 5. h diversity | Hidden variable has ≥ 2 distinct values |

All 5 checks must pass before training proceeds.

### Step 3 — Phase 1: Train on labeled rows

```
y = f(x₁, x₂, ..., xₙ, h)
```

A linear model learns the relationship between observed features, the hidden variable, and the target.

### Step 4 — Recover `h` for unlabeled rows

```
h_estimated = (y_observed − base_prediction) / weight_of_h
```

Clipped to the valid range. Every training row now has an `h` value — either rule-assigned or inversion-recovered.

### Step 5 — Phase 2: Retrain on the full dataset

Retrained on all 800 rows with `h` filled in. Stronger than Phase 1 because it uses the complete training set.

### Step 6 — Interval prediction at test time

At test time `h` is unknown. The model finds the k most similar labeled training rows, uses their `h` distribution to estimate a range `[h_low, h_high]`, and propagates it through the model:

```
Output per test row: [target_low, target_high]
```

This produces calibrated uncertainty instead of a fragile point estimate.

---

## Pipeline

```
Full dataset
     │
     ├──────────────────────────────┐
     ▼                              │ Test set held out
Train / Test split                  │ (rules never applied here)
     │
     ▼
Apply rules to train set
  → ~30% labeled
  → ~70% unlabeled
     │
     ▼
Rule quality check (5 gates)
     │
     ▼
Phase 1 — Train on labeled rows only
     │
     ▼
Recover h via inversion
  → All training rows now have h
     │
     ▼
Phase 2 — Retrain on full training set
     │
     ▼
Similarity search on test rows
  → k nearest labeled neighbors
  → h range from neighbor h distribution
     │
     ▼
Output: [target_low, target_high] per test row
```

---

## Results

**Dataset:** 1,000 rows · 6 features · 1 hidden variable (4 discrete values) · 29.2% rule-labeled

### Training summary

| Metric | Value |
|---|---|
| Training rows | 800 |
| Rule-labeled rows | 234 (29.2%) |
| h recovered by inversion | 566 (70.8%) |
| Test rows | 200 |

### Benchmark comparison

| Method | Test R² | Interval output? |
|---|---|---|
| No hidden var (baseline) | 0.0096 | No |
| Mean imputation | -0.0921 | No |
| KNN imputation (k=10) | 0.0155 | No |
| MICE | 0.0118 | No |
| EM algorithm | -0.6286 | No |
| **Model-T (interval midpoint)** | **0.0251** | **Yes** |
| Oracle (true h known) | ~0.50 | No — reference only |

### Interval coverage

| Metric | Value |
|---|---|
| Test rows inside predicted interval | 169 / 200 |
| Coverage (PICP) | **84.5%** |

Model-T is the only method in this comparison that outputs a prediction interval. At test time, when `h` is unknown, an interval is the correct form of output — and 84.5% of true values fall inside it on held-out data.

---

## Why This Is Different from Standard Imputation

| | Standard imputation | Model-T |
|---|---|---|
| Missing variable | Filled statistically | Physically recovered via inversion |
| Uses domain knowledge | No | Yes — rule-based priming |
| Output | Single point estimate | Calibrated interval |
| Handles hidden causal drivers | No | Yes |
| Requires full labels | Yes | No — 15–30% sufficient |

---

## Comparison to Existing Approaches

**vs. Physics-Informed Neural Networks (PINN)**
PINNs require a complete governing equation encoded as a differentiable loss. Model-T requires only partial domain knowledge — rules covering 15–30% of data. Better choice when the physics is known partially but not fully.

**vs. Structural Equation Modelling (SEM)**
SEM requires a full causal graph with distributional assumptions upfront. Model-T requires only labeling rules with no distributional assumptions — simpler to deploy when the causal structure is partially understood.

**vs. Grey-box / Hybrid Models**
Similar in spirit. Grey-box models use a known partial equation as the structural component. Model-T uses rules instead, making it accessible without equation-level domain expertise.

**vs. Conformal Prediction**
Conformal prediction produces statistically guaranteed intervals on any model. A natural next step is combining both: use Model-T's inversion to recover `h`, then wrap with conformal prediction for certified coverage.

---

## Limitations

| Limitation | Detail |
|---|---|
| Linear relationship required | Inversion assumes `h` has a roughly linear effect on the output. Works for smooth or monotonic systems. |
| Single hidden variable | Inverting for multiple unknowns simultaneously is underdetermined. Current scope handles one. |
| Rules must be meaningful | Weak or conflicting rules lead to poor Phase 1 recovery. The quality checker catches the worst cases. |
| Coverage not statistically guaranteed | 84.5% is empirical. A conformal prediction wrapper is needed for formal guarantees. |

---

## Future Work

- Multiple hidden variables with regularized inversion
- Nonlinear models (neural networks, gradient boosting) with approximate inversion
- Conformal prediction wrapper for statistically certified coverage
- Automatic rule discovery — learn labeling conditions from data
- Benchmark against physics-informed baselines on real industrial datasets
- End-to-end training of partial labeling and recovery together

---

## Installation

```bash
git clone https://github.com/your-username/model-t
cd model-t
pip install -r requirements.txt
jupyter notebook model_t_pipeline.ipynb
```

**requirements.txt**
```
pandas
numpy
scikit-learn
scipy
jupyter
```

---

## Configuration

```python
CONFIG = {
    "dataset_path":     "data/Master_data.csv",
    "rules_path":       "rules/NEW_RULE.json",
    "output_path":      "data/results.csv",
    "test_size":        0.2,
    "random_seed":      42,
    "learning_rate":    0.001,
    "epochs":           300,
    "batch_size":       32,
    "k_neighbors":      10,
    "buffer_pct":       0.10,
    "similarity_sigma": 1.0,
}
```

> Call `np.random.seed(CONFIG["random_seed"])` before weight initialization for reproducible results.

---

## Project Structure

```
project/
│
├── Master_data.csv          # dataset: observed features + target
├── NEW_RULE.json            # rules for partial hidden variable labeling
├── model_t_pipeline.ipynb   # full pipeline
├── requirements.txt
└── README.md
```

---

## Adapting to Your Own Dataset

1. Prepare a CSV with observed features and a target column. The hidden variable column does not need to exist — the framework creates it.
2. Write rules that label at least 15% of training rows with known `h` values.
3. Run the quality checker — all 5 checks must pass before training proceeds.
4. Update `CONFIG` paths and run the notebook.

The framework is domain-agnostic. The hidden variable can be any latent quantity — a material property, a process setting, an environmental condition — as long as it influences the output and can be partially identified through rules.
