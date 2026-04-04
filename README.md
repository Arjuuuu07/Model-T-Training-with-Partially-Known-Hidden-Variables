# Model-T

**Training with Partially Known Hidden Variables**

> Rule-Primed Inversion · Calibrated Interval Prediction · No Full Labels Required

---

## The Core Idea

Most missing-data methods treat a hidden variable as a statistical gap to fill. **Model-T treats it as a physical quantity to recover.**

> *Standard imputation asks: what value is plausible here, given similar rows?*
>
> **Model-T asks: given what I observed and what the output was, what must `h` have been?**

This inversion approach — using the model itself to recover the hidden variable from training residuals — is fundamentally different from MICE, KNN imputation, mean imputation, and EM. None of those methods understand that `h` causally governs the output. Model-T does.

---

## The Problem

In many real-world systems, the relationship between inputs and outputs is governed by a variable that is expensive, delayed, or impossible to measure consistently — the **Hidden Governor**.

A standard ML model trained without this variable encounters two completely different causal states in the same dataset and finds no consistent pattern:

```
Baseline (without hidden variable):  R² =  0.0096   ← model effectively fails
Oracle   (with true hidden variable): R² = ~0.50     ← strong performance
```

This gap is explained entirely by a single missing variable. Model-T is designed to close it — without requiring full labels.

---

## Results

Tested on 1,000 rows · 6 features · 1 hidden variable with 4 discrete values · 29.2% of training rows rule-labeled.

| Method | Test R² | Honest? | Interval output? |
|---|---|---|---|
| No hidden var (baseline) | 0.0096 | ✅ Yes | No |
| Mean imputation | -0.0921 | ✅ Yes | No |
| KNN imputation (k=10) | 0.0155 | ✅ Yes | No |
| MICE | 0.0118 | ✅ Yes | No |
| EM algorithm *(fixed)* | -0.6286 | ✅ Yes | No |
| **Model-T (interval midpoint)** | **0.0251** | ✅ **Yes** | **Yes — 84.5% coverage** |
| Oracle (true h known) | ~0.50 | Reference only | — |

**Model-T is the only method that consistently beats the no-hidden-var baseline while producing calibrated prediction intervals.**

> **Why does fixed EM collapse to R² = −0.63?**EM performs worse than predicting the mean. This reveals a genuine limitation: EM is a parameter estimation algorithm, not a prediction algorithm. At test time, with `y` unknown, there is no principled way to assign `h`. Model-T solves this via similarity-based range estimation.

---

## How It Works

### Step 1 — Partial labeling via rules

Domain knowledge is encoded as rules that assign the hidden variable for rows meeting certain conditions:

```python
if laser > 50 and m1 == 1  →  h = 3
if laser < 20 and m1 == 0  →  h = 0
```

Rules use pandas `eval`-compatible condition strings. A row is labeled only once — the **first matching rule** applies. Rules do not need to cover the full dataset; **15–30% coverage is sufficient**.

### Step 2 — Rule quality check (5 automated gates)

Before any training begins, the labeled subset is validated:

| Check | What it verifies |
|---|---|
| 1. Coverage | ≥ 15% of training rows labeled |
| 2. Matrix rank | Rank equals number of features (no redundant structure) |
| 3. Condition number | < 100 (well-conditioned system) |
| 4. Feature variance | Every feature has std > 0 in labeled rows |
| 5. h diversity | Hidden variable has ≥ 2 distinct values |

All 5 checks must pass before training proceeds.

### Step 3 — Phase 1 training on labeled rows

A linear model is trained on the labeled subset only:

```
y = f(x₁, x₂, ..., xₙ, h)
```

The model learns the relationship between all observed features, the hidden variable, and the target output.

### Step 4 — Recover `h` for unlabeled rows

Because the model is linear, it can be **inverted**. For any row where `y` is observed but `h` is unknown:

```
h_estimated = (y_observed − base_prediction) / weight_of_h
```

The estimate is clipped to the valid range defined by the rules. After this step, **every training row has an h value** — either rule-assigned or model-recovered.

### Step 5 — Phase 2 retraining on the full dataset

With all hidden variable values filled in, the model is retrained on the complete training set. This second model is stronger because it has seen all rows, not just the rule-labeled fraction.

### Step 6 — Range prediction at test time

At test time, `h` is unknown. Rather than guessing a single value, the model:

1. Finds the **k most similar labeled training rows** (by observed features)
2. Uses their `h` values to estimate a plausible range `[h_low, h_high]`
3. Propagates this range through the model to produce a **prediction interval** `[y_low, y_high]`

This prevents the confident-but-wrong errors of single-point predictors. **84.5% coverage achieved on held-out test data.**

---

## Pipeline Overview

```
Full dataset
     │
     ├──────────────────────────────────────┐
     ▼                                      │ Test set held out
Train / Test split                          │ (rules never applied here)
     │
     ▼
Apply rules to train set
  → ~30% of rows: h known
  → ~70% of rows: h unknown
     │
     ▼
Rule quality check  (5 automated checks)
  → Coverage · Rank · Condition number · Variance · h diversity
     │
     ▼
Phase 1 — Train on labeled rows only
  → y = f(observed features, h)
     │
     ▼
Recover h for unlabeled rows
  → h = (y − base_pred) / w_h
  → Clip to valid range
  → All training rows now have h
     │
     ▼
Phase 2 — Retrain on full training set
  → Stronger model, full data
     │
     ▼
Similarity search on test rows
  → k nearest labeled neighbors
  → h range = [10th pct, 90th pct] of neighbor h values
     │
     ▼
Predict interval for test rows
  → Evaluate at h_low and h_high
  → Output: target_low, target_high per row
```

---

## Reference Experiment Stats

| Metric | Value |
|---|---|
| Total rows | 1,000 |
| Training rows | 800 |
| Rule-labeled rows | 234 (29.2%) |
| h recovered by model | 566 (70.8%) |
| Test rows | 200 |
| True target inside predicted interval | 169 / 200 |
| Coverage | **84.5%** |

---

## Why This Is Different from Standard Imputation

| | Standard ML | Model-T |
|---|---|---|
| Missing variable | Ignored or filled with mean | Physically recovered via inversion |
| Uses domain knowledge | No | Yes (rule-based priming) |
| Output | Single point prediction | Calibrated interval |
| Handles hidden causal drivers | No | Yes |
| Requires full labels | Yes | No — 15–30% is sufficient |
| Causal awareness | All variables treated symmetrically | `h` explicitly modeled as governor of `y` |

---

## Comparison to Existing Approaches

**vs. Physics-Informed Neural Networks (PINN)**
PINNs embed physical laws directly into the loss function and achieve higher R² when the governing equation is fully known. Model-T requires only partial domain knowledge (rules covering 15–30% of data) rather than a complete equation — making it the better choice when the physics is partially known or too complex to encode as a differentiable loss.

**vs. Structural Equation Modelling (SEM)**
SEM handles latent variables well but requires a full causal graph with distributional assumptions specified upfront. Model-T requires only labeling rules and no distributional assumptions — easier to deploy in industrial settings where the causal structure is partially understood.

**vs. Grey-box / Hybrid Models**
Similar in spirit — both combine physical knowledge with data-driven fitting. Grey-box models typically use a known partial equation as the structural component. Model-T uses rules instead, making it accessible without equation-level domain expertise.

**vs. Conformal Prediction**
Conformal prediction produces statistically guaranteed coverage intervals on any model. Model-T's 84.5% coverage is empirical, not formally guaranteed. The recommended next step: use Model-T's inversion to recover `h`, then wrap predictions with conformal prediction for certified intervals.

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
    # Paths
    "dataset_path":  "data/Master_data.csv",
    "rules_path":    "rules/NEW_RULE.json",
    "output_path":   "data/results.csv",

    # Split
    "test_size":     0.2,

    # Reproducibility
    "random_seed":   42,

    # Training
    "learning_rate": 0.001,
    "epochs":        300,
    "batch_size":    32,

    # Similarity / range prediction
    "k_neighbors":   10,       # neighbors used to estimate h range
    "buffer_pct":    0.10,     # 10% uncertainty buffer on prediction range
    "similarity_sigma": 1.0,
}
```

> **Reproducibility note:** Call `np.random.seed(CONFIG["random_seed"])` before weight initialization for identical results across runs.

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

**Rules format** (`NEW_RULE.json`): uses pandas `eval`-compatible condition strings. A row is labeled only once — the first matching rule applies.

---

## Adapting to Your Own Dataset

1. Prepare a CSV with observed features and a target column. The hidden variable column does not need to exist — the framework creates it.
2. Write rules that label at least 15% of training rows with known hidden variable values.
3. Run the quality checker — all 5 checks must pass before training proceeds.
4. Update `CONFIG` paths and run the notebook.

The framework is **domain-agnostic**. The hidden variable can be anything that influences the output and can be partially identified through rules — a material property, a process setting, an environmental condition, or any latent quantity.

---

## Limitations

| Limitation | Detail |
|---|---|
| Linear relationship required | Model inversion assumes `h` has a roughly linear effect on the output. Works for smooth or monotonic systems. |
| Single hidden variable | Inverting for multiple unknowns simultaneously is underdetermined. Current scope handles one. |
| Rules must be meaningful | Weak or conflicting rules lead to poor Phase 1 recovery. The quality checker catches the worst cases. |
| Coverage not guaranteed | 84.5% is empirical. A conformal prediction wrapper is needed for statistical guarantees. |

---

## Future Work

- Multiple hidden variables with regularized inversion
- Nonlinear models (neural networks, gradient boosting) with approximate inversion
- Conformal prediction wrapper for statistically certified coverage guarantees
- Automatic rule discovery — learn labeling conditions from data
- Benchmark against physics-informed baselines on real industrial datasets
- End-to-end training of partial labeling and recovery together
