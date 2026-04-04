# Model-T

**Training with Partially Known Hidden Variables**

> Rule-Primed Inversion · Interval Prediction · No Full Labels Required

---

## What This Project Is

Model-T is an experimental ML framework for a specific and under-studied problem:

> *You have a dataset where one important variable is only measurable for a fraction of rows. Standard models fail because they see contradictory patterns. Can you recover that variable — and train a useful model — without labeling everything?*

The answer this project explores: **yes, if the system is roughly linear and you have partial domain knowledge.**

This is not a general-purpose ML library. It is a focused experiment on one idea — model inversion as a recovery mechanism for latent variables — with honest benchmarks showing where it works and where it falls short.

---

## The Problem

In many real-world systems, the relationship between inputs and outputs is governed by a variable that is expensive, delayed, or impossible to measure consistently. Call it the **Hidden Governor**.

A standard model trained without it encounters two completely different causal states in the same dataset and learns no consistent pattern. The result is not a weak model — it is a confused one.

```
Baseline (without hidden variable):  R² =  0.0096
Oracle   (with true hidden variable): R² = ~0.50
```

This gap exists because a single variable changes the entire input-output relationship. Model-T is an attempt to partially close that gap using inversion and domain rules.

---

## The Core Idea

Most missing-data methods ask: *what value is statistically plausible here, given similar rows?*

Model-T asks: *given what I observed and what the output was, what must `h` have been?*

This is **model inversion** — using the fitted model itself to back-calculate the hidden variable from residuals. It works because linear models can be algebraically reversed:

```
h_estimated = (y_observed − base_prediction) / weight_of_h
```

The estimate is clipped to the valid range from the rules. After this step, every training row gets an `h` value — either rule-assigned or inversion-recovered — and Phase 2 retrains on the full dataset.

This is the novel part. It is not imputation. It is not EM. It uses the output to infer the cause, which standard statistical gap-filling methods do not do.

---

## How It Works

### Step 1 — Partial labeling via rules

Domain knowledge is encoded as rules that assign `h` for rows meeting certain conditions:

```python
if laser > 50 and m1 == 1  →  h = 3
if laser < 20 and m1 == 0  →  h = 0
```

Rules use pandas `eval`-compatible condition strings. A row is labeled only once — the first matching rule applies. **15–30% coverage is sufficient** to prime the inversion.

### Step 2 — Rule quality check (5 automated gates)

Before any training begins, the labeled subset is validated:

| Check | What it verifies |
|---|---|
| 1. Coverage | ≥ 15% of training rows labeled |
| 2. Matrix rank | Rank equals number of features |
| 3. Condition number | < 100 (well-conditioned system) |
| 4. Feature variance | Every feature has std > 0 in labeled rows |
| 5. h diversity | Hidden variable has ≥ 2 distinct values |

All 5 checks must pass before training proceeds.

### Step 3 — Phase 1 training on labeled rows

A linear model is trained on the labeled subset:

```
y = f(x₁, x₂, ..., xₙ, h)
```

### Step 4 — Recover `h` for unlabeled rows

```
h_estimated = (y_observed − base_prediction) / weight_of_h
```

Clipped to the valid range. Every training row now has an `h` value.

### Step 5 — Phase 2 retraining on the full dataset

Model retrained on all rows with filled `h`. Stronger than Phase 1 because it uses the full training set.

### Step 6 — Range prediction at test time

At test time `h` is unknown. The model:

1. Finds the k most similar labeled training rows
2. Uses their `h` values to estimate a range `[h_low, h_high]`
3. Propagates this through the model → **prediction interval** `[y_low, y_high]`

This produces an interval instead of a point estimate — the appropriate output when a causal variable is unobserved.

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
  → h range from neighbor distribution
     │
     ▼
Output: [target_low, target_high] per test row
```

---

## Results

Dataset: 1,000 rows · 6 features · 1 hidden variable with 4 discrete values · 29.2% of training rows rule-labeled.

### Benchmark comparison

| Method | Test R² | Honest? | Interval? |
|---|---|---|---|
| No hidden var (baseline) | 0.0096 | ✅ | No |
| Mean imputation | -0.0921 | ✅ | No |
| KNN imputation (k=10) | 0.0155 | ✅ | No |
| MICE | 0.0118 | ✅ | No |
| EM algorithm *(fixed)* | -0.6286 | ✅ | No |
| **Model-T (interval midpoint)** | **0.0251** | ✅ | **Yes** |
| Oracle (true h known) | ~0.50 | reference only | — |

### Coverage

| Metric | Value |
|---|---|
| Test rows | 200 |
| Rows inside predicted interval | 169 / 200 |
| Coverage (PICP) | **84.5%** |

### Honest reading of these numbers

**The R² gain is small.** Model-T's interval midpoint scores 0.0251 vs a baseline of 0.0096. This is a real improvement but a modest one. The reason is fundamental: predicting a single point when `h` is unknown at test time is still a hard problem. Model-T does not solve that — it sidesteps it by outputting an interval instead.

**The coverage number (84.5%) is the more meaningful result.** No other baseline produces an interval at all. Model-T is the only method in this comparison that outputs uncertainty bounds, which is the appropriate response to an unobserved causal variable.

**Important caveat:** coverage alone is not sufficient to claim good calibration. Interval width (MPIW) has not been computed yet. A very wide interval can achieve high coverage trivially. Computing MPIW alongside PICP is a planned next step — see the Limitations section.


---

## What Is Genuinely Novel Here

1. **Inversion as recovery** — using a fitted linear model to back-calculate a hidden variable from training residuals. This is mechanistically different from imputation, EM, and distributional approaches.

2. **Rule-primed semi-supervision** — requiring only 15–30% domain-labeled rows instead of full labels or distributional assumptions.

3. **Interval output by design** — the framework treats `h` uncertainty as a first-class output rather than a nuisance to average away.


---

## Limitations

| Limitation | Detail |
|---|---|
| Linear relationship required | Model inversion assumes `h` has a roughly linear effect on the output. Nonlinear systems are not supported in the current version. |
| Single hidden variable | Inverting for multiple unknowns simultaneously is underdetermined. Current scope handles exactly one. |
| Rules must be meaningful | Weak or conflicting rules lead to poor Phase 1 recovery. The quality checker catches the worst cases but does not guarantee rule quality. |
| R² gain is small | Point-estimate performance (0.0251) is only marginally better than baseline (0.0096). The primary value is interval output, not point accuracy. |
| Interval width not yet measured | 84.5% PICP is reported without MPIW. Coverage without width is an incomplete calibration claim. |
| Coverage not statistically guaranteed | 84.5% is empirical on one test split. A conformal prediction wrapper would be needed for formal guarantees. |
| Hidden variable recovery not directly validated | The inversion step has not been evaluated against ground-truth `h` values. Since the data is synthetic and true `h` is known, this is measurable — it just hasn't been done yet. |
| No ablation study | Individual components (rules, inversion, Phase 2) have not been ablated. Their independent contributions are unverified. |

---

## What Is Missing (Planned)

These additions would make this a more complete evaluation:

- **MPIW** — average interval width, to pair with PICP and make coverage meaningful
- **HVRE** — hidden variable recovery error; confusion matrix of estimated vs true `h` for the 70.8% unlabeled rows
- **Ablation study** — remove rules / inversion / Phase 2 one at a time to isolate each component's contribution
- **Fairer baselines** — regression with missing indicator, random latent sampling
- **Nonlinear extension** — approximate inversion for neural networks or gradient boosting
- **Conformal prediction wrapper** — statistically certified coverage

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

The hidden variable can be any latent quantity that influences the output and can be partially identified through rules.
