# Model-T — Training with Partially Known Hidden Variables

> *You don't need the full dataset to be labeled. If you know the hidden variable for even a subset of rows, you can recover it for the rest — and train a complete model.*

---

## The Problem

In many real-world systems, the relationship between inputs and outputs is governed by a variable that is expensive, delayed, or impossible to measure consistently. Call it the **Hidden Governor**.

Consider cardiac monitoring as a motivating example. Blood flow (*y*) is driven by mechanical features like heart rate and valve diameter. But these features are modulated by hormone levels — a chemical governor that shifts the entire input-output relationship. Hormone levels are only available for ~30% of cases due to the cost and delay of blood tests.

A standard ML model, trained without hormone data, encounters two completely different "chemical states" in the same dataset and finds no consistent pattern. The result: negative R² scores, not because the data is bad, but because a critical variable is invisible.

```
Baseline (without hidden variable):   R² = -0.17  → model fails
Oracle   (with true hidden variable):  R² =  0.50  → strong performance
```

This gap — between a failing model and a strong one — is explained entirely by a single missing variable. Model-T is designed to close that gap.

---

## The Solution

Model-T treats the hidden variable not as missing data, but as a **recoverable quantity**.

If domain knowledge lets you label even a fraction of your rows with confidence, the model can infer the hidden variable for every other row by mathematical inversion. A dataset that looks incomplete becomes fully trainable.

In the cardiac example: rather than requiring a blood draw for every observation, Model-T acts as a **virtual sensor** — inferring the hidden hormonal state from purely mechanical observations, because the relationship between them is approximately linear.

> **Note:** The cardiac example is illustrative. Model-T is domain-agnostic. The hidden variable can represent any latent quantity — a material property, a process setting, an environmental condition — as long as it influences the output and can be partially identified through rules.

---

## How It Works

### Step 1 — Partial labeling via rules

Domain knowledge is encoded as rules that assign the hidden variable value for rows meeting certain conditions:

```
if laser > 50 and m1 == 1  →  h = 3
if laser < 20 and m1 == 0  →  h = 0
```

Rules do not need to cover the full dataset. In the reference experiment, only **29.2% of training rows** were labeled this way.

### Step 2 — Phase 1 training on labeled rows

A linear model is trained on the labeled subset:

```
y = f(x₁, x₂, ..., xₙ, h)
```

The model learns the relationship between all observed features, the hidden variable, and the target output.

### Step 3 — Recover the hidden variable for unlabeled rows

Because the model is linear, it can be inverted. For any row where *y* is observed but *h* is unknown:

```
h_estimated = (y_observed − base_prediction) / weight_of_h
```

The estimate is clipped to the valid range defined by the rules. After this step, every training row has an *h* value — either rule-assigned or model-recovered.

### Step 4 — Phase 2 retraining on the full dataset

With all hidden variable values filled in, the model is retrained on the complete training set. This second model is stronger because it has seen all rows, not just the rule-labeled fraction.

### Step 5 — Range prediction at test time

At test time, the hidden variable is unknown. Rather than guessing a single value, the model:

1. Finds the *k* most similar labeled training rows
2. Uses their hidden variable values to estimate a plausible range `[h_low, h_high]`
3. Propagates this range through the model to produce a **prediction interval** `[y_low, y_high]`

This prevents the confident-but-wrong errors of single-point predictors. In high-stakes settings, knowing that a prediction falls in `[0, 3]` rather than being a fragile point estimate is often more useful.

---

## Pipeline

```
Full dataset
     │
     ├─────────────────────────────────────┐
     ▼                                     │ Test set held out
Train / Test split                         │ (never sees rules)
     │
     ▼
Apply rules to train set
→ ~30% of rows: h known
→ ~70% of rows: h unknown
     │
     ▼
Rule quality check
→ 5 automated checks: rank, condition number,
  coverage, variance, h diversity
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

## Results

Dataset: 1000 rows, 6 features, 1 hidden variable with 4 possible values.

| Metric | Value |
|---|---|
| Training rows | 800 |
| Rule-labeled rows (h known) | 234 (29.2%) |
| Rows with h recovered by model | 566 (70.8%) |
| Test rows | 200 |
| True target inside predicted range | 162 / 200 |
| **Coverage** | **81.0%** |

Starting from less than 30% labeled data, the framework recovers the full training set and achieves 81% range coverage on held-out test data.

---

## Why This Is Different from Standard Imputation

| | Standard ML (drop/impute) | Model-T |
|---|---|---|
| Missing variable | Ignored or filled with mean | Physically recovered via inversion |
| Uses domain knowledge | No | Yes (rule-based priming) |
| Output | Single point prediction | Calibrated interval |
| Handles hidden causal drivers | No | Yes |
| Requires full labels | Yes | No — 15–30% is sufficient |

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

Rules use pandas `eval`-compatible condition strings. A row is labeled only once — the first matching rule applies.

---

## Configuration

```python
CONFIG = {
    "dataset_path":  "data/Master_data.csv",
    "rules_path":    "rules/NEW_RULE.json",
    "output_path":   "data/results.csv",

    "test_size":     0.2,
    "random_seed":   42,

    "learning_rate": 0.001,
    "epochs":        300,
    "batch_size":    32,

    "k_neighbors":   10,      # neighbors used to estimate h range
    "buffer_pct":    0.10,    # 10% uncertainty buffer on prediction range
}
```

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
jupyter
```

---

## Adapting to Your Own Dataset

1. Prepare a CSV with observed features and a target column. The hidden variable column does not need to exist — the framework creates it.
2. Write rules that label at least **15% of training rows** with known hidden variable values.
3. Run the quality checker — all 5 checks must pass before training proceeds.
4. Update `CONFIG` paths and run the notebook.

The framework is domain-agnostic. The hidden variable can be anything that influences the output and can be partially identified through rules.

---

## Limitations

| Limitation | Detail |
|---|---|
| Linear relationship required | Model inversion assumes h has a roughly linear effect on the output. Works for smooth or monotonic systems. |
| Single hidden variable | Inverting for multiple unknowns simultaneously is underdetermined. Current scope handles one. |
| Rules must be meaningful | Weak or conflicting rules lead to poor Phase 1 recovery. The quality checker catches the worst cases. |
| Reproducibility | Call `np.random.seed(CONFIG["random_seed"])` before weight initialization for identical results across runs. |

---

## Future Work

- Multiple hidden variables with regularized inversion
- Nonlinear models (neural networks, gradient boosting) with approximate inversion
- Automatic rule discovery — learn labeling conditions from data
- Conformal prediction for statistically calibrated coverage guarantees
- End-to-end training of partial labeling and recovery together
