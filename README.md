# Model-T — Training with Partially Known Hidden Variables

> *You don't need the full dataset to be labeled. If you know the hidden variable for even a subset of rows, you can recover it for the rest — and train a complete model.*

---

## The Core Idea

In many real-world datasets, there is a variable that influences the output but is **not recorded for every row**. It might be expensive to measure, only observable under certain conditions, or simply missing for most of the dataset.

The standard reaction to this is to either drop those rows or ignore the variable entirely. Both approaches throw away useful information.

**Model-T takes a different approach:**

> If you know the hidden variable for *some* rows — even a small fraction — that is enough to train a model and recover the hidden variable for all the remaining rows.

This means a dataset that looks incomplete is actually fully trainable. You are not blocked by missing values. You use what you know to fill in what you don't.

---

## How It Works

### Step 1 — Partial labeling via rules

Domain knowledge is encoded as rules that assign the hidden variable value for rows that meet certain conditions:

```
if laser > 50 and m1 == 1  →  h = 3
if laser < 20 and m1 == 0  →  h = 0
```

These rules do not need to cover the whole dataset. In the example here, only **29.2% of training rows** were labeled this way. The rest had the hidden variable unknown.

### Step 2 — Train on the labeled subset

A model is trained using only the rows where the hidden variable is known:

```
y = f(x₁, x₂, ..., xₙ, h)
```

The model learns the relationship between all features — including the hidden variable — and the target output.

### Step 3 — Recover the hidden variable for unlabeled rows

This is the key step. Because the model is linear, it can be **inverted**. For any unlabeled row where `y` is observed but `h` is unknown:

```
h_estimated = (y_observed − base_prediction) / weight_of_h
```

The model tells us what `h` must have been to produce the observed output. The estimate is clipped to the valid range defined by the rules.

After this step, **every training row has an h value** — either from the rules directly, or recovered through the model.

### Step 4 — Retrain on the complete dataset

With all hidden variable values filled in, the model is retrained on the full training set. This second model is stronger because it has seen all 800 rows, not just the 234 rule-labeled ones.

### Step 5 — Range prediction for test rows

At test time, the hidden variable is still unknown. Rather than guessing a single value, the model finds the k most similar labeled training rows and uses their hidden variable values to estimate a plausible range `[h_low, h_high]`. This range is propagated through the model to produce a prediction interval `[y_low, y_high]`.

---

## Why Partial Knowledge Is Enough

The insight is that the hidden variable does not need to be measured for every row. As long as:

1. A subset of rows can be labeled with reasonable confidence (via rules or any other method)
2. The model can learn the relationship between the hidden variable and the output from that subset
3. The relationship is approximately linear or monotonic

...then the model itself becomes the tool for recovering the hidden variable everywhere else.

This turns what looks like an incomplete dataset into a fully trainable one.

---

## Pipeline

```
Full dataset
     │
     ├─────────────────────────────────────────────┐
     ▼                                             │
Train / Test split                                 │ Test set held out
     │                                             │ (never sees rules)
     ▼
Apply rules to train set
→ Some rows now have h known (29.2% in this example)
→ Most rows still have h unknown
     │
     ▼
Rule quality check
→ 5 automated checks: rank, condition number,
  coverage, variance, h diversity
     │
     ▼
Phase 1 — Train on labeled rows only
→ Model learns: y = f(observed features, h)
     │
     ▼
Recover h for unlabeled rows
→ Invert: h = (y − base_pred) / w_h
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
Predict range for test rows
→ Evaluate model at h_low and h_high
→ Output: target_low, target_high per row
```

---

## Results

Dataset: 1000 rows, 6 features, 1 hidden variable with 4 possible values.

| | |
|---|---|
| Training rows | 800 |
| Rule-labeled rows (h known from rules) | 234 (29.2%) |
| Rows with h recovered by model | 566 (70.8%) |
| Test rows | 200 |
| True target inside predicted range | 162 / 200 |
| **Coverage** | **81.0%** |

Starting from less than 30% labeled data, the framework recovers the full training set and achieves 81% range coverage on held-out test data.

---

## Project Structure

```
project/
│
├── data/
│   └── Master_data.csv          # dataset: observed features + target
│
├── rules/
│   └── NEW_RULE.json            # rules for partial hidden variable labeling
│
├── notebooks/
│   └── model_t_pipeline.ipynb   # full pipeline
│
├── requirements.txt
└── README.md
```

---

## Rules JSON Format

```json
{
  "variables": {
    "m5": {
      "range": [0, 3]
    }
  },
  "rules": [
    {
      "condition": "laser > 50 and m1 == 1",
      "assign": { "m5": 3 }
    },
    {
      "condition": "laser < 20 and m1 == 0",
      "assign": { "m5": 0 }
    }
  ]
}
```

Rules use pandas `eval`-compatible condition strings. A row is only labeled once — first matching rule applies, later rules are skipped for that row.

---

## Configuration

```python
CONFIG = {
    "dataset_path":  "data/Master_data.csv",
    "rules_path":    "rules/NEW_RULE.json",
    "output_path":   "data/results.csv",

    "test_size":     0.2,
    "random_seed":   42,       # used for both train/test split and numpy RNG

    "learning_rate": 0.001,
    "epochs":        300,
    "batch_size":    32,

    "k_neighbors":   10,       # neighbors used to estimate hidden variable range
    "buffer_pct":    0.10,     # 10% uncertainty buffer added to prediction range
}
```

---

## Installation

```bash
git clone https://github.com/your-username/model-t
cd model-t
pip install -r requirements.txt
jupyter notebook notebooks/model_t_pipeline.ipynb
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
2. Write rules that can label at least 15% of your training rows with known hidden variable values.
3. Run the quality checker — all 5 checks must pass before training.
4. Update CONFIG paths and run the notebook.

The framework is domain-agnostic. The hidden variable can represent anything — a material property, a process setting, an environmental condition — as long as it influences the output and can be partially identified through rules.

---

## Limitations

| | |
|---|---|
| Linear relationship required | Model inversion assumes h has a roughly linear effect on the output. Works for smooth or monotonic systems. |
| Single hidden variable | Inverting for multiple unknowns simultaneously is underdetermined. Current scope handles one hidden variable. |
| Rules must be meaningful | Weak or conflicting rules → poor Phase 1 model → poor recovery. The quality checker catches the worst cases. |
| Reproducibility | Call `np.random.seed(CONFIG["random_seed"])` before weight initialization to ensure identical results across runs. |

---

## Future Work

- Multiple hidden variables with regularized inversion
- Nonlinear models (neural networks, gradient boosting) with approximate inversion
- Automatic rule discovery — learn labeling conditions from data
- Conformal prediction for statistically calibrated coverage guarantees
- End-to-end training of partial labeling and recovery together

---

