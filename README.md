# Model-T
### Training with Partially Known Hidden Variables

> *Rule-Primed Inversion · Calibrated Interval Prediction · No Full Labels Required*

---

## The Problem This Solves

Many real-world systems — biological, industrial, physical — are governed by variables that cannot be measured consistently. The variable exists. It drives the output. But you cannot see it for most of your data.

A sensor reading that only fires under specific conditions. A molecular state that is measurable in a lab but not in the field. A process parameter that is logged only when an engineer manually records it. These are not edge cases — they are the norm in applied science and engineering.

The standard response is to either drop the variable entirely or fill it using statistical imputation. Both approaches treat the missing variable as a data problem. Model-T treats it as a physics problem: **if the variable drives the output, the output contains information about the variable.** Model-T extracts that information directly.

---

## Why Hidden Variables Matter Especially in Biological and Complex Systems

Biological systems are among the hardest to model precisely because they operate under layers of hidden state. Gene expression, receptor occupancy, intracellular signaling, metabolic flux — these are variables that govern measurable outcomes but are almost never observed directly in clinical or field data. The same is true for ecological systems, neurological systems, and many chemical processes.

The challenge is not just that the data is small — it is that the hidden variable creates what looks like random noise in the observed data. A model trained without it will see the same inputs produce wildly different outputs and conclude that the system is stochastic when it is not. It is deterministic under a variable that is invisible to the model.

**If you have partial knowledge — even for 15–30% of your data — about what that hidden variable looks like under certain conditions, Model-T can use that partial knowledge to recover approximate values for the rest, and then produce a calibrated range of possible outputs at prediction time.**

This matters especially when:
- You cannot afford to measure the hidden variable for every sample (expensive assays, destructive testing, delayed lab results)
- You have domain knowledge encoded as rules or thresholds — partial, not complete — about when the variable takes certain values
- A point prediction is not trustworthy without knowing the uncertainty introduced by the hidden state
- You are working with small datasets where statistical imputation introduces more noise than signal

Rather than pretending the hidden variable does not exist or guessing its value blindly, Model-T gives you a range: *"given what we observed and what we know, the true output lies somewhere between these two values."* That range is the honest answer when a hidden variable is involved.

---

## The Core Idea

Standard missing-data methods ask:
> *"What value is statistically plausible here, given similar rows?"*

Model-T asks a different question:
> *"Given what I observed as input and what the output actually was — what must the hidden variable have been?"*

This is **model inversion**. Because the model is linear, the inversion is exact:

```
h_estimated = (y_observed − prediction_without_h) / weight_of_h
```

The recovered value is clipped to the valid range and used to retrain a stronger model on the full dataset. At test time, when the hidden variable is unknown, Model-T searches for the most similar training rows whose hidden variable was either rule-labeled or inversion-recovered, extracts the range of plausible values, and propagates that range through the model to produce a **prediction interval** rather than a fragile point estimate.

---

## How It Works — Step by Step

### Step 1 — Encode Partial Knowledge as Rules

Domain knowledge is written as condition-formula pairs in a JSON file. Each rule says: *"when these conditions hold, the hidden variable equals this formula."*

```json
{
  "variables": {
    "m5": { "range": [0, 3], "type": "continuous" }
  },
  "rules": [
    {
      "name": "band0_m1_off_m2_off",
      "condition": "(m1 == 0) and (m2 == 0) and (laser < 4)",
      "formula": "laser * 0.75"
    },
    {
      "name": "band1_m1_on_m2_off",
      "condition": "(m1 == 1) and (m2 == 0) and (laser >= 4) and (laser < 8)",
      "formula": "(laser - 4) * 0.75"
    }
  ]
}
```

Rules use pandas `eval`-compatible syntax. A row is labeled by the first rule it matches. **15–30% coverage is sufficient** to run the full pipeline.

---

### Step 2 — Automated Rule Quality Check (5 Gates)

Before any training begins, the pipeline runs 5 automated checks to verify the rules produce usable labeled data:

| Check | What It Verifies |
|---|---|
| Coverage | ≥ 15% of training rows are labeled |
| Matrix rank | Rank equals number of features (no redundancy) |
| Condition number | < 100 — system is well-conditioned |
| Feature variance | Every feature has std > 0 in labeled rows |
| Hidden variable diversity | At least 2 distinct values of h in labeled rows |

All 5 must pass before training proceeds. This prevents silent failures from weak or conflicting rules.

---

### Step 3 — Phase 1: Train on Rule-Labeled Rows

A linear model is trained on the subset of rows where the hidden variable is known from rules:

```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + wₕ·h + b
```

This gives the model a first estimate of each feature's contribution, including the weight of the hidden variable `wₕ`.

---

### Step 4 — Recover the Hidden Variable via Inversion

Using the Phase 1 model, the hidden variable is back-calculated for every unlabeled training row:

```
h_estimated = (y_observed − (w₁x₁ + ... + wₙxₙ + b)) / wₕ
```

The result is clipped to the valid range defined in the rules JSON. Every training row now has an h value — either rule-assigned or inversion-recovered. No statistical assumptions are made about the distribution of h.

---

### Step 5 — Phase 2: Retrain on the Full Dataset

The model is retrained on all training rows with h filled in. Because Phase 2 uses the complete dataset rather than the 15–30% rule-labeled subset, it produces substantially stronger weight estimates.

---

### Step 6 — Interval Prediction at Test Time

At test time, h is unknown and cannot be recovered (the true output is what we are trying to predict). Instead, the model finds the `k` most similar labeled training rows using a weighted similarity search — weighted by feature importance from the Phase 2 model. The h values from those neighbors define a range `[h_low, h_high]`, which is propagated through the model:

```
target_low  = model.predict(h = h_low)
target_high = model.predict(h = h_high)
```

**Output per test row: `[target_low, target_high]`**

This is a calibrated uncertainty interval — not a confidence interval in the statistical sense, but an honest expression of what the model does not know because h is hidden.

---

## Results

**Dataset:** 10,000 rows · 6 features · 1 hidden variable · 29.9% rule-labeled  
**Split:** 8,000 train / 2,000 test

### Training Summary

| Metric | Value |
|---|---|
| Training rows | 8,000 |
| Rule-labeled rows | 2,394 (29.9%) |
| h recovered by inversion | 5,606 (70.1%) |
| Test rows | 2,000 |

### Rule Quality Check — All Passed

| Check | Result |
|---|---|
| Coverage ≥ 15% | ✅ 29.9% |
| Matrix rank = n_features | ✅ 6 / 6 |
| Condition number < 100 | ✅ 1.9 (excellent) |
| Feature variance > 0 | ✅ All features |

### Feature Weights Learned (Phase 2)

| Feature | Normalized Weight | Role |
|---|---|---|
| laser | 0.584 | Dominant predictor |
| m5 (hidden) | 0.539 | Second most important |
| m4 | 0.114 | Minor |
| m1 | 0.108 | Minor |
| m3 | 0.101 | Minor |
| m2 | 0.092 | Minor |

---

## Evaluation

Because Model-T produces intervals rather than point estimates, it must be evaluated differently from standard models. Two separate evaluation frameworks apply:

### Point Prediction Methods — Evaluated by R²

These methods predict a single value and are evaluated by how close that value is to the truth.

| Method | Test R² |
|---|---|
| No hidden variable (floor) | 0.4897 |
| Mean imputation | 0.4892 |
| MICE | 0.4897 |
| KNN imputation (k=10) | 0.4941 |
| Oracle (true h known) | ~0.50 |

### Interval Prediction — Model-T Evaluated by Coverage

Model-T solves a harder problem: predict a calibrated range that contains the true value. The correct metric is **coverage (PICP — Prediction Interval Coverage Probability)**.

| Metric | Value |
|---|---|
| Test rows inside predicted interval | 1,814 / 2,000 |
| **Coverage (PICP)** | **90.7%** |
| Average interval width (target) | 25.3 units |
| Median interval width (target) | 28.1 units |
| Average interval width (hidden h) | 2.05 units |

> R² and coverage are not interchangeable metrics. Comparing them directly is like judging a weather forecast that says "20–25°C" by asking whether 22.5° was the exact temperature. Model-T is not optimized for R² — it is optimized to give you a range that honestly reflects what is unknown.

---

## Why This Is Different from Standard Imputation

| | Standard Imputation | Model-T |
|---|---|---|
| How h is estimated | Statistically (from similar rows) | Physically (from the output, via inversion) |
| Uses domain knowledge | No | Yes — rules encode partial physics |
| Output at test time | Single point estimate | Calibrated interval |
| Handles hidden causal drivers | No | Yes |
| Requires full labels | Yes | No — 15–30% sufficient |
| What it communicates | A guess | A range with honest uncertainty |

---

## Comparison to Related Approaches

**vs. Physics-Informed Neural Networks (PINNs)**  
PINNs require a complete governing equation encoded as a differentiable loss term. Model-T requires only partial domain knowledge — rules covering 15–30% of data. It is the better choice when the physics is understood partially but not fully formalized.

**vs. Structural Equation Modelling (SEM)**  
SEM requires a full causal graph with explicit distributional assumptions. Model-T requires only labeling rules — no distributional assumptions, simpler to deploy when the causal structure is partially understood.

**vs. Conformal Prediction**  
Conformal prediction produces statistically guaranteed intervals on any model. A natural extension of Model-T is combining both: use inversion to recover h, then wrap with conformal prediction for formally certified coverage guarantees.

---

## Limitations

| Limitation | Detail |
|---|---|
| Linear relationship required | Inversion assumes h has a roughly linear effect on the output. Applicable to smooth or monotonic systems. |
| Single hidden variable | Inverting for multiple unknowns simultaneously is underdetermined. Current version handles one hidden variable. |
| Rules must carry real signal | Weak or contradictory rules produce poor Phase 1 recovery. The quality checker prevents the worst cases but cannot substitute for meaningful domain knowledge. |
| Coverage is empirical | 90.7% is measured on held-out data, not statistically guaranteed. A conformal wrapper is needed for formal guarantees. |

The linear constraint is the most significant current limitation. Many biological and physical systems are nonlinear — enzyme kinetics, receptor saturation, gene regulatory networks. Extending Model-T to nonlinear models through approximate inversion or gradient-based back-calculation is the most important direction for future work.

---

## Future Work

- Nonlinear inversion via gradient-based or approximate methods (neural networks, gradient boosting)
- Multiple hidden variables with regularized joint inversion
- Conformal prediction wrapper for statistically certified coverage
- Automatic rule discovery — learn labeling conditions from data rather than encoding them manually
- Benchmark against physics-informed baselines on real biological and industrial datasets
- Tighter interval prediction by learning the h-to-target mapping more precisely

---

## Installation

```bash
git clone https://github.com/your-username/model-t
cd model-t
pip install -r requirements.txt
jupyter notebook model_T_pipeline_REAL.ipynb
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

All pipeline settings are controlled through a single `CONFIG` dictionary at the top of the notebook:

```python
CONFIG = {
    "dataset_path":     "data/MASTERDATA.csv",
    "rules_path":       "rules/json_RULE.json",
    "output_path":      "data/model_T_results.csv",

    "test_size":        0.2,
    "random_seed":      42,

    "learning_rate":    0.001,
    "epochs":           300,
    "batch_size":       32,

    "k_neighbors":      10,
    "buffer_pct":       0.10,
}
```

---

## Project Structure

```


## Adapting to Your Own Dataset

1. **Prepare your CSV** with observed features and a target column. The hidden variable column does not need to exist — the pipeline creates it.

2. **Write rules** that label at least 15% of training rows with known h values. Rules must use `pandas eval`-compatible condition strings.

3. **Run the quality checker** — all 5 gates must pass before training proceeds.

4. **Update CONFIG paths** and run the notebook.

The framework is domain-agnostic. The hidden variable can be any latent quantity — a molecular concentration, a process parameter, an environmental condition, a physiological state — as long as it influences the output and can be partially characterized through rules derived from domain knowledge.

---

## The Broader Principle

Most real systems — biological, ecological, chemical, physical — do not operate in fully observable space. There is almost always a variable underneath that drives behavior and that we see only partially, or only under specific conditions, or only in retrospect.

Model-T is built on the observation that partial knowledge, properly used, is more powerful than it appears. If you know the relationship between the hidden variable and the output even approximately, and if you can identify its value for even a fraction of your data, you can recover it for the rest — not by guessing, but by reading it from the output itself.

The interval at the end is not a failure to predict precisely. It is the correct answer when something is genuinely unknown. A range that contains the truth 90% of the time is more useful than a point estimate that is confidently wrong.
