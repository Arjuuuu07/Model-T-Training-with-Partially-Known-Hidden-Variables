import numpy as np
import pandas as pd
import json


# LOAD RULES

with open(r"D:\project MODEL.T\rule.json\json_RULE.json") as f:
    rules = json.load(f)

rule_list = rules["rules"]


# CONFIG
N = 10000
np.random.seed(42)


# GENERATE DATA
df = pd.DataFrame({
    "m1": np.random.randint(0, 2, N),
    "m2": np.random.randint(0, 2, N),
    "m3": np.random.randint(0, 2, N),
    "m4": np.random.randint(0, 2, N),
    "laser": np.random.uniform(0, 20, N)
})


# SELECT 30% RULE ROWS
mask_rule = np.random.rand(N) < 0.30

df["m5"] = np.nan

# APPLY RULES FROM JSON

def apply_rules(row):
    local_vars = {
        "m1": row["m1"],
        "m2": row["m2"],
        "m3": row["m3"],
        "m4": row["m4"],
        "laser": row["laser"]
    }
    for rule in rule_list:
        if eval(rule["condition"], {}, local_vars):
            return eval(rule["formula"], {}, local_vars)
    return np.nan

# apply only to 30% rows
df.loc[mask_rule, "m5"] = df[mask_rule].apply(apply_rules, axis=1)


# ADD 10% NOISE IN RULE ROWS

mask_noise = mask_rule & (np.random.rand(N) < 0.10)

noise = np.random.uniform(-0.5, 0.5, N)
df.loc[mask_noise, "m5"] += noise[mask_noise]

# clip to valid range
df["m5"] = np.clip(df["m5"], 0, 3)


# FILL REMAINING 70% RANDOM

mask_random = df["m5"].isna()
df.loc[mask_random, "m5"] = np.random.uniform(0, 3, mask_random.sum())


# TARGET  (FIXED FORMULA)

target_noise = np.random.normal(0, 2.5, N)

df["target"] = (
    60
    + 8   * df["m5"]
    - 2.5 * (df["m1"] + df["m2"] + df["m3"] + df["m4"])
    + 1.2 * df["laser"]
    + target_noise
)


# SAVE
df.to_csv(r"D:\project MODEL.T\dataset\new files\dataset.csv", index=False)


# INFO
print(df.head())
print("\nTotal:", len(df))
print("Rule rows (~30%):", mask_rule.sum())
print("Noise rows (10%):", mask_noise.sum())
print("Random fill rows:", mask_random.sum())

print("\nm5 stats:")
print(f"  Rule-row m5 range : [{df.loc[mask_rule,'m5'].min():.4f}, {df.loc[mask_rule,'m5'].max():.4f}]")
print(f"  Overall m5 mean   : {df['m5'].mean():.4f}")
print(f"  Overall m5 std    : {df['m5'].std():.4f}")

print("\ntarget stats:")
print(f"  Range : [{df['target'].min():.2f}, {df['target'].max():.2f}]")
print(f"  Mean  : {df['target'].mean():.2f}")
print(f"  Std   : {df['target'].std():.2f}")

print("\nCorrelations with target:")
print(df[["m1","m2","m3","m4","laser","m5"]].corrwith(df["target"]).round(4))