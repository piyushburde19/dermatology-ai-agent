import re
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_FILE = Path("data/ood/external_ood_results.csv")

# Run evaluate_ood.py output should be saved separately.
# This script reads the manually exported result file.

if not OUTPUT_FILE.exists():
    print(f"Missing: {OUTPUT_FILE}")
    print("First save the evaluate_ood output into this CSV.")
    raise SystemExit(1)

df = pd.read_csv(OUTPUT_FILE)

print("=" * 60)
print("OOD SIMILARITY STATISTICS")
print("=" * 60)

print("\nOverall:")
print(df["similarity"].describe(percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]))

print("\nBy category:")
print(
    df.groupby("category")["similarity"]
    .agg(["count", "min", "mean", "median", "max"])
    .round(4)
)

print("\nCurrent OOD rejection rate:")
print(
    df.groupby("category")["is_ood"]
    .mean()
    .mul(100)
    .round(2)
    .astype(str)
    .add("%")
)