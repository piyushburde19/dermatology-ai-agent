import pandas as pd

df = pd.read_csv("tests/test_set_ood_results.csv")

print()
print("THRESHOLD ANALYSIS")
print("=" * 70)
print("Threshold | Rejected | Accepted | Accepted Accuracy")
print("-" * 70)

for threshold in [0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37]:

    accepted = df[df["similarity"] >= threshold]
    rejected = df[df["similarity"] < threshold]

    if len(accepted) > 0:
        accuracy = (
            accepted["prediction"].str.upper()
            == accepted["actual_class"].str.upper()
        ).mean() * 100
    else:
        accuracy = 0

    print(
        f"{threshold:.2f}      | "
        f"{len(rejected):8d} | "
        f"{len(accepted):8d} | "
        f"{accuracy:.2f}%"
    )