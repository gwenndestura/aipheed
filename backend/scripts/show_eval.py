"""Quick reader for data/processed/eval_results.json."""
import json

with open("data/processed/eval_results.json") as f:
    e = json.load(f)

print("=== eval_results.json baselines ===")
print(f"{'baseline':30s}  {'F1':>6s}  {'AUC':>6s}  {'Acc':>6s}  {'Prec':>6s}  {'Recall':>6s}")
for name, m in e.get("baselines", {}).items():
    o = m.get("overall", {})
    print(
        f"{name:30s}  "
        f"{o.get('weighted_f1', 0):6.3f}  "
        f"{o.get('roc_auc', 0):6.3f}  "
        f"{o.get('accuracy', 0):6.3f}  "
        f"{o.get('precision', 0):6.3f}  "
        f"{o.get('recall', 0):6.3f}"
    )

print()
print("Top-level keys:", list(e.keys()))
