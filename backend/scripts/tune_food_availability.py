"""
scripts/tune_food_availability.py
---------------------------------
Optuna search over the pooled food-availability panel.

Tuning was meaningless at 120 rows; with ~4,000 it is not. Fixed parameters were
used up to this point, so this is the last untried legitimate lever.

Honest split. Quarters are ordered; the search sees only the EARLY folds and the
final numbers come from LATER folds it never touched. Selecting hyperparameters
on the same folds you report is optimistic, so both are printed and labelled --
the held-out figure is the one to quote.

    tune folds   : test quarters 9 .. 12
    report folds : test quarters 13 .. 14  (never seen by the search)

Persistence is recomputed on the identical folds so the comparison is like for
like.
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_food_availability import (  # noqa: E402
    GOV, MATCHED, NLP, SEASONAL, SERIES, load,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)
log = logging.getLogger("tune_food")

OUT = Path("data/processed/food_availability_tuned.json")
MIN_TRAIN = 8
N_TRIALS = 120
SEED = 42

FEATURE_SETS = {
    "series + seasonal": SERIES + SEASONAL,
    "ALL (series+seasonal+gov+NLP+matched)": SERIES + SEASONAL + GOV + NLP + MATCHED,
}


def folds(df: pd.DataFrame) -> list[tuple[list[str], str]]:
    qs = sorted(df["quarter"].unique())
    return [(qs[:i], qs[i]) for i in range(MIN_TRAIN, len(qs))]


def score(df: pd.DataFrame, cols: list[str], params: dict,
          fold_list) -> dict:
    acc, f1s, aucs, pers, pers4 = [], [], [], [], []
    for tr_q, te_q in fold_list:
        tr = df[df["quarter"].isin(tr_q)]
        te = df[df["quarter"] == te_q]
        if len(te) < 10 or tr["label_shock"].nunique() < 2:
            continue
        m = LGBMClassifier(**params).fit(tr[cols], tr["label_shock"])
        pred = m.predict(te[cols])
        acc.append(accuracy_score(te["label_shock"], pred))
        f1s.append(f1_score(te["label_shock"], pred, average="weighted", zero_division=0))
        if te["label_shock"].nunique() > 1:
            aucs.append(roc_auc_score(te["label_shock"], m.predict_proba(te[cols])[:, 1]))
        pers.append(accuracy_score(te["label_shock"], te["shock_lag1"].astype(int)))
        pers4.append(accuracy_score(te["label_shock"], te["shock_lag4"].astype(int)))
    return {"accuracy": float(np.mean(acc)) if acc else 0.0,
            "f1": float(np.mean(f1s)) if f1s else 0.0,
            "roc_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "persistence": float(np.mean(pers)) if pers else float("nan"),
            "seasonal_persistence": float(np.mean(pers4)) if pers4 else float("nan"),
            "n_folds": len(acc)}


def main() -> None:
    df = load()
    all_folds = folds(df)
    tune_folds, report_folds = all_folds[:-2], all_folds[-2:]
    log.info("folds: %d total | %d for search | %d held out for reporting",
             len(all_folds), len(tune_folds), len(report_folds))

    results = {}
    for name, cols in FEATURE_SETS.items():
        cols = [c for c in cols if c in df.columns]

        def objective(trial: optuna.Trial) -> float:
            params = dict(
                objective="binary", verbosity=-1, random_state=SEED,
                class_weight="balanced",
                n_estimators=trial.suggest_int("n_estimators", 100, 600),
                learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.2, log=True),
                num_leaves=trial.suggest_int("num_leaves", 4, 64),
                max_depth=trial.suggest_int("max_depth", 2, 8),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            )
            s = score(df, cols, params, tune_folds)
            # 0.6 accuracy + 0.4 AUC: accuracy is the comparison against
            # persistence, AUC is what a triage tool actually needs.
            auc = 0.5 if np.isnan(s["roc_auc"]) else s["roc_auc"]
            return 0.6 * s["accuracy"] + 0.4 * auc

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

        best = dict(objective="binary", verbosity=-1, random_state=SEED,
                    class_weight="balanced", **study.best_params)
        results[name] = {
            "best_params": study.best_params,
            "tuned_on_search_folds": score(df, cols, best, tune_folds),
            "held_out": score(df, cols, best, report_folds),
            "all_folds": score(df, cols, best, all_folds),
        }
        log.info("%s: search best %.4f", name, study.best_value)

    print("\n" + "=" * 94)
    print("TUNED — pooled food-availability panel")
    print("=" * 94)
    for name, r in results.items():
        print(f"\n{name}")
        for split in ("tuned_on_search_folds", "held_out", "all_folds"):
            s = r[split]
            tag = "  <-- quote this" if split == "held_out" else ""
            print(f"   {split:24s} folds={s['n_folds']}  acc={s['accuracy']:.4f}  "
                  f"F1={s['f1']:.4f}  AUC={s['roc_auc']:.4f}  "
                  f"lag1={s['persistence']:.4f}  lag4={s['seasonal_persistence']:.4f}  "
                  f"skill4={s['accuracy'] - s['seasonal_persistence']:+.4f}{tag}")
    print("\n" + "=" * 94)

    OUT.write_text(json.dumps(results, indent=2))
    log.info("saved -> %s", OUT)


if __name__ == "__main__":
    main()
