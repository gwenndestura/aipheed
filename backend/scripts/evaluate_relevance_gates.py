"""
evaluate_relevance_gates.py
----------------------------
Score candidate news relevance gates against the hand-labelled sample.

Reads the workbook produced by build_relevance_sample.py, joins the labels back
to the cached ten-hypothesis vectors, and reports precision, recall and F1 for
each candidate gate -- plus how much of the corpus each one would keep.

Why more than one number matters: tightening any threshold raises precision by
construction. Only the rejected-pool arm shows what tightening costs, and only
the corpus-impact column shows whether a gate leaves enough articles for the
`limitedSignal` flag to mean anything. A gate at 99% precision that keeps four
articles a quarter is not an improvement.

Gates compared
--------------
    current          core_max >= 0.30                      (status quo)
    strict_XX        core_max >= 0.XX                      (raise the bar)
    noncore_block    core_max >= 0.30 AND noncore_max < T   (reject when the
                     non-food hypotheses also fire -- the all-entailment
                     collapse)
    margin_XX        core_max - noncore_max >= 0.XX        (require the food
                     reading to beat the off-topic reading)
    keyword_and      core_max >= 0.30 AND a keyword-bank hit
    combined         strict + margin

Usage
-----
    python scripts/evaluate_relevance_gates.py
    python scripts/evaluate_relevance_gates.py --min-labelled 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openpyxl import load_workbook  # noqa: E402

from app.ml.nlp.classifier import _KW_BY_TOPIC, CORE_TOPIC_IDS  # noqa: E402

OUTDIR = ROOT / "data" / "processed" / "relevance_eval"
FEATURES_PATH = OUTDIR / "sample_features.parquet"
WORKBOOK_PATH = OUTDIR / "relevance_labelling.xlsx"
REPORT_PATH = OUTDIR / "gate_evaluation.csv"
SCORED = ROOT / "data" / "processed" / "corpus_geocoded.parquet"

CORE_KEYWORDS = sorted({
    kw for tid, kws in _KW_BY_TOPIC.items() if tid in CORE_TOPIC_IDS for kw in kws
})


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def read_labels() -> pd.DataFrame:
    if not WORKBOOK_PATH.exists():
        raise SystemExit(
            f"No labelling workbook at {WORKBOOK_PATH}.\n"
            "Run scripts/build_relevance_sample.py first."
        )
    ws = load_workbook(WORKBOOK_PATH, data_only=True)["Label"]
    # Columns: # | Headline | Lead | Relevant? | Topic | Notes
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        answer = (row[3] or "").strip()
        rows.append({
            "row_no": int(row[0]),
            "label_raw": answer,
            "label": ("yes" if answer.startswith("Yes")
                      else "no" if answer.startswith("No")
                      else "unsure" if answer.startswith("Unsure")
                      else None),
            "topic_label": (row[4] or "").strip() or None,
            "notes": (row[5] or "").strip() or None,
        })
    return pd.DataFrame(rows)


def load_sample() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise SystemExit(f"No cached features at {FEATURES_PATH}.")
    feats = pd.read_parquet(FEATURES_PATH)
    labels = read_labels()
    merged = feats.merge(labels[["row_no", "label", "topic_label", "notes"]],
                         on="row_no", how="left")
    merged["text"] = (merged["title"].fillna("") + " "
                      + merged["summary"].fillna("")).str.lower()
    merged["kw_hit"] = merged["text"].apply(
        lambda t: any(kw in t for kw in CORE_KEYWORDS))
    return merged


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def gate_definitions() -> dict[str, tuple]:
    """
    Each gate is (predicate, needs_full_vector).

    `needs_full_vector` marks gates that read `noncore_max`. The stored corpus
    keeps only the winning score, so those cannot be applied to it and their
    corpus-impact column is left empty rather than reported as zero -- which
    would read as "this gate keeps nothing".
    """
    gates: dict[str, tuple] = {
        "current (>=0.30)": (lambda d: d["core_max"] >= 0.30, False),
    }
    for t in (0.50, 0.70, 0.90, 0.95):
        gates[f"strict (>={t:.2f})"] = (lambda d, t=t: d["core_max"] >= t, False)
    for t in (0.90, 0.80):
        gates[f"noncore_block (<{t:.2f})"] = (
            lambda d, t=t: (d["core_max"] >= 0.30) & (d["noncore_max"] < t), True)
    for m in (0.05, 0.20, 0.40):
        gates[f"margin (>={m:.2f})"] = (
            lambda d, m=m: (d["core_max"] >= 0.30)
            & ((d["core_max"] - d["noncore_max"]) >= m), True)
    gates["keyword_and"] = (
        lambda d: (d["core_max"] >= 0.30) & d["kw_hit"], False)
    gates["strict0.70 + margin0.20"] = (
        lambda d: (d["core_max"] >= 0.70)
        & ((d["core_max"] - d["noncore_max"]) >= 0.20), True)
    gates["strict0.50 + keyword"] = (
        lambda d: (d["core_max"] >= 0.50) & d["kw_hit"], False)
    return gates


def evaluate(sample: pd.DataFrame, corpus: pd.DataFrame | None) -> pd.DataFrame:
    """
    Precision, recall and F1 per gate.

    'Unsure' rows are dropped rather than assigned: forcing them either way
    would move the numbers without any evidence behind the move.
    """
    judged = sample[sample["label"].isin(["yes", "no"])].copy()
    judged["truth"] = judged["label"] == "yes"

    rows = []
    for name, (predicate, needs_vector) in gate_definitions().items():
        keeps = predicate(judged)
        tp = int((keeps & judged["truth"]).sum())
        fp = int((keeps & ~judged["truth"]).sum())
        fn = int((~keeps & judged["truth"]).sum())
        tn = int((~keeps & ~judged["truth"]).sum())

        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall and precision + recall else float("nan"))

        corpus_kept = None
        if corpus is not None and not needs_vector:
            corpus_kept = int(predicate(corpus).sum())

        rows.append({
            "gate": name,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 3) if precision == precision else None,
            "recall": round(recall, 3) if recall == recall else None,
            "f1": round(f1, 3) if f1 == f1 else None,
            "corpus_kept": corpus_kept,
        })
    return pd.DataFrame(rows).sort_values("f1", ascending=False, na_position="last")


def corpus_frame() -> pd.DataFrame | None:
    """
    The published corpus expressed in the same columns the gates read.

    Only gates that need `noncore_max` are unavailable here: the stored corpus
    keeps the winning score, not the whole vector. Those cells stay blank
    rather than being guessed at.
    """
    if not SCORED.exists():
        return None
    df = pd.read_parquet(SCORED)
    df = df[df["food_insecurity_score"].notna()].copy()
    df["core_max"] = df["food_insecurity_score"]
    df["noncore_max"] = float("nan")  # not stored; margin gates cannot be applied
    df["text"] = (df["title"].fillna("") + " " + df["summary"].fillna("")).str.lower()
    df["kw_hit"] = df["text"].apply(lambda t: any(kw in t for kw in CORE_KEYWORDS))
    return df


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0].strip())
    ap.add_argument("--min-labelled", type=int, default=40,
                    help="refuse to report below this many judged rows")
    args = ap.parse_args()

    sample = load_sample()
    judged = sample[sample["label"].isin(["yes", "no"])]
    unsure = int((sample["label"] == "unsure").sum())
    blank = int(sample["label"].isna().sum())

    print(f"sample rows      : {len(sample)}")
    print(f"labelled yes/no  : {len(judged)}")
    print(f"unsure (excluded): {unsure}")
    print(f"not yet labelled : {blank}")

    if len(judged) < args.min_labelled:
        raise SystemExit(
            f"\nOnly {len(judged)} rows judged; need at least {args.min_labelled} "
            "for the rates to mean anything. Label more of the workbook."
        )

    for arm, group in judged.groupby("arm"):
        rate = (group["label"] == "yes").mean()
        print(f"  {arm:9s}: {len(group):4d} judged, "
              f"{rate * 100:.1f}% genuinely relevant")

    print()
    report = evaluate(sample, corpus_frame())
    print(report.to_string(index=False))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)
    print(f"\nwrote {REPORT_PATH}")
    print(
        "\nRead the table with the rejected-pool arm in mind: any gate can buy "
        "precision by discarding more, and `corpus_kept` shows what that costs "
        "in articles. Margin gates show no corpus_kept because the stored "
        "corpus keeps only the winning score, not the full vector -- rescoring "
        "the corpus would fill that column in."
    )


if __name__ == "__main__":
    main()
