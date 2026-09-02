"""
scripts/probe_lead_effect.py
----------------------------
Does adding the lead change the NLI score enough to break scale comparability?

Scores every audited row that HAS real lead text twice — title-only and
title+lead — and reports the paired difference. If the shift is small and
unbiased, a hybrid column (title+lead where available) is safe. If it is
systematic, mixing the two would bias every province-quarter whose article mix
happens to include lead-bearing rows.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.nlp.classifier import load_classifier, score_article  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("probe_lead")

DATASET = Path("data/processed/calabarzon_food_insecurity_dataset.parquet")
OUT = Path("data/processed/_lead_effect_probe.parquet")


def main() -> None:
    df = pd.read_parquet(DATASET)
    df["lead"] = df["content_lead"].fillna("").astype(str)
    have = df[df["lead"].str.len() > 0].copy()
    log.info("rows with real lead text: %d of %d", len(have), len(df))

    clf = load_classifier()
    rows = []
    for n, (_, r) in enumerate(have.iterrows(), 1):
        title = str(r["title"] or "")
        t_only = score_article(clf, title, "")
        t_lead = score_article(clf, title, r["lead"])
        rows.append({
            "article_id": r["article_id"],
            "relevance_tier": r["relevance_tier"],
            "lead_len": len(r["lead"]),
            "score_title": t_only["food_insecurity_score"],
            "score_title_lead": t_lead["food_insecurity_score"],
            "hyp_title": t_only["top_hypothesis"],
            "hyp_title_lead": t_lead["top_hypothesis"],
        })
        if n % 20 == 0 or n == len(have):
            log.info("probed %d/%d", n, len(have))

    out = pd.DataFrame(rows)
    out["delta"] = out["score_title_lead"] - out["score_title"]
    out.to_parquet(OUT, index=False)

    print("\n===== PAIRED TITLE vs TITLE+LEAD =====")
    print(out[["score_title", "score_title_lead", "delta"]].describe().round(4).to_string())
    print(f"\nmean delta      : {out['delta'].mean():+.4f}")
    print(f"median delta    : {out['delta'].median():+.4f}")
    print(f"mean |delta|    : {out['delta'].abs().mean():.4f}")
    print(f"corr(title, title+lead): {out['score_title'].corr(out['score_title_lead']):.4f}")
    print(f"lead raises score in    : {int((out['delta'] > 0).sum())} of {len(out)}")
    print(f"top hypothesis flips in : {int((out['hyp_title'] != out['hyp_title_lead']).sum())} of {len(out)}")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
