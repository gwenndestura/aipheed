"""
rescore_corpus_relevance.py
----------------------------
Re-apply the relevance gate to the geocoded news corpus.

Needed because the margin gate reads the full ten-hypothesis entailment vector
and the corpus only ever stored the winning score. This run persists
`noncore_max` and `relevance_margin` alongside it, so the next gate change can
be evaluated from stored output instead of another full pass.

The original file is copied to corpus_geocoded.prerescore.parquet before
anything is written. Progress is checkpointed, so an interrupted run resumes
where it stopped rather than starting the whole pass again.

Usage
-----
    python scripts/rescore_corpus_relevance.py
    python scripts/rescore_corpus_relevance.py --batch 8
    python scripts/rescore_corpus_relevance.py --dry-run    # report, write nothing
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ml.nlp.classifier import (  # noqa: E402
    CORE_TOPIC_IDS,
    HYPOTHESES,
    PREMISE_CHARS,
    RELEVANCE_MARGIN,
    RELEVANCE_THRESHOLD,
    load_classifier,
)

CORPUS = ROOT / "data" / "processed" / "corpus_geocoded.parquet"
BACKUP = ROOT / "data" / "processed" / "corpus_geocoded.prerescore.parquet"
CHECKPOINT = ROOT / "data" / "processed" / "relevance_eval" / "_rescore_checkpoint.parquet"

TOPIC_IDS = list(HYPOTHESES.keys())
CORE_IDX = [i for i, t in enumerate(TOPIC_IDS) if t in CORE_TOPIC_IDS]
NONCORE_IDX = [i for i, t in enumerate(TOPIC_IDS) if t not in CORE_TOPIC_IDS]


def score_batch(clf, texts: list[str]) -> np.ndarray:
    """
    Entailment probabilities for a batch of articles: (n_articles, 10).

    Batching across articles as well as hypotheses is what makes a full pass
    finish in under an hour rather than three.
    """
    hyps = list(HYPOTHESES.values())
    premises, hypotheses = [], []
    for text in texts:
        premises.extend([text] * len(hyps))
        hypotheses.extend(hyps)

    enc = clf._tok(premises, hypotheses, return_tensors="pt",
                   truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        logits = clf._model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[:, clf._ent_idx].numpy()
    return probs.reshape(len(texts), len(hyps))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0].strip())
    ap.add_argument("--batch", type=int, default=6,
                    help="articles per forward pass (each expands to 10 pairs)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(CORPUS)
    # Must match score_article() exactly, or the corpus is scored on a
    # different premise from anything the gate was measured on.
    df["text"] = (
        df["title"].fillna("") + ". " + df["summary"].fillna("")
    ).str[:PREMISE_CHARS]
    print(f"corpus: {len(df)} articles | premise window {PREMISE_CHARS} chars")

    done: dict[str, np.ndarray] = {}
    if CHECKPOINT.exists():
        ck = pd.read_parquet(CHECKPOINT)
        done = {r["article_id"]: np.array([r[f"h_{t}"] for t in TOPIC_IDS])
                for _, r in ck.iterrows()}
        print(f"resuming: {len(done)} already scored")

    todo = df[~df["article_id"].isin(done)]
    print(f"to score: {len(todo)}")

    clf = load_classifier()
    if getattr(clf, "mode", None) != "xlm-roberta":
        raise SystemExit("NLI model did not load; refusing to rescore with the "
                         "keyword fallback.")

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(torch.get_num_threads())
    started = time.time()

    # Sort by length before batching. Padding runs to the longest pair in a
    # batch, so mixing a 40-token headline with a 512-token article made every
    # short one cost as much as the long one -- the first attempt measured
    # 0.4 articles/sec. Length-homogeneous batches remove most of that waste.
    todo = todo.assign(_len=todo["text"].str.len()).sort_values("_len")
    rows = list(todo.itertuples(index=False))

    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]
        probs = score_batch(clf, [c.text for c in chunk])
        for c, vec in zip(chunk, probs):
            done[c.article_id] = vec

        seen = start + len(chunk)
        if seen % (args.batch * 20) == 0 or seen >= len(rows):
            rate = seen / max(time.time() - started, 1e-6)
            eta = (len(rows) - seen) / max(rate, 1e-6) / 60
            print(f"  {seen}/{len(rows)}  {rate:.1f} art/s  ETA {eta:.0f} min",
                  flush=True)
            pd.DataFrame([
                {"article_id": aid, **{f"h_{t}": v for t, v in zip(TOPIC_IDS, vec)}}
                for aid, vec in done.items()
            ]).to_parquet(CHECKPOINT, index=False)

    matrix = np.vstack([done[a] for a in df["article_id"]])
    core = matrix[:, CORE_IDX]
    noncore = matrix[:, NONCORE_IDX]

    best_core_pos = core.argmax(axis=1)
    df["food_insecurity_score"] = core.max(axis=1).round(4)
    df["noncore_max"] = noncore.max(axis=1).round(4)
    df["relevance_margin"] = (df["food_insecurity_score"] - df["noncore_max"]).round(4)
    df["top_hypothesis"] = [TOPIC_IDS[CORE_IDX[i]] for i in best_core_pos]
    df["top_topic_name"] = [HYPOTHESES[t] for t in df["top_hypothesis"]]
    df["is_relevant"] = (
        (df["food_insecurity_score"] >= RELEVANCE_THRESHOLD)
        & (df["relevance_margin"] >= RELEVANCE_MARGIN)
    )

    kept = int(df["is_relevant"].sum())
    print()
    print(f"relevant under the new gate : {kept} of {len(df)} "
          f"({kept / len(df) * 100:.1f}%)")
    print(f"dropped                     : {len(df) - kept}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    if not BACKUP.exists():
        shutil.copy2(CORPUS, BACKUP)
        print(f"backed up original to {BACKUP.name}")

    df.drop(columns=["text"]).to_parquet(CORPUS, index=False)
    print(f"wrote {CORPUS}")


if __name__ == "__main__":
    main()
