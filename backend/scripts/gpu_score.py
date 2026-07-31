"""
scripts/gpu_score.py
--------------------
Self-contained GPU scorer for the CALABARZON recovery pool. Runs anywhere
with a CUDA GPU (Google Colab free T4, RunPod, Lambda, etc.) — no aiPHeed
package imports, so it works on a bare machine.

Scores each article's (title + summary) against the 10 HungerGist
hypotheses using joeddav/xlm-roberta-large-xnli, taking the MAX
P(entailment) across hypotheses — identical math to the CPU pipeline
(app/ml/nlp/classifier.py), so GPU and CPU scores are interchangeable and
merge cleanly by article_id.

Colab runbook
-------------
1. Colab → Runtime → Change runtime type → T4 GPU.
2. Upload gdelt_calabarzon_recovered_enriched.parquet (Files pane) and this
   script (or paste the SCORING cell below).
3. Run:
     !pip -q install transformers torch sentencepiece pandas pyarrow
     !python gpu_score.py --src gdelt_calabarzon_recovered_enriched.parquet \\
             --out recovered_scores.parquet --batch 64
4. Download recovered_scores.parquet, drop it in backend/, and tell me —
   I merge it into data/raw/checkpoints/xlmr_scores.parquet and assemble.

Usage (local/remote):
  python gpu_score.py --src IN.parquet --out OUT.parquet [--batch 64]
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "joeddav/xlm-roberta-large-xnli"
RELEVANCE_THRESHOLD = 0.30
HYPOTHESES = {
    "T1": "This article is about food prices, food supply problems, or difficulty accessing food",
    "T2": "This article is about hunger, malnutrition, or nutrition and feeding programs",
    "T3": "This article is about government food assistance, rice subsidies, or relief distribution",
    "T4": "This article is about poverty, unemployment, or economic hardship of families",
    "T5": "This article is about roads, transport, or storage problems affecting food supply",
    "T6": "This article is about farmland loss, crop damage, or reduced harvests",
    "T7": "This article is about evacuation or displacement of families due to disaster",
    "T8": "This article is about strikes, protests, or unrest disrupting food or livelihoods",
    "T1b": "This article is about fish kills, fishing bans, or aquaculture losses",
    "T9": "This article is about overseas Filipino workers or remittances supporting families",
}
TOPIC_IDS = list(HYPOTHESES.keys())
HYP_TEXTS = list(HYPOTHESES.values())
N_HYP = len(TOPIC_IDS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64, help="articles per batch")
    ap.add_argument("--resume", action="store_true",
                    help="skip article_ids already present in --out")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA GPU found — this will be slow. Use a GPU runtime.")
    dtype = torch.float16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model.to(device).eval()
    ent_idx = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}.get("entailment", 2)

    df = pd.read_parquet(args.src)
    done: set[str] = set()
    prior: list[dict] = []
    if args.resume:
        try:
            pv = pd.read_parquet(args.out)
            prior = pv.to_dict("records")
            done = set(pv["article_id"])
            print(f"resume: {len(done)} already scored")
        except Exception:
            pass
    todo = df[~df["article_id"].isin(done)].reset_index(drop=True)
    print(f"scoring {len(todo)} articles on {device} (batch={args.batch})")

    rows = list(prior)
    t0 = time.time()
    for start in range(0, len(todo), args.batch):
        chunk = todo.iloc[start:start + args.batch]
        premises, pairs = [], []
        for _, r in chunk.iterrows():
            text = f"{r.get('title') or ''}. {r.get('summary') or ''}"[:512]
            premises.append(text)
        # Build article × hypothesis pairs
        prem_rep, hyp_rep = [], []
        for p in premises:
            prem_rep.extend([p] * N_HYP)
            hyp_rep.extend(HYP_TEXTS)
        enc = tok(prem_rep, hyp_rep, return_tensors="pt", truncation=True,
                  max_length=512, padding=True).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits.float(), dim=-1)[:, ent_idx]
        probs = probs.view(len(premises), N_HYP)          # (batch, 10)
        best_p, best_i = probs.max(dim=1)
        for j, (_, r) in enumerate(chunk.iterrows()):
            score = float(best_p[j])
            tid = TOPIC_IDS[int(best_i[j])]
            rows.append({
                "article_id": r["article_id"],
                "food_insecurity_score": round(score, 4),
                "is_relevant": bool(score >= RELEVANCE_THRESHOLD),
                "top_hypothesis": tid,
                "top_topic_name": HYPOTHESES[tid],
            })
        if (start // args.batch) % 20 == 0 and start:
            n = len(rows) - len(prior)
            rate = n / max(time.time() - t0, 1)
            rel = sum(1 for x in rows if x["is_relevant"])
            pd.DataFrame(rows).to_parquet(args.out, index=False)
            print(f"  {len(rows)}/{len(df)} scored | {rel} relevant | "
                  f"{rate:.0f} art/s", flush=True)

    pd.DataFrame(rows).to_parquet(args.out, index=False)
    rel = sum(1 for x in rows if x["is_relevant"])
    print(f"DONE: {len(rows)} scored, {rel} relevant -> {args.out}")


if __name__ == "__main__":
    main()
