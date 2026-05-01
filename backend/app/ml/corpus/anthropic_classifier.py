"""
app/ml/corpus/anthropic_classifier.py
----------------------------------------
Anthropic API-powered food insecurity relevance classifier.

PURPOSE
-------
Keyword-based filters (rss_fetcher.py) cast a wide net but produce false
positives — articles about Batangas tourism, Cavite tech parks, or Rizal
politics slip through because they mention province names + weather. This
classifier uses Claude claude-haiku-4-5 to make a final relevance judgment on
each article title (and summary when available), ensuring only genuinely
food-insecurity-relevant articles enter the training corpus.

TWO-PASS DESIGN
---------------
Pass 1 — Batch keyword pre-filter (instant, no API cost):
  Reject articles with zero food-insecurity keyword signals.
  ~60–70% of articles eliminated without any API call.

Pass 2 — Claude claude-haiku-4-5 classification (batched, ~$0.0003/article):
  For remaining articles, ask Claude to score 0–4 food insecurity relevance
  using a structured prompt. Articles scoring < 2 are marked irrelevant.

SCORING RUBRIC (communicated to Claude in system prompt)
-----------------------------------------------------------
  4 — Directly about food insecurity: prices, hunger, malnutrition,
      food assistance, crop damage, fish kill, rice supply crisis.
  3 — Clearly proximate: typhoon/flood + agriculture, poverty + food,
      OFW remittance loss, employment + food access.
  2 — Contextually relevant: agriculture news, fishermen livelihood,
      food production, feeding programs.
  1 — Tangential: disaster relief (non-food), general agriculture business.
  0 — Irrelevant: politics, crime, tourism, infrastructure, entertainment.

Threshold: score >= 2 → food_relevant = "Y"

COST ESTIMATE
-------------
claude-haiku-4-5 pricing: $0.25/MTok input, $1.25/MTok output
Typical batch: 100 articles × ~80 tokens each = 8,000 input tokens → $0.002
Full 60,000-article corpus → ~$1.20 total

SETUP
-----
Set in backend/.env:
    ANTHROPIC_API_KEY=sk-ant-...

Usage:
    from app.ml.corpus.anthropic_classifier import classify_food_relevance
    df = classify_food_relevance(articles_df)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"  # fast + cheap for classification
BATCH_SIZE = 50        # articles per Claude API call
REQUEST_DELAY = 0.5    # seconds between batches
MAX_TOKENS_OUT = 512   # short responses only

# ---------------------------------------------------------------------------
# Pre-filter keyword sets (instant, no API cost)
# ---------------------------------------------------------------------------

_PREFILTER_REJECT = frozenset([
    # Pure non-food topics → safely reject before calling Claude
    "basketball", "volleyball", "sports", "election", "candidate", "mayor",
    "governor", "president", "senator", "congress", "tourism", "hotel",
    "resort", "mall", "real estate", "condo", "subdivision", "infrastructure",
    "expressway", "toll", "road", "bridge" , "airport", "seaport",
    "crime", "murder", "drug bust", "arrested", "police", "shabu", "firearm",
    "hospital", "covid vaccine", "vaccination", "healthcare",
    "beauty queen", "filmfest", "concert", "entertainment",
    "technology", "tech park", "industrial park", "factory",
    "solar panel", "power plant", "energy", "electricity",
])

_PREFILTER_REQUIRE_ANY = frozenset([
    # At least one of these must be present for the article to advance to Claude
    "food", "rice", "bigas", "pagkain", "gutom", "hunger", "presyo",
    "price", "supply", "shortage", "inflation", "harvest", "ani",
    "crop", "palay", "agriculture", "malnutrition", "poverty", "kahirapan",
    "relief", "ayuda", "assistance", "flood", "typhoon", "bagyo",
    "fish", "isda", "fishermen", "livelihood", "kadiwa", "4ps",
    "feeding", "nutrition", "fertilizer", "farm",
])

# ---------------------------------------------------------------------------
# System + user prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a food insecurity expert analyst specializing in the Philippines, \
specifically the CALABARZON region (Batangas, Cavite, Laguna, Quezon Province, Rizal).

Your task: classify news article titles for food insecurity relevance.

Scoring rubric (0–4):
  4 = Directly about food insecurity — food prices, hunger, malnutrition, \
food shortage, food assistance, crop damage, fish kill, rice supply crisis, \
feeding programs, poverty + food.
  3 = Clearly proximate driver — typhoon/flood + agriculture, drought + crops, \
OFW job/remittance loss, unemployment + food access, ASF pork ban, oil spill + fishermen.
  2 = Contextually relevant — agriculture/fisheries news, food production support, \
government food programs (Kadiwa, NFA, 4Ps), commodity price reports.
  1 = Tangential — general calamity/disaster without food angle, rural livelihood \
without food-specific signal.
  0 = Irrelevant — politics, crime, tourism, infrastructure, business/tech, sports, \
entertainment, healthcare unrelated to nutrition.

Return ONLY a JSON array of objects with "id" and "score" fields.
No explanation needed. Example:
[{"id": 0, "score": 4}, {"id": 1, "score": 0}, {"id": 2, "score": 2}]
"""


def _build_batch_prompt(batch: list[dict]) -> str:
    lines = []
    for item in batch:
        title = item.get("title", "")[:200]
        summary = (item.get("summary") or "")[:150]
        combined = title + (f" | {summary}" if summary else "")
        lines.append(f'[{item["_idx"]}] {combined}')
    return "Classify these CALABARZON news articles:\n\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-filter
# ---------------------------------------------------------------------------

def _passes_prefilter(title: str, summary: str) -> bool:
    """Return True if article should be sent to Claude for classification."""
    text = (title + " " + summary).lower()
    # Reject if only rejected keywords found and no food signal
    has_food_signal = any(kw in text for kw in _PREFILTER_REQUIRE_ANY)
    return has_food_signal


# ---------------------------------------------------------------------------
# Claude API caller
# ---------------------------------------------------------------------------

def _classify_batch(batch: list[dict], client: Any) -> dict[int, int]:
    """
    Call Claude to classify a batch of articles.
    Returns {idx: score} dict.
    """
    import json
    prompt = _build_batch_prompt(batch)
    try:
        response = client.messages.create(
            model=CLASSIFIER_MODEL,
            max_tokens=MAX_TOKENS_OUT,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.content[0].text.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        results = json.loads(content)
        return {int(r["id"]): int(r["score"]) for r in results}
    except Exception as exc:
        logger.warning("Claude classification error: %s", exc)
        # On error, default all to score 2 (pass-through with uncertainty)
        return {item["_idx"]: 2 for item in batch}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_food_relevance(
    df: pd.DataFrame,
    score_threshold: int = 2,
    force_reclassify: bool = False,
) -> pd.DataFrame:
    """
    Classify articles for food insecurity relevance using Claude claude-haiku-4-5.

    Adds three columns to the DataFrame:
      claude_fi_score    : int   — 0–4 food insecurity relevance score
      claude_fi_relevant : str   — "Y" | "N"
      claude_fi_label    : str   — human-readable category label

    Parameters
    ----------
    df               : DataFrame with 'title' and optionally 'summary' columns
    score_threshold  : minimum score for "Y" verdict (default: 2)
    force_reclassify : re-run Claude even if claude_fi_score already present

    Returns
    -------
    DataFrame with added classification columns
    """
    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set in .env — skipping Claude classification")
        df["claude_fi_score"] = -1
        df["claude_fi_relevant"] = "?"
        df["claude_fi_label"] = "unclassified"
        return df

    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed — run: pip install anthropic")
        df["claude_fi_score"] = -1
        df["claude_fi_relevant"] = "?"
        df["claude_fi_label"] = "unclassified"
        return df

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # ── Determine which rows need classification ──────────────────────────
    if "claude_fi_score" not in df.columns:
        df["claude_fi_score"] = -1
        df["claude_fi_relevant"] = "?"
        df["claude_fi_label"] = "unclassified"

    needs_classification = df[df["claude_fi_score"] == -1].index
    if not force_reclassify:
        to_classify = list(needs_classification)
    else:
        to_classify = list(df.index)

    logger.info("Claude classifier: %d articles to classify", len(to_classify))

    # ── Pass 1: keyword pre-filter ────────────────────────────────────────
    passed_prefilter: list[int] = []
    for idx in to_classify:
        row = df.loc[idx]
        title = str(row.get("title") or "")
        summary = str(row.get("summary") or "")
        if _passes_prefilter(title, summary):
            passed_prefilter.append(idx)
        else:
            df.at[idx, "claude_fi_score"] = 0
            df.at[idx, "claude_fi_relevant"] = "N"
            df.at[idx, "claude_fi_label"] = "pre-filter:no_food_signal"

    logger.info(
        "Pre-filter: %d/%d articles pass to Claude",
        len(passed_prefilter), len(to_classify),
    )

    # ── Pass 2: Claude batch classification ───────────────────────────────
    total_classified = 0
    for batch_start in range(0, len(passed_prefilter), BATCH_SIZE):
        batch_idxs = passed_prefilter[batch_start: batch_start + BATCH_SIZE]
        batch = []
        for i, idx in enumerate(batch_idxs):
            row = df.loc[idx]
            batch.append({
                "_idx": i,
                "title": str(row.get("title") or ""),
                "summary": str(row.get("summary") or ""),
            })

        scores = _classify_batch(batch, client)

        for i, idx in enumerate(batch_idxs):
            score = scores.get(i, 2)
            relevant = "Y" if score >= score_threshold else "N"
            label = {
                4: "direct_food_insecurity",
                3: "proximate_driver",
                2: "contextual_relevant",
                1: "tangential",
                0: "irrelevant",
            }.get(score, "unknown")
            df.at[idx, "claude_fi_score"] = score
            df.at[idx, "claude_fi_relevant"] = relevant
            df.at[idx, "claude_fi_label"] = label

        total_classified += len(batch_idxs)
        logger.info(
            "Claude: classified %d/%d articles",
            total_classified, len(passed_prefilter),
        )
        time.sleep(REQUEST_DELAY)

    # ── Summary ───────────────────────────────────────────────────────────
    relevant_count = (df["claude_fi_relevant"] == "Y").sum()
    total = len(df)
    logger.info(
        "Claude classification complete: %d/%d relevant (%.1f%%)",
        relevant_count, total, relevant_count / total * 100 if total else 0,
    )

    return df


# ---------------------------------------------------------------------------
# Convenience: classify a raw list of article dicts
# ---------------------------------------------------------------------------

def classify_articles_list(
    articles: list[dict],
    score_threshold: int = 2,
) -> list[dict]:
    """
    Classify a list of article dicts (title + summary).
    Returns the same list with 'claude_fi_score', 'claude_fi_relevant',
    and 'claude_fi_label' fields added.
    """
    df = pd.DataFrame(articles)
    df = classify_food_relevance(df, score_threshold=score_threshold)
    return df.to_dict(orient="records")
