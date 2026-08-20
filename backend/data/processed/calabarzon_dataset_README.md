# CALABARZON Food-Insecurity News Dataset

**Rows:** 664 articles · **Provinces:** {'Batangas': 215, 'Laguna': 147, 'Cavite': 146, 'Quezon': 92, 'Rizal': 64}
**Distinct cities/municipalities:** 67 of 142 · **barangay-level rows:** 10
**By source:** {'corpus_recall': 598, 'strict_reanalysis': 66}

## What this is
News articles that provide usable evidence about food insecurity (or its clear
determinants) in Region IV-A (CALABARZON). Every row satisfies BOTH inclusion
conditions: (1) a substantive food-insecurity connection, and (2) a geographic
tie to a CALABARZON province / city / municipality / barangay. General CALABARZON
news (politics, crime, sports, entertainment, food-culture) is excluded.

## Source pool (relevance faithful to the thesis)
Union of two relevance-scored pools, both from the thesis's zero-shot XLM-RoBERTa
NLI classifier (10 food-insecurity hypotheses):
  * `strict_reanalysis` — the reanalysis pipeline's retained HIGH/MEDIUM set
    (NLI + food-anchor + dimension-evidence + geo-verification gates).
  * `corpus_recall` — additional relevance-passing CALABARZON articles the strict
    pipeline had excluded, recovered from the geocoded corpus so genuine coverage
    is not lost. (`data_source` column records which pool each row came from.)

## Quality gates applied to every row (reproducible, no LLM)
  1. **Food-anchor** — a concrete foodstuff/agri/hunger/nutrition/fishery/food-
     program term (`FOOD_ANCHOR` lexicon, shared with `precision_pass.py`).
  2. **Food-insecurity topic** — matches >=1 specific thesis topic (hunger, prices,
     malnutrition, crop loss, assistance, ...); drops articles that mention food
     only incidentally.
  3. **CALABARZON geography** — has a province and is NOT an article whose subject
     is another region (competing other-region cue with no CALABARZON LGU → dropped:
     e.g. Mandaue, Bulacan, Pangasinan, "Bay Area" USA).
  4. **Negative filter** — drops incidental food/farm words in non-food contexts:
     energy ("solar/wind farm", megawatt deals), foreign locations, and food-culture
     (recipes, restaurants, "101: getting to know").
  5. **Syndication dedup** — same story under a near-identical title collapsed to one
     copy (strict version kept).
Every dropped row and its failing gate is logged to
`calabarzon_dataset_dropped_audit.csv` (removed this run: {'no_food_anchor+no_food_insecurity_topic': 229, 'other_region_subject+no_food_anchor+no_food_insecurity_topic': 85, 'no_food_insecurity_topic': 67, 'no_food_anchor': 53, 'off_topic_energy_foreign_or_culture+no_food_anchor+no_food_insecurity_topic': 31, 'other_region_subject': 22, 'other_region_subject+no_food_anchor': 17, 'off_topic_energy_foreign_or_culture': 15, 'needs_review_weak_signal': 12, 'off_topic_energy_foreign_or_culture+no_food_anchor': 8, 'other_region_subject+off_topic_energy_foreign_or_culture+no_food_anchor+no_food_insecurity_topic': 6, 'no_calabarzon_province+other_region_subject': 6, 'off_topic_energy_foreign_or_culture+no_food_insecurity_topic': 4, 'other_region_subject+off_topic_energy_foreign_or_culture': 3, 'no_calabarzon_province+other_region_subject+no_food_anchor': 1, 'other_region_subject+no_food_insecurity_topic': 1, 'no_calabarzon_province': 1, 'other_region_subject+off_topic_energy_foreign_or_culture+no_food_anchor': 1}).

## Geography (142-LGU enhancement)
Re-geocoded with the full **142-LGU PSGC gazetteer** geocoder (province → city/
municipality → barangay), with disambiguation (ambiguous/surname/feature names
require a locality cue) and a non-CALABARZON conflict guard.

## Columns
title, publication_date, news_source, author*, url, content_lead*, province,
city_municipality, barangay, relevance_tier (HIGH/MEDIUM), food_security_dimension
(A–F), food_insecurity_category, food_insecurity_topics, event_type,
affected_commodity*, affected_population*, is_direct_food_insecurity,
relevance_summary, relevance_reason*, relevance_score, match_level, needs_review,
data_source, article_id.  (* populated for strict_reanalysis rows; sparse for
corpus_recall rows.)

`needs_review` = True for the softest rows (only indirect poverty/livelihood/supply
signals) — 0 of 664 rows, for an optional eyeball.

## Known limitations (scope notes)
- **Residual precision (~99%):** the gates are lexical, so ~1 row may survive on an
  incidental token (e.g. a "Cordillera vegetable prices" story pre-tagged to a
  CALABARZON province). Retained for reproducibility; the audit CSV lists all drops.
- **Coverage ceiling:** 67/142 LGUs have a qualifying article. Targeted collection
  via two independent global news indexes (GDELT, Event Registry) confirmed most
  remaining municipalities have no food-insecurity news — a data-availability limit.
  See `calabarzon_lgu_coverage_matrix.csv` (covered / mentioned-only / absent).
- *`author` and full `content` were not retained by the historical fetchers
  (title + lead only); barangay tagging is limited because barangays are usually
  named deeper in bodies than the stored lead.
