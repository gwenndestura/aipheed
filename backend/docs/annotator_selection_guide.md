# Annotator & Validator Selection — aiPHeed Dataset Credibility
### CALABARZON Food-Insecurity News Dataset + Structured Government Indicators

**Context.** DOST Region IV-A is no longer the single validating body for this study
(the earlier annotation guide and pre-interview guide built around them have been
retired). This document replaces that plan: it recommends who should validate each of
aiPHeed's two data streams, and — more importantly — the process that actually makes a
"gold standard" credible, since the process matters more than any one reviewer's title.

---

## 1. The two credibility problems are different

The **news corpus** is subjective: a human has to judge whether an article is really
about food insecurity, which of six dimensions it fits, and how severe it is. Its
credibility rests on **who judges** and **how consistently they agree**.

The **government/official indicators** (PSA food CPI & commodity prices, BSP
remittances, PAGASA rainfall, SWS hunger incidence, DOST-FNRI ENNS/FIES nutrition) are
already authoritative at the source — PSA, BSP, PAGASA and DOST-FNRI are the country's
official statistical agencies, and SWS is the long-running, widely-cited private survey
house those agencies' own communications routinely reference. The risk here isn't "is
the number believable" — it's whether the pipeline **collected, joined, and thresholded
it correctly**. That's a fidelity check, not a labeling task.

---

## 2. What actually makes an annotator credible

Four things, regardless of which institution someone comes from:

1. **Domain qualification** — documented expertise in food security, nutrition, or the
   specific indicator they're reviewing, not just general seniority.
2. **Independence** — not a member of the team that built the classifier or chose the
   thresholds; someone with no stake in the system scoring well.
3. **Plurality** — never a single rater. Two or more independent reviewers per item is
   what lets you *measure* agreement instead of just asserting it.
4. **An audit trail** — identity, date, and rationale recorded per decision, so a thesis
   panel (or anyone auditing the dataset later) can see who decided what and why.

Sections 5 and 6 turn these into a concrete process; Sections 3–4 name concrete
candidates for each dataset.

---

## 3. Recommended panel — news dataset

The retired guide's four questions actually call for **two different kinds of
expertise**. Q2–Q4 (dimension, severity) need **food-security/nutrition domain
knowledge**. Q1 ("is this really about food insecurity") implicitly also needs a check
the guide never separated out: **is the underlying article itself a credible piece of
reporting** (a real, sourced news story — not an opinion piece, a press release
repackaged as news, or a low-quality/clickbait outlet), which is a media-credibility
judgment, not a food-security one. Splitting these two into different reviewer roles is
worth doing explicitly.

| Role | Candidate | Why they fit | What they check |
|---|---|---|---|
| Food-security / nutrition domain reviewer (primary — need ≥2 independent people in this role) | **UPLB Institute of Human Nutrition and Food (IHNF)**, College of Human Ecology, Los Baños, Laguna — inside CALABARZON, active in regional nutrition-sensitivity research | Academic nutrition/food-security specialists with direct CALABARZON research experience | Q1 relevance, Q2 dimension (A–F), severity |
| Food-security domain reviewer (parallel/alternate ask) | **National Nutrition Council (NNC) Region IV-A** | The nutrition-coordinating government body for the region; complements rather than duplicates UPLB's academic lens | Same as above — run in parallel so the study isn't dependent on a single institution again |
| Production/fishery specialist (for Dimension **A** calls specifically) | **DA Regional Field Office IV-A** and/or **BFAR Region IV-A** | Technical authority on crop damage, harvest loss, fish kills, ASF — the exact events that populate Dimension A | Confirms A-vs-B and A-vs-D boundary calls (production loss vs. price spike vs. disruption) |
| News-source credibility reviewer | A professional fact-checker (**VERA Files** or **Rappler**, both IFCN-verified Philippine signatories) or a journalism faculty member (e.g., UP College of Mass Communication) | Domain expertise is in *source reliability*, not food policy — the check the old guide never separated out | New "Q0": is this a genuine, sourced report from a credible outlet, not opinion/clickbait/duplicate |
| Adjudicator | Thesis adviser or panel member | Independent of both rater pools, final authority | Breaks ties when domain reviewers disagree; sets the final gold label |

---

## 4. Recommended panel — government/structured indicator dataset

Two distinct checks, not one:

**(a) Source-fidelity audit.** Confirm that a sample of collected values actually match
what PSA, BSP, PAGASA, SWS and DOST-FNRI published for that period. This doesn't need
an external expert — it's a documentable spot-check: pick a handful of quarters,
pull the agency's own published bulletin, and compare. Do it as two independent passes
(e.g., the researcher plus one other team member working from the raw bulletins
separately) so it isn't just one person's read of the source.

**(b) Label/threshold correctness.** The composite `label_stress` (SWS hunger + PSA
food-CPI deviation, anchored to FIES/ENNS) and the High/Medium/Low severity cutoffs
encode a judgment call about what counts as a "food-insecure quarter." Getting this
wrong biases everything downstream, so it needs two kinds of sign-off:

| Role | Candidate | Checks |
|---|---|---|
| Substantive/domain correctness | **NNC Region IV-A** or **UPLB IHNF** (same pool as Section 3) | Do the label composite and severity cutoffs match real-world crisis thresholds a nutrition body would recognize? |
| Methodological correctness | Thesis statistics adviser / panel member | Is the composite label leakage-free, are the cutoffs and weak-supervision anchoring statistically defensible? |
| Supplementary source reactors (only if the study adds the candidate sources already listed as gaps — DA/BFAR damage reports, DSWD DROMIC/4Ps, DILG, PhilRice, LGU Nutrition Action Plans, RDRRMC bulletins) | The issuing agency's own regional office — **DSWD Field Office IV-A (DROMIC desk)**, **DA-RFO IV-A**, **BFAR IV-A** | Confirms the new source is being read and used the way the issuing agency intends |

---

## 5. The process that actually "deeply ensures correctness"

Who you pick matters less than this — a single well-credentialed reviewer is still a
single point of failure. The design below is what produces a defensible gold standard:

**Never a single rater.** Every item — news article or sampled indicator-quarter — gets
at least two independent reviewers from the domain-reviewer pool, working without
seeing each other's calls (and, for a clean accuracy read on a subset, without seeing
the machine's proposed tag either).

**Pilot before scaling.** Run the guide on a small batch first — roughly 10 articles, or
2–3 quarters of indicators — before committing reviewer time to the full set. Use it to
catch ambiguous wording in the guide itself (the existing A-vs-B / A-vs-D / B-vs-F
mix-up notes in `build_review_sheet.py` are exactly this kind of fix, already applied
once).

**Measure agreement, don't just assert it.** Compute Cohen's kappa (two raters) or
Fleiss' kappa (three or more) per question — keep/drop, dimension, severity — and report
the number in the thesis. "An expert reviewed it" is a much weaker credibility claim
than "two independent experts agreed at κ = 0.78."

**Route disagreement to a named adjudicator**, not a majority vote. The adviser/panel
role in Section 3 exists specifically for this — every disagreement should have a
recorded resolution and a one-line rationale, not just a tie-break.

**Keep an audit trail.** `build_review_sheet.py` already has "Reviewer initials" and
"Reviewer notes" columns in the output workbook. Running it once per independent
reviewer (saved under a reviewer-specific filename, e.g.
`review_sheet_reviewerA.xlsx` / `_reviewerB.xlsx`) gets you multi-rater data
without building new tooling — just don't let the two copies get merged before both are
complete.

**Document credentials in a short annex.** For each annotator: title, years of relevant
experience, institutional affiliation. This is standard for a thesis validity/reliability
section, and it's the thing an examiner is most likely to actually ask about.

---

## 6. Practical recruitment note

Treat NNC Region IV-A and UPLB IHNF as **parallel primary asks** rather than picking one
— the review-sheet format is agency-agnostic, so use whichever responds first, and
having two live options avoids being dependent on a single institution's availability
the way the DOST 4A plan was. If neither is reachable in time, the internal-team + IAA
design in Section 5 still produces a defensible gold standard using co-researchers as
the raters — note that as a limitation (no external panel sign-off) rather than skipping
the agreement statistics, which are what actually carry the credibility argument.

---

## Sources

Institutional facts above were verified against each organization's own site as of
August 2026:

- [National Nutrition Council – Region IV-A Profile](https://nnc.gov.ph/luzon-region/region-iv-a-profile/)
- [UPLB Institute of Human Nutrition and Food (IHNF)](https://ihnf.uplb.edu.ph/)
- [UPLB OVCRE — CALABARZON Regional Food Innovation Center](https://ovcre.uplb.edu.ph/research/our-projects/article/19533-research-and-extension-services-through-the-calabarzon-regional-food-innovation-center)
- [VERA Files — Fact Check](https://verafiles.org/fact-check)
- [Why fact-check — and why Rappler and VERA Files (CMFR)](https://cmfr-phil.org/in-context/why-fact-check-and-why-rappler-and-vera-files/)
- [DSWD DROMIC — Disaster Response Operations Monitoring, Information and Communication](https://dromic.dswd.gov.ph/)
- [Philippine Statistics Authority](https://psa.gov.ph/)
- [Philippine Standard Geographic Code (PSGC)](https://psa.gov.ph/classification/psgc)
- [Social Weather Stations — Home](https://www.sws.org.ph/swsmain/home/)
- [The SWS Surveys of Philippine Hunger, 1998–2024 (AJAD)](https://ajad.searca.org/article?p=5053)
