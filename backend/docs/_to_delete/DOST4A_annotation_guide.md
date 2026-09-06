# Annotation Guide — DOST Region IV-A Expert Review
### aiPHeed CALABARZON Food-Insecurity News Dataset

**What we're asking you to do.** The system has already collected news articles and
tagged each one automatically. We need your expertise to **check those tags** — confirm,
correct, or reject them. Your reviewed labels become the official answer key ("gold
standard") that we measure the system's accuracy against. You are not writing anything
from scratch and not coding — you are grading pre-filled forms.

**What you're reviewing.** ~104 news articles (2020–2026) across Batangas, Quezon,
Laguna, Cavite, and Rizal. Each shows a **headline + opening lines** and the machine's
proposed labels. (Full article bodies were not stored, so judge from the headline and
lead — that is enough for the questions below.)

---

## For each article, answer 4 questions

**1 · Is this really about food insecurity?**  →  **Keep / Drop**
Keep it only if it gives real evidence about food access, supply, hunger, nutrition,
farming, fisheries, or food assistance. Drop incidental "food" words in a non-food story
(restaurant reviews, recipes, a "solar farm" energy deal, agri-tourism).

**2 · Is it really about CALABARZON?**  →  **Confirm / Fix the place**
Check the province and city/municipality tag. Drop or correct it if the story is really
about another region (e.g. it was tagged "Batangas" but is actually about Bulacan).

**3 · Which food-security dimension?**  →  **Pick one (A–F)**

| Code | Dimension | Typical story |
|---|---|---|
| **A** | Food **availability** (production & fisheries) | crop damage, low harvest, fish kill, ASF |
| **B** | Food **accessibility / affordability** | rice/vegetable price spikes, poverty limiting food access |
| **C** | Food **utilization / nutrition** | malnutrition, stunting, feeding programs |
| **D** | Food **stability** (shocks) | typhoon/flood displacement, supply-chain or transport disruption, unrest |
| **E** | **Hunger / food deprivation** (assistance) | relief/ayuda distribution, food packs, hunger relief |
| **F** | **Livelihood** affecting food access | OFW remittance loss, job loss reducing food-buying power |

**4 · How serious?**  →  **High / Medium / Low**
- **High** — crisis, state of calamity, famine, starvation, deaths, thousands affected
- **Medium** — damage, losses, shortage, price surge, displacement, families "hit"/"affected"
- **Low** — mentioned as a risk or minor/localized issue, no major impact stated

*(Optional, if obvious from the text: main commodity — rice, fish, vegetables, etc.)*

---

## Two worked examples

> **"Taal eruption buries Batangas farms; ₱1.2B in crops destroyed, thousands of
> farmers displaced."**
> ✅ Keep · Batangas ✔ · Dimension **A** (production loss) · Severity **High**

> **"New farm-to-table restaurant opens in Tagaytay, draws weekend crowds."**
> ❌ Drop — food word, but no food-insecurity content (food-culture, not food access).

---

## Ground rules

- **When unsure, flag it** (mark "needs review") rather than guessing — we adjudicate flagged rows together.
- **Judge the article as written**, not what you personally know about the area.
- **One primary dimension per article.** If two fit, pick the one the story is *most* about.
- Your call overrides the machine's tag. Disagreeing with the system is the whole point —
  that's how we prove (and improve) its accuracy.

*Questions during review → contact the aiPHeed research team.*
