// Quarter-keyed mock data for CALABARZON — single source of truth
import { RiskLevel } from "./types";

export interface ProvinceQuarterRow {
  id: string;
  name: string;
  scoresByQuarter: Record<string, number>;
  articles: number;
}

export const QUARTER_IDS = [
  "2025-Q1",
  "2025-Q2",
  "2025-Q3",
  "2025-Q4",
  "2026-Q1",
  "2026-Q2",
  "2026-Q3",
  "2026-Q4",
  "2027-Q1",
] as const;
export type QuarterId = typeof QUARTER_IDS[number];

// Risk Level per province per quarter (2027-Q1 is forecast — slight uptick)
export const PROVINCE_QUARTER_DATA: ProvinceQuarterRow[] = [
  {
    id: "quezon",
    name: "Quezon",
    articles: 142,
    scoresByQuarter: {
      "2025-Q1": 0.54, "2025-Q2": 0.56,
      "2025-Q3": 0.58, "2025-Q4": 0.61, "2026-Q1": 0.63,
      "2026-Q2": 0.64, "2026-Q3": 0.65, "2026-Q4": 0.66,
      "2027-Q1": 0.69,
    },
  },
  {
    id: "batangas",
    name: "Batangas",
    articles: 118,
    scoresByQuarter: {
      "2025-Q1": 0.39, "2025-Q2": 0.41,
      "2025-Q3": 0.42, "2025-Q4": 0.44, "2026-Q1": 0.45,
      "2026-Q2": 0.46, "2026-Q3": 0.47, "2026-Q4": 0.48,
      "2027-Q1": 0.50,
    },
  },
  {
    id: "rizal",
    name: "Rizal",
    articles: 97,
    scoresByQuarter: {
      "2025-Q1": 0.37, "2025-Q2": 0.38,
      "2025-Q3": 0.40, "2025-Q4": 0.41, "2026-Q1": 0.42,
      "2026-Q2": 0.43, "2026-Q3": 0.44, "2026-Q4": 0.45,
      "2027-Q1": 0.47,
    },
  },
  {
    id: "cavite",
    name: "Cavite",
    articles: 89,
    scoresByQuarter: {
      "2025-Q1": 0.34, "2025-Q2": 0.36,
      "2025-Q3": 0.37, "2025-Q4": 0.38, "2026-Q1": 0.39,
      "2026-Q2": 0.40, "2026-Q3": 0.40, "2026-Q4": 0.41,
      "2027-Q1": 0.43,
    },
  },
  {
    id: "laguna",
    name: "Laguna",
    articles: 74,
    scoresByQuarter: {
      "2025-Q1": 0.32, "2025-Q2": 0.34,
      "2025-Q3": 0.35, "2025-Q4": 0.36, "2026-Q1": 0.37,
      "2026-Q2": 0.37, "2026-Q3": 0.38, "2026-Q4": 0.39,
      "2027-Q1": 0.41,
    },
  },
];

export function getRiskLevelFromScore(score: number): RiskLevel {
  return score >= 0.5 ? "high" : "low";
}

export function getProvinceScore(provinceId: string, qid: string): number {
  const row = PROVINCE_QUARTER_DATA.find((r) => r.id === provinceId);
  return row?.scoresByQuarter[qid] ?? 0;
}

export function getCalabarzonAverage(qid: string): number {
  const all = PROVINCE_QUARTER_DATA.map((r) => r.scoresByQuarter[qid] ?? 0);
  return all.reduce((s, v) => s + v, 0) / all.length;
}

// Default SHAP for CALABARZON summary (matches Fix 8)
export const DEFAULT_SHAP = [
  { feature: "Engel Coefficient", value: 0.192 },
  { feature: "FPSI Price Stress", value: 0.163 },
  { feature: "Dependency Rate", value: 0.128 },
  { feature: "Income Decile", value: -0.122 },
  { feature: "Household Size", value: 0.075 },
  { feature: "Income Sources", value: -0.061 },
];

// Per-province SHAP — vary slightly so each province feels distinct
export function getShapForProvince(id: string | null): { feature: string; value: number }[] {
  if (!id) return DEFAULT_SHAP;
  const seed = (id.charCodeAt(0) + id.length * 7) % 50;
  return DEFAULT_SHAP.map((s, i) => ({
    feature: s.feature,
    value: Number((s.value + ((seed + i) % 7 - 3) * 0.008).toFixed(3)),
  }));
}

// 5 trigger categories per spec (market, climate, employment, OFW remittance, fish kill)
export const TRIGGER_CATEGORIES = [
  { key: "market", name: "Market / Presyo", color: "hsl(var(--risk-moderate))" },
  { key: "climate", name: "Climate / Panahon", color: "hsl(var(--risk-high))" },
  { key: "employment", name: "Employment / Hanapbuhay", color: "hsl(var(--primary))" },
  { key: "ofw", name: "OFW Remittance", color: "#8b5cf6" },
  { key: "fishkill", name: "Fish Kill", color: "#06b6d4" },
] as const;

// Per-province trigger composition (proportions sum ~1.0)
export const TRIGGER_COMPOSITION_BY_PROVINCE: Record<string, number[]> = {
  quezon:   [0.32, 0.34, 0.16, 0.10, 0.08],
  batangas: [0.36, 0.22, 0.20, 0.14, 0.08],
  rizal:    [0.40, 0.20, 0.22, 0.14, 0.04],
  cavite:   [0.42, 0.18, 0.24, 0.14, 0.02],
  laguna:   [0.38, 0.24, 0.20, 0.14, 0.04],
};
export function getTriggerComposition(provinceId: string | null): number[] {
  if (provinceId && TRIGGER_COMPOSITION_BY_PROVINCE[provinceId])
    return TRIGGER_COMPOSITION_BY_PROVINCE[provinceId];
  // Regional average
  const arrs = Object.values(TRIGGER_COMPOSITION_BY_PROVINCE);
  return TRIGGER_CATEGORIES.map((_, i) => arrs.reduce((s, a) => s + a[i], 0) / arrs.length);
}

/* ═══════════════════════════════════════════════════════════════════════
   5-TRIGGER BREAKDOWN — single source of truth for:
     • Left panel  "Why is this Province at Risk?"  (RiskDriversCardBody)
     • Right panel "…Summary" SHAP explainability card (ShapNarrativeCardBody)
     • Visualization "SHAP Explainability" chart + its PNG / PDF export
     • Quarterly PDF report "Why this forecast" section
   ALL FIVE triggers are fixed basis data and are ALWAYS shown (even at 0%),
   ranked high → low. Each `pct` is the trigger's share of the risk; the five
   sum to exactly 100%. Colour: pct > 20% ⇒ red, otherwise yellow. Recomputed
   per province × per quarter so every quarter's forecast updates automatically.
   ═══════════════════════════════════════════════════════════════════════ */

export const TRIGGER_KEYS = ["market", "climate", "employment", "ofw", "fishkill"] as const;
export type TriggerKey = typeof TRIGGER_KEYS[number];

export const TRIGGER_LABELS: Record<TriggerKey, string> = {
  market: "Market / Prices",
  climate: "Climate Stress",
  employment: "Employment",
  ofw: "OFW Remittance",
  fishkill: "Fish Kill",
};

// Red / yellow cutoff — shared everywhere. `pct` is a trigger's share of ALL 5
// triggers (the five sum to 100%). An even split is 20% each, so a trigger
// ABOVE 20% is carrying more than its share ⇒ red; at or below 20% ⇒ yellow.
export const TRIGGER_RED_CUTOFF = 20;

export function triggerColor(pct: number): string {
  return pct > TRIGGER_RED_CUTOFF
    ? "hsl(var(--risk-high))"      // red
    : "hsl(var(--risk-moderate))"; // yellow
}

// Base composition per province (order = TRIGGER_KEYS, shares sum to ~1.0).
// Regional (null province) = mean of the five provinces.
function triggerBase(provinceId: string | null): Record<TriggerKey, number> {
  const arr = getTriggerComposition(provinceId);
  const out = {} as Record<TriggerKey, number>;
  TRIGGER_KEYS.forEach((k, i) => { out[k] = arr[i]; });
  return out;
}

// Deterministic, smooth per-quarter drift (±0.06) so each quarter differs
// without random noise. Keyed on quarter index + trigger index.
function triggerQuarterDrift(key: TriggerKey, qid: string): number {
  const qi = Math.max(0, (QUARTER_IDS as readonly string[]).indexOf(qid));
  const ki = TRIGGER_KEYS.indexOf(key);
  return Math.sin((qi + 1) * 1.3 + ki * 1.7) * 0.06;
}

// Largest-remainder rounding so the integer percentages sum to exactly 100.
function sharesToPct(shares: number[]): number[] {
  const total = shares.reduce((a, b) => a + b, 0) || 1;
  const exact = shares.map((s) => (s / total) * 100);
  const floors = exact.map(Math.floor);
  const remaining = 100 - floors.reduce((a, b) => a + b, 0);
  const order = exact
    .map((v, i) => [v - floors[i], i] as const)
    .sort((a, b) => b[0] - a[0]);
  const out = [...floors];
  for (let j = 0; j < remaining && j < order.length; j++) out[order[j][1]] += 1;
  return out;
}

export interface TriggerContribution {
  key: TriggerKey;
  label: string;
  share: number;  // 0..1 — this trigger's slice of ALL 5 (the 5 sum to 1)
  pct: number;    // integer % — all 5 sum to exactly 100
  color: string;  // shared colour, from triggerColor(pct)
}

/**
 * All 5 triggers for a province (null = CALABARZON regional avg) in a quarter.
 * Always returns 5 entries (a trigger may be 0%). `pct` is each trigger's share
 * of the whole; the five sum to 100. Sorted by pct, descending.
 */
export function getTriggerBreakdown(
  provinceId: string | null,
  qid: string,
): TriggerContribution[] {
  const base = triggerBase(provinceId);

  const raw = TRIGGER_KEYS.map((k) => Math.max(0, base[k] + triggerQuarterDrift(k, qid)));
  const sum = raw.reduce((a, b) => a + b, 0) || 1;
  const shares = raw.map((x) => x / sum);
  const pcts = sharesToPct(shares); // integers summing to exactly 100

  return TRIGGER_KEYS.map((k, i) => ({
    key: k,
    label: TRIGGER_LABELS[k],
    share: shares[i],
    pct: pcts[i],
    color: triggerColor(pcts[i]),
  })).sort((a, b) => b.pct - a.pct);
}

/**
 * Plain-language "SHAP explainability" paragraph for a trigger breakdown.
 * Rendered on-screen with the Visualization charts and embedded verbatim in
 * the PNG / PDF exports and the quarterly PDF report.
 */
export function explainTriggerBreakdown(
  subject: string,
  quarterLabel: string,
  items: TriggerContribution[],
): string {
  const ranked = [...items].sort((a, b) => b.pct - a.pct);
  const red = ranked.filter((i) => i.pct > TRIGGER_RED_CUTOFF).map((i) => i.label);
  const list = ranked.map((i) => `${i.label} ${i.pct}%`).join(", ");

  const sentences: string[] = [];
  sentences.push(
    `This SHAP explainability view breaks ${subject}'s ${quarterLabel} food-insecurity risk into all five trigger categories, scaled so the shares add up to 100%.`,
  );
  sentences.push(`Ranked contribution this quarter: ${list}.`);
  sentences.push(
    red.length
      ? `${red.join(" and ")} ${red.length > 1 ? "each sit" : "sits"} above the 20% even-share line and ${red.length > 1 ? "are" : "is"} flagged red; the rest are yellow.`
      : `No trigger crosses the 20% even-share line this quarter, so all are shown yellow.`,
  );
  sentences.push(
    `Shares are AI estimates from news-signal and indicator data, recomputed every quarter - an explanation aid, not an official measurement.`,
  );
  return sentences.join(" ");
}

// Algorithm thresholds (spec rule 4)
export const RISK_DISPLAY_CUTOFF = 0.5;   // HIGH/LOW pill threshold for forecast display
export const ALERT_THRESHOLD = 0.6;        // Active alert gating threshold
export const LIMITED_SIGNAL_MIN_ARTICLES = 5;

export function isLimitedSignal(provinceId: string, qid?: string): boolean {
  const row = PROVINCE_QUARTER_DATA.find((r) => r.id === provinceId);
  if (!row) return false;
  // Demo: mark provinces with low total article volume as LIMITED for forecast quarters
  return row.articles < 80 && (qid?.startsWith("2027") ?? false);
}

export function getRiskLabel(score: number): "HIGH" | "LOW" {
  return score >= RISK_DISPLAY_CUTOFF ? "HIGH" : "LOW";
}

// Forecast metadata per spec rule 6 (3-month-ahead)
export function getQuarterMeta(qid: string): {
  horizonLabel: string;
  generatedOn: string;
  verificationBy: string;
} {
  const [yStr, qStr] = qid.split("-Q");
  const year = parseInt(yStr, 10);
  const q = parseInt(qStr, 10) as 1 | 2 | 3 | 4;
  // Generated 3 months before quarter start
  const startMonth = (q - 1) * 3; // 0,3,6,9
  const gen = new Date(year, startMonth - 3, 1);
  const ver = new Date(year, startMonth + 3, 0); // last day of quarter
  const fmt = (d: Date) =>
    d.toLocaleDateString("en-PH", { year: "numeric", month: "short", day: "numeric" });
  return {
    horizonLabel: "3-month-ahead",
    generatedOn: fmt(gen),
    verificationBy: fmt(ver),
  };
}

// Vulnerability driver classification (spec section 4)
export function getVulnerabilityDriver(povertyNorm: number, densityNorm: number):
  "poverty-led" | "density-led" | "balanced" {
  const diff = povertyNorm - densityNorm;
  if (diff > 0.15) return "poverty-led";
  if (diff < -0.15) return "density-led";
  return "balanced";
}

export const QUEZON_TOP_MUNIS = [
  { name: "Infanta", score: 0.71 },
  { name: "General Nakar", score: 0.68 },
  { name: "Polillo", score: 0.64 },
  { name: "Real", score: 0.61 },
  { name: "Alabat", score: 0.58 },
  { name: "Atimonan", score: 0.54 },
];

export interface Article {
  title: string;
  source: string;
  date: string;
  excerpt: string;
  url: string;
}
export const SAMPLE_ARTICLES: Article[] = [
  {
    title: "Rice still above P50/kilo levels; no dip in sight",
    source: "Inquirer.net",
    date: "2024-09-12",
    excerpt: "Retail rice prices remain stubbornly high across Luzon despite tariff cuts, sustaining food-cost pressure on low-income households.",
    url: "https://newsinfo.inquirer.net/1884191/rice-still-above-p50-kilo-levels-no-dip-in-sight",
  },
  {
    title: "Typhoon Aghon leaves 3 dead in Quezon",
    source: "Philstar",
    date: "2024-05-28",
    excerpt: "Typhoon Aghon's landfall over Quezon damaged crops and infrastructure, adding pressure on farmers in CALABARZON.",
    url: "https://www.philstar.com/headlines/2024/05/28/2358462/typhoon-aghon-leaves-3-dead-quezon",
  },
  {
    title: "Remittance growth slows in October",
    source: "BusinessWorld",
    date: "2024-12-17",
    excerpt: "OFW cash remittances grew just 2.7% in October — the slowest in four months — softening household spending power in remittance-reliant provinces like Cavite.",
    url: "https://www.bworldonline.com/top-stories/2024/12/17/641938/remittance-growth-slows-in-october/",
  },
  {
    title: "DA Region IV-A adds satellite warehouses to boost disaster response",
    source: "Philippine News Agency",
    date: "2024-12-13",
    excerpt: "DA Region IV-A expands its prepositioning network to deliver food packs faster to disaster- and food-stress-prone LGUs.",
    url: "https://www.pna.gov.ph/articles/1239902",
  },
  {
    title: "Intra-household inequality in food expenditures and diet quality in the Philippines",
    source: "Food Security (Springer)",
    date: "2024-10-04",
    excerpt: "Peer-reviewed study showing rising food-expenditure shares (Engel coefficient) signal worsening diet quality among low-income Filipino households.",
    url: "https://link.springer.com/article/10.1007/s12571-024-01485-6",
  },
];

export const REGION_OPTIONS = [
  "Region I — Ilocos",
  "Region II — Cagayan Valley",
  "Region III — Central Luzon",
  "Region IV-A CALABARZON",
  "Region IV-B MIMAROPA",
  "Region V — Bicol",
  "NCR",
  "CAR",
  "BARMM",
];
