import jsPDF from "jspdf";
import {
  PROVINCE_QUARTER_DATA,
  getRiskLevelFromScore,
  getTriggerBreakdown,
  explainTriggerBreakdown,
  TRIGGER_RED_CUTOFF,
  QUEZON_TOP_MUNIS,
  SAMPLE_ARTICLES,
} from "@/data/quarterData";

// ─────────────────────────────────────────────────────────────────────────────
// Editorial PDF report — mirrors the aiPHeed quarterly forecast brief layout.
// Uses jsPDF only (no html2canvas) for crisp vector text.
// ─────────────────────────────────────────────────────────────────────────────

const COL = {
  ink: [20, 28, 50] as const,        // near-black navy
  body: [55, 60, 75] as const,
  muted: [125, 130, 145] as const,
  rule: [205, 200, 185] as const,
  amber: [168, 122, 42] as const,    // gold accent
  amberSoft: [240, 222, 185] as const,
  blueAccent: [110, 130, 170] as const,
  cream: [251, 246, 232] as const,
  tier1: [127, 36, 36] as const,     // brick red
  tier2: [222, 196, 142] as const,
  tier3: [228, 220, 198] as const,
  rowAlt: [248, 246, 240] as const,
};

const FONT_SERIF = "times";
const FONT_SANS = "helvetica";
const FONT_MONO = "courier";

const PAGE_W = 210; // A4 mm
const PAGE_H = 297;
const MARGIN_X = 16;
const CONTENT_W = PAGE_W - MARGIN_X * 2;

interface ReportScope {
  kind: "region" | "province" | "municipality";
  quarterId: string;
  quarterLabel: string; // e.g. "Q3 2026 (Jul-Sep)"
  quarterShort: string; // e.g. "Q3 2026"
  monthsShort: string;  // e.g. "Jul-Sep"
  title: string;        // e.g. "Quezon Province" or "CALABARZON"
  provinceId?: string | null; // for the trigger breakdown; null = regional average
  subtitle: string;     // narrative sub-clause (italic)
  probability: number;  // 0..1
  delta: number;        // vs previous quarter
  atRisk: number;       // estimated residents
  populationPct?: number;
  tier1Count: number;
  totalMunis: number;
  articles: number;
  reportNo: string;
  pageOf: number;
  // optional municipal table (Quezon only)
  municipalRows?: { name: string; pop: number; poverty: number; prob: number; delta: number; articles: number; tier: 1 | 2 | 3 }[];
}

// ─────────── drawing helpers ───────────

function setFill(doc: jsPDF, c: readonly [number, number, number]) {
  doc.setFillColor(c[0], c[1], c[2]);
}
function setStroke(doc: jsPDF, c: readonly [number, number, number]) {
  doc.setDrawColor(c[0], c[1], c[2]);
}
function setText(doc: jsPDF, c: readonly [number, number, number]) {
  doc.setTextColor(c[0], c[1], c[2]);
}

function drawHairline(doc: jsPDF, y: number, x1 = MARGIN_X, x2 = PAGE_W - MARGIN_X) {
  setStroke(doc, COL.rule);
  doc.setLineWidth(0.2);
  doc.line(x1, y, x2, y);
}

function drawRunningHeader(doc: jsPDF, scope: ReportScope, sectionLabel: string) {
  setText(doc, COL.amber);
  doc.setFont(FONT_MONO, "normal").setFontSize(7.5);
  const left = `○ AIPHEED · ${scope.quarterShort.toUpperCase()} · ${scope.title.toUpperCase()}`;
  doc.text(left, MARGIN_X, 12);
  doc.text(sectionLabel.toUpperCase(), PAGE_W - MARGIN_X, 12, { align: "right" });
  drawHairline(doc, 14.5);
}

function drawRunningFooter(doc: jsPDF, scope: ReportScope, page: number, sectionLabel: string, leftNote = "") {
  drawHairline(doc, PAGE_H - 16);
  setText(doc, COL.muted);
  doc.setFont(FONT_MONO, "normal").setFontSize(7.5);
  doc.text(leftNote.toUpperCase(), MARGIN_X, PAGE_H - 11);
  doc.text(`P. ${page} · ${sectionLabel}`.toUpperCase(), PAGE_W - MARGIN_X, PAGE_H - 11, { align: "right" });
}

function drawTag(doc: jsPDF, x: number, y: number, label: string, fill: readonly [number, number, number], textC: readonly [number, number, number]) {
  doc.setFont(FONT_MONO, "bold").setFontSize(7.5);
  const w = doc.getTextWidth(label) + 4;
  setFill(doc, fill);
  doc.rect(x, y - 3.4, w, 4.6, "F");
  setText(doc, textC);
  doc.text(label, x + 2, y);
  return w;
}

function wrap(doc: jsPDF, text: string, maxWidth: number): string[] {
  return doc.splitTextToSize(text, maxWidth) as string[];
}

// Big editorial headline with one italic emphasis word/phrase.
// Renders text with `*emphasis*` markers in serif italic blue, rest in serif regular ink.
function drawDisplayTitle(doc: jsPDF, text: string, x: number, y: number, maxWidth: number, sizeMm = 26): number {
  doc.setFont(FONT_SERIF, "normal").setFontSize(sizeMm);
  // tokenize on * markers
  const segments: { text: string; italic: boolean }[] = [];
  text.split(/(\*[^*]+\*)/).forEach((s) => {
    if (!s) return;
    if (s.startsWith("*") && s.endsWith("*")) segments.push({ text: s.slice(1, -1), italic: true });
    else segments.push({ text: s, italic: false });
  });

  // word-wrap respecting segment boundaries
  const lineHeight = sizeMm * 0.42;
  const lines: { text: string; italic: boolean }[][] = [[]];
  segments.forEach((seg) => {
    doc.setFont(FONT_SERIF, seg.italic ? "italic" : "normal");
    const words = seg.text.split(/(\s+)/);
    words.forEach((w) => {
      if (!w) return;
      const cur = lines[lines.length - 1];
      const tentative = cur.map((s) => s.text).join("") + w;
      if (doc.getTextWidth(tentative) > maxWidth && cur.length > 0) {
        lines.push([{ text: w.trimStart(), italic: seg.italic }]);
      } else {
        cur.push({ text: w, italic: seg.italic });
      }
    });
  });

  let cy = y;
  lines.forEach((line) => {
    let cx = x;
    line.forEach((seg) => {
      doc.setFont(FONT_SERIF, seg.italic ? "italic" : "normal");
      setText(doc, seg.italic ? COL.blueAccent : COL.ink);
      doc.text(seg.text, cx, cy);
      cx += doc.getTextWidth(seg.text);
    });
    cy += lineHeight;
  });
  return cy;
}

function drawStatStrip(
  doc: jsPDF,
  x: number,
  y: number,
  totalW: number,
  stats: { label: string; value: string; sub?: string }[],
  bigSize = 16
) {
  drawHairline(doc, y);
  const colW = totalW / stats.length;
  stats.forEach((s, i) => {
    const cx = x + colW * i + 2;
    setText(doc, COL.amber);
    doc.setFont(FONT_MONO, "normal").setFontSize(7);
    doc.text(s.label.toUpperCase(), cx, y + 6);
    setText(doc, COL.ink);
    doc.setFont(FONT_SERIF, "normal").setFontSize(bigSize);
    doc.text(s.value, cx, y + 14);
    if (s.sub) {
      setText(doc, COL.muted);
      doc.setFont(FONT_MONO, "normal").setFontSize(6.5);
      const subLines = wrap(doc, s.sub.toUpperCase(), colW - 4);
      subLines.forEach((ln, k) => doc.text(ln, cx, y + 18 + k * 3));
    }
    if (i > 0) {
      setStroke(doc, COL.rule);
      doc.setLineWidth(0.15);
      doc.line(x + colW * i, y + 1, x + colW * i, y + 22);
    }
  });
  drawHairline(doc, y + 24);
  return y + 24;
}

// ─────────── Page 1: Cover ───────────

function drawCover(doc: jsPDF, scope: ReportScope) {
  // Top branding band
  setFill(doc, COL.ink);
  doc.rect(MARGIN_X, 14, 8, 8, "F");
  setText(doc, COL.amber);
  doc.setFont(FONT_SERIF, "italic").setFontSize(11);
  doc.text("◉", MARGIN_X + 1.6, 19.6);

  setText(doc, COL.ink);
  doc.setFont(FONT_SERIF, "normal").setFontSize(18);
  doc.text("aiPHeed", MARGIN_X + 12, 20);
  setText(doc, COL.muted);
  doc.setFont(FONT_MONO, "normal").setFontSize(7);
  doc.text(`FORECAST SYSTEM  ·  REGION IV-A`, MARGIN_X + 12, 24.5);

  // Right side: DA attribution
  setText(doc, COL.ink);
  doc.setFont(FONT_SANS, "bold").setFontSize(8.5);
  doc.text("Department of Agriculture", PAGE_W - MARGIN_X, 18, { align: "right" });
  setText(doc, COL.muted);
  doc.setFont(FONT_MONO, "normal").setFontSize(7);
  doc.text("REGION IV-A  ·  CALABARZON", PAGE_W - MARGIN_X, 22, { align: "right" });
  doc.text("REPUBLIC OF THE PHILIPPINES", PAGE_W - MARGIN_X, 25.5, { align: "right" });

  // Heavy rule
  setStroke(doc, COL.ink);
  doc.setLineWidth(0.6);
  doc.line(MARGIN_X, 30, PAGE_W - MARGIN_X, 30);

  // Tag row
  setText(doc, COL.amber);
  doc.setFont(FONT_MONO, "bold").setFontSize(8);
  const tagText = `QUARTERLY FORECAST REPORT  ·  ${scope.quarterShort.toUpperCase()}  ·  ${scope.monthsShort.toUpperCase()}`;
  doc.text(tagText, MARGIN_X, 44);
  const tagW = doc.getTextWidth(tagText);
  const tier =
    scope.probability >= 0.75 ? { label: "TIER-1 ADVISORY", c: COL.tier1, t: [255, 255, 255] as const } :
    scope.probability >= 0.55 ? { label: "TIER-2 WATCH",     c: COL.amber, t: [255, 255, 255] as const } :
                                 { label: "TIER-3 BASELINE",  c: COL.tier3, t: COL.ink };
  drawTag(doc, MARGIN_X + tagW + 4, 44, tier.label, tier.c, tier.t);

  // Display title
  const endY = drawDisplayTitle(doc, scope.title + " · " + scope.subtitle, MARGIN_X, 60, CONTENT_W, 26);

  // Italic narrative
  setText(doc, COL.body);
  doc.setFont(FONT_SERIF, "italic").setFontSize(11);
  const narrative = wrap(
    doc,
    `Food insecurity is forecast to ${scope.delta >= 0 ? "rise" : "ease"} across ${scope.tier1Count} municipalities in ${scope.title}. The dominant signal this cycle is rice price volatility, compounded by climate-linked crop pressure observed in trailing news evidence.`,
    CONTENT_W
  );
  let ny = endY + 4;
  narrative.forEach((ln) => {
    doc.text(ln, MARGIN_X, ny);
    ny += 5.2;
  });

  // Stat strip
  const stripY = Math.max(ny + 6, 168);
  drawStatStrip(doc, MARGIN_X, stripY, CONTENT_W, [
    { label: "Forecast Probability", value: scope.probability.toFixed(2) },
    { label: "At-Risk Residents", value: formatThousands(scope.atRisk) + (scope.atRisk >= 1000 ? "K" : "") },
    { label: "Tier-1 Municipalities", value: `${scope.tier1Count}`, sub: `of ${scope.totalMunis}` },
    { label: "News Articles Processed", value: formatThousands(scope.articles) },
  ]);

  // Metadata block
  const mY = stripY + 36;
  const colW = CONTENT_W / 3;
  const metaCols = [
    { h: "Forecast generated", l1: prettyDate(new Date()) + "  ·  06:00 PHT", l2: "Model version 3.2 · walk-forward validation" },
    { h: "Horizon", l1: scope.monthsShort, l2: "3-month binary risk probability" },
    { h: "Verification date", l1: nextQuarterCheckDate(scope.quarterId), l2: "Anchored to DA-FNRI NNS" },
  ];
  metaCols.forEach((m, i) => {
    const cx = MARGIN_X + colW * i;
    setText(doc, COL.ink);
    doc.setFont(FONT_SANS, "bold").setFontSize(9);
    doc.text(m.h, cx, mY);
    setText(doc, COL.body);
    doc.setFont(FONT_MONO, "normal").setFontSize(7.5);
    doc.text(m.l1, cx, mY + 5);
    setText(doc, COL.muted);
    doc.text(wrap(doc, m.l2, colW - 4), cx, mY + 9);
  });

  drawHairline(doc, mY + 18);
  setText(doc, COL.ink);
  doc.setFont(FONT_SANS, "bold").setFontSize(9);
  doc.text("Prepared by the aiPHeed team", MARGIN_X, mY + 24);
  setText(doc, COL.muted);
  doc.setFont(FONT_MONO, "normal").setFontSize(7.5);
  doc.text(`DESTURA  ·  ESICO  ·  MELINDO  ·  UNIVERSITY OF THE PHILIPPINES`, MARGIN_X, mY + 28);
  doc.text(`REPORT #${scope.reportNo}  ·  P. 1 OF ${scope.pageOf}`, PAGE_W - MARGIN_X, mY + 28, { align: "right" });
}

// ─────────── Page 2: Executive Summary ───────────

function drawExecutiveSummary(doc: jsPDF, scope: ReportScope, pageNum: number) {
  drawRunningHeader(doc, scope, "EXECUTIVE SUMMARY");
  setText(doc, COL.amber);
  doc.setFont(FONT_MONO, "bold").setFontSize(8);
  doc.text("SECTION 1  ·  EXECUTIVE SUMMARY", MARGIN_X, 22);

  const titleEndY = drawDisplayTitle(
    doc,
    scope.kind === "region"
      ? `Region-wide signal in *${scope.quarterShort}*.`
      : scope.kind === "municipality"
        ? `Municipal hotspot enters *advisory*.`
        : `Eastern coastal ${scope.title.split(" ")[0]} enters *Tier-1 advisory*.`,
    MARGIN_X,
    36,
    CONTENT_W,
    22
  );

  // Quote / summary block
  const qY = titleEndY + 4;
  const qHeight = 36;
  setFill(doc, COL.cream);
  doc.rect(MARGIN_X, qY, CONTENT_W, qHeight, "F");
  setFill(doc, COL.amber);
  doc.rect(MARGIN_X, qY, 1.2, qHeight, "F");
  setText(doc, COL.ink);
  doc.setFont(FONT_SERIF, "normal").setFontSize(10.5);
  const summary = `The ${scope.quarterShort} forecast places ${scope.title} at ${scope.probability.toFixed(2)} probability of elevated food insecurity, ${scope.delta >= 0 ? "up" : "down"} ${(scope.delta >= 0 ? "+" : "")}${scope.delta.toFixed(2)} from the previous quarter. ${scope.tier1Count} ${scope.tier1Count === 1 ? "municipality meets" : "contiguous municipalities meet"} Tier-1 advisory criteria. An estimated ${formatThousands(scope.atRisk * 1000)} residents are at heightened risk through the end of ${scope.monthsShort}.`;
  const sLines = wrap(doc, summary, CONTENT_W - 10);
  sLines.forEach((ln, i) => doc.text(ln, MARGIN_X + 5, qY + 7 + i * 5.2));

  // Stat strip
  const statY = qY + qHeight + 8;
  drawStatStrip(doc, MARGIN_X, statY, CONTENT_W, [
    { label: "Probability", value: scope.probability.toFixed(2), sub: `Baseline ${(scope.probability - scope.delta).toFixed(2)}  ·  Δ ${scope.delta >= 0 ? "+" : ""}${scope.delta.toFixed(2)}` },
    { label: "At-Risk Residents", value: formatThousands(scope.atRisk * 1000), sub: scope.populationPct ? `${scope.populationPct.toFixed(1)}% of population` : "" },
    { label: "Tier-1 Municipalities", value: `${scope.tier1Count}`, sub: scope.kind === "province" ? "Eastern coastal corridor" : "Within scope" },
  ], 14);

  // Why this forecast — SHAP explainability over the 5 trigger categories
  let y = statY + 32;
  setText(doc, COL.ink);
  doc.setFont(FONT_SANS, "bold").setFontSize(11);
  doc.text("Why this forecast", MARGIN_X, y);
  setText(doc, COL.amber);
  doc.setFont(FONT_MONO, "normal").setFontSize(7.5);
  doc.text("SHAP EXPLAINABILITY  ·  5 TRIGGER CATEGORIES", MARGIN_X + 38, y);
  y += 4;
  drawHairline(doc, y);
  y += 5;
  // header row
  setText(doc, COL.muted);
  doc.setFont(FONT_MONO, "normal").setFontSize(7);
  doc.text("TRIGGER", MARGIN_X, y);
  doc.text("SHARE OF RISK  (5 SUM TO 100%)", MARGIN_X + 100, y);
  y += 3.5;

  const triggers = getTriggerBreakdown(scope.provinceId ?? null, scope.quarterId);
  triggers.forEach((t) => {
    const red = t.pct > TRIGGER_RED_CUTOFF;
    setText(doc, COL.ink);
    doc.setFont(FONT_SANS, "normal").setFontSize(9);
    doc.text(t.label, MARGIN_X, y + 4);
    // bar — length scaled against a 50% ceiling; brick red above the 20% even-share line, else amber
    const barX = MARGIN_X + 100;
    const barW = 50;
    setFill(doc, [232, 228, 216]);
    doc.rect(barX, y + 1.5, barW, 3, "F");
    const len = Math.min(t.pct / 50, 1) * barW;
    setFill(doc, red ? COL.tier1 : COL.amber);
    doc.rect(barX, y + 1.5, len, 3, "F");
    setText(doc, red ? COL.tier1 : COL.amber);
    doc.setFont(FONT_MONO, "bold").setFontSize(8.5);
    doc.text(`${t.pct}%`, PAGE_W - MARGIN_X, y + 4, { align: "right" });
    y += 6.5;
  });
  drawHairline(doc, y + 1);
  y += 6;
  setText(doc, COL.muted);
  doc.setFont(FONT_SERIF, "italic").setFontSize(8.5);
  const explLines = wrap(
    doc,
    explainTriggerBreakdown(scope.title, scope.quarterShort, triggers),
    CONTENT_W,
  );
  explLines.forEach((ln) => {
    doc.text(ln, MARGIN_X, y);
    y += 3.8;
  });
  y += 6;

  // Most-cited evidence
  setText(doc, COL.ink);
  doc.setFont(FONT_SANS, "bold").setFontSize(11);
  doc.text("Most-cited evidence", MARGIN_X, y);
  setText(doc, COL.amber);
  doc.setFont(FONT_MONO, "normal").setFontSize(7.5);
  doc.text(`LAST 30 DAYS  ·  4 OF ${scope.articles} SHOWN`, MARGIN_X + 42, y);
  y += 4;
  drawHairline(doc, y);
  y += 5;

  const tagFor = (i: number) => (["CLIMATE", "MARKET", "MARKET", "CLIMATE"][i] ?? "MARKET");
  SAMPLE_ARTICLES.slice(0, 4).forEach((a, i) => {
    setText(doc, COL.muted);
    doc.setFont(FONT_MONO, "normal").setFontSize(7.5);
    doc.text(a.date, MARGIN_X, y + 4);
    setText(doc, COL.ink);
    doc.setFont(FONT_SERIF, "normal").setFontSize(9.5);
    const titleLines = wrap(doc, `"${a.title}"`, 110);
    titleLines.forEach((ln, k) => doc.text(ln, MARGIN_X + 22, y + 4 + k * 4.4));
    setText(doc, COL.muted);
    doc.setFont(FONT_MONO, "normal").setFontSize(7);
    doc.text(a.source, MARGIN_X + 22, y + 4 + titleLines.length * 4.4);
    setText(doc, COL.amber);
    doc.setFont(FONT_MONO, "bold").setFontSize(7.5);
    doc.text(tagFor(i), PAGE_W - MARGIN_X, y + 4, { align: "right" });
    y += 4.4 * titleLines.length + 6;
    drawHairline(doc, y - 2);
  });

  drawRunningFooter(doc, scope, pageNum, "Section 1 · Executive summary", "Prepared under human review");
}

// ─────────── Page 3: Municipal disaggregation ───────────

function drawMunicipalSection(doc: jsPDF, scope: ReportScope, pageNum: number) {
  drawRunningHeader(doc, scope, "MUNICIPAL DISAGGREGATION");
  setText(doc, COL.amber);
  doc.setFont(FONT_MONO, "bold").setFontSize(8);
  doc.text("SECTION 2  ·  MUNICIPAL DISAGGREGATION", MARGIN_X, 22);

  const titleEndY = drawDisplayTitle(
    doc,
    `${scope.tier1Count} coastal municipalities drive the *provincial* signal.`,
    MARGIN_X,
    36,
    CONTENT_W,
    22
  );

  // Map placeholder block (dark navy)
  const mapY = titleEndY + 4;
  const mapH = 60;
  setFill(doc, COL.ink);
  doc.rect(MARGIN_X, mapY, CONTENT_W, mapH, "F");
  setText(doc, [255, 255, 255]);
  doc.setFont(FONT_MONO, "normal").setFontSize(7);
  doc.text("N ↑   ·   1:250k", MARGIN_X + 4, mapY + 6);
  // dot scatter representing municipalities
  const rows = scope.municipalRows ?? [];
  rows.slice(0, 12).forEach((r, i) => {
    const cx = MARGIN_X + 18 + (CONTENT_W - 36) * (i / 11);
    const cy = mapY + 24 + Math.sin(i * 1.3) * 8;
    const radius = 2 + (r.pop / 100000) * 2;
    const sev = r.tier === 1 ? COL.tier1 : r.tier === 2 ? COL.amber : [120, 150, 100] as const;
    setFill(doc, sev);
    doc.circle(cx, cy, radius, "F");
  });
  // legend
  setText(doc, [220, 220, 220]);
  doc.setFont(FONT_MONO, "normal").setFontSize(6.5);
  doc.text("Forecast severity  ·  Low → Critical", PAGE_W - MARGIN_X - 2, mapY + mapH - 3, { align: "right" });

  // Table
  let y = mapY + mapH + 8;
  const cols = [
    { h: "MUNICIPALITY", w: 38, align: "left" as const },
    { h: "POP.", w: 18, align: "right" as const },
    { h: "POVERTY", w: 22, align: "right" as const },
    { h: "PROB.", w: 18, align: "right" as const },
    { h: "Δ VS Q-1", w: 22, align: "right" as const },
    { h: "ARTICLES", w: 22, align: "right" as const },
    { h: "TIER", w: CONTENT_W - 38 - 18 - 22 - 18 - 22 - 22, align: "right" as const },
  ];
  setFill(doc, COL.cream);
  doc.rect(MARGIN_X, y, CONTENT_W, 6, "F");
  setText(doc, COL.ink);
  doc.setFont(FONT_MONO, "bold").setFontSize(7);
  let cx = MARGIN_X + 2;
  cols.forEach((c) => {
    doc.text(c.h, c.align === "right" ? cx + c.w - 3 : cx, y + 4, { align: c.align });
    cx += c.w;
  });
  y += 6;
  rows.forEach((r, idx) => {
    if (idx % 2 === 1) {
      setFill(doc, COL.rowAlt);
      doc.rect(MARGIN_X, y, CONTENT_W, 6.4, "F");
    }
    setText(doc, COL.ink);
    doc.setFont(FONT_SANS, "normal").setFontSize(8.5);
    cx = MARGIN_X + 2;
    const vals = [
      r.name,
      formatThousands(r.pop),
      r.poverty.toFixed(1) + "%",
      r.prob.toFixed(2),
      `${r.delta >= 0 ? "+" : ""}${r.delta.toFixed(2)}`,
      `${r.articles}`,
    ];
    vals.forEach((v, i) => {
      const c = cols[i];
      if (i === 4) setText(doc, r.delta >= 0 ? COL.amber : COL.blueAccent);
      else setText(doc, COL.ink);
      doc.text(v, c.align === "right" ? cx + c.w - 3 : cx, y + 4.4, { align: c.align });
      cx += c.w;
    });
    // tier pill
    const tierC = r.tier === 1 ? COL.tier1 : r.tier === 2 ? COL.tier2 : COL.tier3;
    const tierTextC = r.tier === 1 ? [255, 255, 255] as const : COL.ink;
    const tierLabel = `TIER-${r.tier}`;
    doc.setFont(FONT_MONO, "bold").setFontSize(6.8);
    const pw = doc.getTextWidth(tierLabel) + 4;
    setFill(doc, tierC);
    const px = cx + cols[6].w - pw - 3;
    doc.rect(px, y + 1.6, pw, 3.6, "F");
    setText(doc, tierTextC);
    doc.text(tierLabel, px + 2, y + 4.3);
    y += 6.4;
  });

  // Caveat box
  y += 6;
  setFill(doc, [240, 238, 230]);
  doc.rect(MARGIN_X, y, CONTENT_W, 16, "F");
  setText(doc, COL.ink);
  doc.setFont(FONT_SANS, "bold").setFontSize(8.5);
  doc.text("Estimate, not measurement.", MARGIN_X + 4, y + 5.5);
  setText(doc, COL.body);
  doc.setFont(FONT_SANS, "normal").setFontSize(8);
  const cav = wrap(
    doc,
    `Municipal values are spatially disaggregated from the province forecast using PSA poverty incidence (60%) and population density (40%). Treat as allocation guidance. Confidence intervals widen for municipalities with fewer than 20 articles in the trailing 90-day window.`,
    CONTENT_W - 8
  );
  cav.forEach((ln, i) => doc.text(ln, MARGIN_X + 4, y + 9.5 + i * 3.6));

  drawRunningFooter(doc, scope, pageNum, "Section 2 · Municipal disaggregation", "Source: PSA · GNews · aiPHeed v3.2");
}

// ─────────── Page 4: Recommended actions ───────────

function drawActionsSection(doc: jsPDF, scope: ReportScope, pageNum: number) {
  drawRunningHeader(doc, scope, "RECOMMENDED ACTIONS");
  setText(doc, COL.amber);
  doc.setFont(FONT_MONO, "bold").setFontSize(8);
  doc.text("SECTION 3  ·  RECOMMENDED ANTICIPATORY ACTION", MARGIN_X, 22);

  const titleEndY = drawDisplayTitle(
    doc,
    `Three actions, *audit-traced* to this forecast.`,
    MARGIN_X,
    36,
    CONTENT_W,
    22
  );

  setText(doc, COL.body);
  doc.setFont(FONT_SANS, "normal").setFontSize(9.5);
  const lead = wrap(
    doc,
    `These recommendations are draft suggestions requiring human confirmation by DA Region IV-A before any downstream notification. Each action lists the evidence chain the model used to surface it.`,
    CONTENT_W
  );
  let y = titleEndY + 6;
  lead.forEach((ln) => {
    doc.text(ln, MARGIN_X, y);
    y += 4.6;
  });
  y += 4;

  const actions = [
    {
      title: scope.kind === "municipality"
        ? `Coordinate feeding buffer in ${scope.title}`
        : `Pre-position feeding buffer in top-3 hotspots`,
      body: `Forecast > ${(scope.probability + 0.03).toFixed(2)} across ${Math.min(3, scope.tier1Count)} contiguous municipalities with historical typhoon exposure. Estimated households requiring supplementary feeding this cycle: ${formatThousands(Math.round(scope.atRisk * 0.65))}. Pre-position before the seasonality peak to enable response within 48h of activation.`,
      evidence: ["6 ARTICLES", "3 SHAP DRIVERS", "PSA FIES '23"],
      highlighted: true,
    },
    {
      title: `Coordinate with LGUs on crop-insurance fast-track`,
      body: `Articles reference existing ₱420M DA crop-insurance releases. Action: brief LGU teams to ensure municipal claims are filed before peak season. Estimated reach: ${formatThousands(Math.round(scope.atRisk * 0.18))} farmer households.`,
      evidence: ["2 ARTICLES", "ABS-CBN · JUN 05", "GMA · MAY 28"],
      highlighted: false,
    },
    {
      title: `Place watch-list municipalities for re-evaluation`,
      body: `Tier-3 areas with rising quarter-on-quarter deltas (+0.08, +0.09). No pre-positioning indicated this cycle. Re-assess in next quarterly publication.`,
      evidence: ["Δ WATCHLIST", `VERIFY ${nextQuarterCheckDate(scope.quarterId)}`],
      highlighted: false,
    },
  ];

  actions.forEach((a, i) => {
    const cardH = 32;
    if (a.highlighted) {
      setFill(doc, COL.cream);
      doc.rect(MARGIN_X, y, CONTENT_W, cardH, "F");
    } else {
      setStroke(doc, COL.rule);
      doc.setLineWidth(0.2);
      doc.rect(MARGIN_X, y, CONTENT_W, cardH);
    }
    setText(doc, COL.amber);
    doc.setFont(FONT_MONO, "bold").setFontSize(8);
    doc.text(`0${i + 1}`, MARGIN_X + 4, y + 6);
    setText(doc, COL.ink);
    doc.setFont(FONT_SANS, "bold").setFontSize(10);
    doc.text(a.title, MARGIN_X + 14, y + 6);
    setText(doc, COL.body);
    doc.setFont(FONT_SANS, "normal").setFontSize(8.5);
    const bodyLines = wrap(doc, a.body, CONTENT_W - 60);
    bodyLines.slice(0, 4).forEach((ln, k) => doc.text(ln, MARGIN_X + 14, y + 11 + k * 4));
    setText(doc, COL.muted);
    doc.setFont(FONT_MONO, "normal").setFontSize(7);
    a.evidence.forEach((ev, k) => doc.text(ev, PAGE_W - MARGIN_X - 2, y + 6 + k * 3.5, { align: "right" }));
    y += cardH + 4;
  });

  // Human-in-the-loop block
  y += 2;
  setFill(doc, [240, 238, 230]);
  doc.rect(MARGIN_X, y, CONTENT_W, 18, "F");
  setText(doc, COL.ink);
  doc.setFont(FONT_SANS, "bold").setFontSize(9);
  doc.text("Human-in-the-loop protocol.", MARGIN_X + 4, y + 6);
  setText(doc, COL.body);
  doc.setFont(FONT_SANS, "normal").setFontSize(8.5);
  const hl = wrap(
    doc,
    `No recommendation in this report is auto-delivered. DA Region IV-A reviews, revises, and signs each action before external notification. aiPHeed logs each sign-off for audit. This report is advisory; statutory classification of food-insecurity status remains with DA-FNRI.`,
    CONTENT_W - 8
  );
  hl.forEach((ln, i) => doc.text(ln, MARGIN_X + 4, y + 11 + i * 3.6));

  // Signatures
  y = PAGE_H - 50;
  drawHairline(doc, y);
  const colW = CONTENT_W / 2;
  setText(doc, COL.amber);
  doc.setFont(FONT_MONO, "normal").setFontSize(7);
  doc.text("REVIEWED BY  ·  DA REGION IV-A", MARGIN_X, y + 5);
  doc.text("PREPARED BY  ·  AIPHEED TEAM", MARGIN_X + colW, y + 5);
  setText(doc, COL.ink);
  doc.setFont(FONT_SANS, "bold").setFontSize(10);
  doc.text(`Provincial Director, ${scope.kind === "region" ? "CALABARZON" : scope.title}`, MARGIN_X, y + 11);
  doc.text("Destura · Esico · Melindo", MARGIN_X + colW, y + 11);

  drawRunningFooter(doc, scope, pageNum, "End of report", "Confidential until released  ·  for internal DA review");
}

// ─────────── helpers ───────────

function formatThousands(n: number): string {
  if (n >= 1000) {
    if (n >= 100000) return Math.round(n / 1000).toLocaleString();
    return (n / 1000).toFixed(1);
  }
  return n.toLocaleString();
}

function prettyDate(d: Date): string {
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function nextQuarterCheckDate(qid: string): string {
  // very lightweight — push +90 days from quarter id (e.g. "2026-Q3")
  const m = qid.match(/(\d{4})-Q(\d)/);
  if (!m) return "Sep 30, 2026";
  const year = parseInt(m[1]);
  const q = parseInt(m[2]);
  const months = [3, 6, 9, 12];
  const monthIdx = months[q - 1] - 1;
  const d = new Date(year, monthIdx, 15);
  return prettyDate(d);
}

function buildScopeCommon(quarterLabel: string, quarterId: string): Pick<ReportScope, "quarterLabel" | "quarterShort" | "monthsShort" | "quarterId"> {
  // quarterLabel like "Q3 2026 (Jul-Sep)"
  const m = quarterLabel.match(/(Q\d\s+\d{4})\s*\(([^)]+)\)/);
  return {
    quarterId,
    quarterLabel,
    quarterShort: m?.[1] ?? quarterLabel,
    monthsShort: m?.[2] ?? "",
  };
}

function reportNumber(qid: string, target: string) {
  const m = qid.match(/(\d{4})-Q(\d)/);
  const yy = m ? m[1].slice(2) : "00";
  const q = m ? m[2] : "0";
  return `Q${q}${yy}-${target.toUpperCase().slice(0, 3)}-01`;
}

// ─────────── public entry points ───────────

interface CommonOpts {
  quarterLabel: string;
  quarterId: string;
}

export function downloadRegionReport({ quarterLabel, quarterId }: CommonOpts) {
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  const totalPages = 4;
  const provinces = PROVINCE_QUARTER_DATA;
  const avg = provinces.reduce((s, p) => s + (p.scoresByQuarter[quarterId] ?? 0), 0) / provinces.length;
  const prevQid = previousQuarterId(quarterId);
  const prevAvg = provinces.reduce((s, p) => s + (p.scoresByQuarter[prevQid] ?? 0), 0) / provinces.length;
  const tier1 = provinces.filter((p) => (p.scoresByQuarter[quarterId] ?? 0) >= 0.65).length;
  const articles = provinces.reduce((s, p) => s + p.articles, 0);

  const scope: ReportScope = {
    ...buildScopeCommon(quarterLabel, quarterId),
    kind: "region",
    title: "CALABARZON",
    provinceId: null,
    subtitle: `*regional* food-insecurity outlook.`,
    probability: avg,
    delta: avg - prevAvg,
    atRisk: 612, // K
    populationPct: 4.1,
    tier1Count: tier1,
    totalMunis: 142,
    articles,
    reportNo: reportNumber(quarterId, "REG"),
    pageOf: totalPages,
    municipalRows: provinces.map((p) => ({
      name: p.name,
      pop: 1500000,
      poverty: 9.2,
      prob: p.scoresByQuarter[quarterId] ?? 0,
      delta: (p.scoresByQuarter[quarterId] ?? 0) - (p.scoresByQuarter[prevQid] ?? 0),
      articles: p.articles,
      tier: ((p.scoresByQuarter[quarterId] ?? 0) >= 0.65 ? 1 : (p.scoresByQuarter[quarterId] ?? 0) >= 0.5 ? 2 : 3) as 1 | 2 | 3,
    })),
  };

  drawCover(doc, scope);
  doc.addPage(); drawExecutiveSummary(doc, scope, 2);
  doc.addPage(); drawMunicipalSection(doc, scope, 3);
  doc.addPage(); drawActionsSection(doc, scope, 4);

  doc.save(`aipheed_calabarzon_${quarterId}.pdf`);
}

export function downloadProvinceReport({
  quarterLabel,
  quarterId,
  provinceId,
  provinceName,
}: CommonOpts & { provinceId: string; provinceName: string }) {
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  const row = PROVINCE_QUARTER_DATA.find((p) => p.id === provinceId);
  if (!row) {
    doc.text("Province not found", 20, 20);
    doc.save(`aipheed_${provinceId}_${quarterId}.pdf`);
    return;
  }
  const score = row.scoresByQuarter[quarterId] ?? 0;
  const prevQid = previousQuarterId(quarterId);
  const prev = row.scoresByQuarter[prevQid] ?? score;
  const tier1Count = provinceId === "quezon" ? 6 : score >= 0.65 ? 3 : score >= 0.5 ? 1 : 0;

  const municipalRows =
    provinceId === "quezon"
      ? QUEZON_TOP_MUNIS.map((m, i) => ({
          name: m.name,
          pop: [89400, 34200, 43800, 30900, 15700, 83100][i] ?? 50000,
          poverty: [12.8, 15.1, 11.2, 14.7, 13.4, 10.8][i] ?? 11,
          prob: m.score + 0.18,
          delta: 0.21 - i * 0.02,
          articles: [47, 38, 41, 29, 18, 34][i] ?? 25,
          tier: 1 as 1 | 2 | 3,
        })).concat(
          [
            { name: "Mauban", pop: 57200, poverty: 9.6, prob: 0.72, delta: 0.11, articles: 22, tier: 2 as 1 | 2 | 3 },
            { name: "Gumaca", pop: 80600, poverty: 9.1, prob: 0.68, delta: 0.09, articles: 27, tier: 2 as 1 | 2 | 3 },
            { name: "Tayabas", pop: 112800, poverty: 7.3, prob: 0.61, delta: 0.08, articles: 19, tier: 3 as 1 | 2 | 3 },
            { name: "Lucena City", pop: 278900, poverty: 4.2, prob: 0.54, delta: 0.06, articles: 42, tier: 3 as 1 | 2 | 3 },
          ]
        )
      : [
          { name: provinceName + " (capital)", pop: 180000, poverty: 8.4, prob: score + 0.06, delta: score - prev + 0.02, articles: 28, tier: (score >= 0.65 ? 1 : 2) as 1 | 2 | 3 },
          { name: provinceName + " North", pop: 120000, poverty: 11.2, prob: score + 0.02, delta: score - prev, articles: 22, tier: (score >= 0.6 ? 2 : 3) as 1 | 2 | 3 },
          { name: provinceName + " South", pop: 95000, poverty: 9.8, prob: score - 0.02, delta: score - prev - 0.01, articles: 18, tier: 3 as const },
        ];

  const scope: ReportScope = {
    ...buildScopeCommon(quarterLabel, quarterId),
    kind: "province",
    title: `${provinceName} Province`,
    provinceId,
    subtitle: score >= 0.7 ? "*critical* food stress forecast." : score >= 0.55 ? "*elevated* risk forecast." : "*stable* outlook with watchpoints.",
    probability: score,
    delta: score - prev,
    atRisk: provinceId === "quezon" ? 217 : Math.round(score * 240),
    populationPct: 10.2,
    tier1Count,
    totalMunis: provinceId === "quezon" ? 42 : 30,
    articles: row.articles,
    reportNo: reportNumber(quarterId, provinceName.slice(0, 3)),
    pageOf: 4,
    municipalRows,
  };

  drawCover(doc, scope);
  doc.addPage(); drawExecutiveSummary(doc, scope, 2);
  doc.addPage(); drawMunicipalSection(doc, scope, 3);
  doc.addPage(); drawActionsSection(doc, scope, 4);

  doc.save(`aipheed_${provinceId}_${quarterId}.pdf`);
}

export function downloadMunicipalityReport({
  quarterLabel,
  quarterId,
  name,
  provinceName,
  score,
}: CommonOpts & { name: string; provinceName: string; score: number }) {
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  const prev = score - 0.05;
  const scope: ReportScope = {
    ...buildScopeCommon(quarterLabel, quarterId),
    kind: "municipality",
    title: name,
    provinceId: PROVINCE_QUARTER_DATA.find((p) => p.name.toLowerCase() === provinceName.toLowerCase())?.id ?? null,
    subtitle: score >= 0.7 ? "*critical* municipal advisory." : score >= 0.55 ? "*elevated* municipal watch." : "*baseline* municipal outlook.",
    probability: score,
    delta: 0.05,
    atRisk: Math.round(score * 60),
    populationPct: 12.4,
    tier1Count: 1,
    totalMunis: 1,
    articles: Math.round(score * 60),
    reportNo: reportNumber(quarterId, name.slice(0, 3)),
    pageOf: 3,
    municipalRows: [
      { name, pop: 89400, poverty: 12.8, prob: score, delta: 0.21, articles: 47, tier: (score >= 0.65 ? 1 : score >= 0.5 ? 2 : 3) as 1 | 2 | 3 },
      { name: `Adjacent A`, pop: 34200, poverty: 15.1, prob: score - 0.05, delta: 0.18, articles: 22, tier: 2 as const },
      { name: `Adjacent B`, pop: 43800, poverty: 11.2, prob: score - 0.08, delta: 0.14, articles: 19, tier: 2 as const },
    ],
  };

  drawCover(doc, scope);
  doc.addPage(); drawExecutiveSummary(doc, scope, 2);
  doc.addPage(); drawActionsSection(doc, scope, 3);

  doc.save(`aipheed_${name.toLowerCase().replace(/\s+/g, "_")}_${quarterId}.pdf`);
}

function previousQuarterId(qid: string): string {
  const m = qid.match(/(\d{4})-Q(\d)/);
  if (!m) return qid;
  let y = parseInt(m[1]);
  let q = parseInt(m[2]);
  q -= 1;
  if (q === 0) { q = 4; y -= 1; }
  return `${y}-Q${q}`;
}
