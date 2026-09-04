"""
app/services/report.py
-----------------------
Server-rendered PDF assessments, for a province or for CALABARZON as a whole.

Rendered here rather than in the browser so every copy of a given
subject-quarter is byte-comparable and the numbers come straight from the
model, not from whatever the client had cached.

Two deliberate omissions, both of which the orphaned frontend generator
included:

* No "recommended actions" section. There is no model output, rule table or
  DA-validated intervention mapping behind one. Generating operational advice
  and printing it under a government heading would be the single most harmful
  thing this file could do. `ACTIONS_PLACEHOLDER` marks where a DA-supplied
  protocol table would slot in.
* No municipality-scope report. Municipal values are the province figure
  reweighted by poverty and density, so a report about one LGU would restate
  its province's findings under a heading implying independent evidence. They
  appear as a ranked table inside the province report instead, where the
  derivation is stated on the same page.

reportlab, not WeasyPrint: WeasyPrint needs GTK system libraries that are not
present on Windows, and a report that only renders on the deploy host is not
much of a report.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.ml.inference.predictor import Predictor
from app.services import dashboard as svc
from app.services import reference as ref

logger = logging.getLogger(__name__)

ACTIONS_PLACEHOLDER = (
    "This assessment reports findings only. It does not recommend interventions: "
    "no validated mapping exists between this indicator and a response protocol. "
    "Pair it with the Department's own standing guidance."
)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

INK = colors.HexColor("#171C18")
INK_SOFT = colors.HexColor("#414A41")
MUTED = colors.HexColor("#6A7269")
RULE = colors.HexColor("#D8DCD5")
RULE_SOFT = colors.HexColor("#E7EAE5")
ACCENT = colors.HexColor("#2F6F4E")
RAISING = colors.HexColor("#A63A2B")
LOWERING = colors.HexColor("#2F6F4E")
WARN_BG = colors.HexColor("#F7EEDA")
WARN_INK = colors.HexColor("#8A6410")
PANEL = colors.HexColor("#F1F4F0")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body", parent=base["Normal"], fontName="Helvetica", fontSize=9.2,
        leading=13.4, textColor=INK_SOFT, alignment=TA_LEFT, spaceAfter=0,
    )
    return {
        "title": ParagraphStyle(
            "Title", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=20, leading=23, textColor=INK, spaceAfter=2),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=11.5, leading=15, textColor=MUTED),
        "eyebrow": ParagraphStyle(
            "Eyebrow", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=7.4, leading=10, textColor=MUTED),
        "h2": ParagraphStyle(
            "H2", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=12.2, leading=15, textColor=INK, spaceBefore=2, spaceAfter=3),
        "h3": ParagraphStyle(
            "H3", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=9.6, leading=13, textColor=INK),
        "body": body,
        "small": ParagraphStyle(
            "Small", parent=body, fontSize=8.1, leading=11.6, textColor=MUTED),
        "cell": ParagraphStyle(
            "Cell", parent=body, fontSize=8.4, leading=11.2),
        "cellb": ParagraphStyle(
            "CellB", parent=body, fontSize=8.4, leading=11.2,
            fontName="Helvetica-Bold", textColor=INK),
        "figure": ParagraphStyle(
            "Figure", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=30, leading=32, textColor=INK),
    }


S = _styles()


def _p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def _rule(thickness: float = 0.6, color=RULE, space_before=4, space_after=6) -> HRFlowable:
    return HRFlowable(width="100%", thickness=thickness, color=color,
                      spaceBefore=space_before, spaceAfter=space_after)


def _section(title: str) -> list:
    return [Spacer(1, 7 * mm), _p(title, "h2"), _rule()]


def _pct(value: float | None, digits: int = 1) -> str:
    return "--" if value is None else f"{value * 100:.{digits}f}%"


def _num(value: float | None, digits: int = 4) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def _signed(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:+.4f}"


# ---------------------------------------------------------------------------
# Small visual components
# ---------------------------------------------------------------------------

def _bar_table(rows: list[tuple[str, int, str, colors.Color]], width: float) -> Table:
    """
    Ranked horizontal bars: label, bar, value.

    Drawn as nested tables rather than a chart object so the whole block
    paginates with the rest of the flow.
    """
    label_w, value_w = 46 * mm, 30 * mm
    track_w = width - label_w - value_w
    # A full-width bar would butt straight into the value column; hold back a
    # margin so even the longest one keeps a visible gap.
    usable_w = track_w * 0.94

    data = []
    for label, pct, note, color in rows:
        fill = max(usable_w * pct / 100.0, 0.6)
        bar = Table([[""]], colWidths=[fill], rowHeights=[3.4 * mm])
        bar.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), color),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        holder = Table([[bar, ""]], colWidths=[fill, max(track_w - fill, 0.1)])
        holder.setStyle(TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        data.append([_p(label, "cellb"), holder, _p(note, "cell")])

    table = Table(data, colWidths=[label_w, track_w, value_w])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, -1), 0),
        ("RIGHTPADDING", (1, 0), (1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE_SOFT),
    ]))
    return table


def _data_table(header: list[str], rows: list[list], widths: list[float],
                aligns: dict[int, str] | None = None) -> Table:
    aligns = aligns or {}
    data = [[_p(f"<b>{h}</b>", "small") for h in header]]
    data += [[cell if isinstance(cell, Paragraph) else _p(str(cell), "cell")
              for cell in row] for row in rows]

    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.35, RULE_SOFT),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (0, -1), 0),
    ]
    for col, align in aligns.items():
        style.append(("ALIGN", (col, 0), (col, -1), align))

    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle(style))
    return table


def _callout(title: str, body: str, background=PANEL, ink=INK) -> Table:
    inner = [[_p(f"<b>{title}</b>", "h3")], [_p(body, "small")]]
    table = Table(inner, colWidths=["100%"])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (0, 0), 6),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 6),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, ACCENT),
    ]))
    return table


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------

class _Canvas:
    """Header rule and footer, drawn on every page."""

    def __init__(self, subject: str, quarter: str):
        self.subject = subject
        self.quarter = quarter

    def __call__(self, canvas, doc):
        canvas.saveState()
        width, height = A4

        canvas.setFont("Helvetica-Bold", 7.2)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, height - 12 * mm, "aiPHeed")
        canvas.setFont("Helvetica", 7.2)
        canvas.drawString(32 * mm, height - 12 * mm,
                          f"{self.subject}  |  {self.quarter}")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(18 * mm, height - 14.5 * mm, width - 18 * mm, height - 14.5 * mm)

        canvas.setFont("Helvetica", 6.9)
        canvas.setFillColor(MUTED)
        canvas.drawString(
            18 * mm, 11 * mm,
            "Food availability shock share -- production shortfall against seasonal "
            "baseline. Not a household food-insecurity probability.")
        canvas.drawRightString(width - 18 * mm, 11 * mm, f"Page {doc.page}")
        canvas.restoreState()


# ---------------------------------------------------------------------------
# Content builders
# ---------------------------------------------------------------------------

def _quarter_label(quarter: str) -> str:
    year, qn = ref.quarter_parts(quarter)
    return f"Q{qn} {year} ({ref.QUARTER_MONTHS[qn]})"


def _headline(name: str, score: float | None, level: str | None,
              qoq: float | None, extra: list[tuple[str, str]], width: float) -> list:
    left = [
        _p("FOOD AVAILABILITY SHOCK SHARE", "eyebrow"),
        Spacer(1, 1.5 * mm),
        _p(_num(score, 3) if score is not None else "--", "figure"),
        _p(f"{(level or 'n/a').upper()}  ·  quarter-on-quarter {_signed(qoq)}", "small"),
    ]
    right_rows = [[_p(k, "small"), _p(f"<b>{v}</b>", "cell")] for k, v in extra]
    right = _data_table([], right_rows, [40 * mm, 38 * mm]) if right_rows else Spacer(1, 1)

    block = Table([[left, right]], colWidths=[width * 0.45, width * 0.55])
    block.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (0, -1), 0),
    ]))
    return [block]


def _drivers_block(explain: dict, subject: str, quarter: str, width: float) -> list:
    rows = []
    for trigger in explain["triggers"]:
        color = RAISING if trigger["direction"] == "increases_risk" else LOWERING
        arrow = "raises" if trigger["direction"] == "increases_risk" else "lowers"
        rows.append((trigger["label"], trigger["pct"],
                     f"{trigger['pct']}%  {arrow}", color))

    return [
        _bar_table(rows, width),
        Spacer(1, 3 * mm),
        _p(
            "<b>Red</b> bars raise the predicted shortfall, <b>green</b> lower it. "
            "A large bar is not automatically bad news -- share of the explanation "
            "and direction of effect are separate things.", "small"),
        Spacer(1, 2.5 * mm),
        _p(svc.compose_narrative(subject, quarter, explain["triggers"]), "body"),
    ]


def _method_block(performance: dict) -> list:
    baselines = performance.get("baselines", {})
    skill = performance.get("skill", {})
    rows = [
        ["Model accuracy", _num(performance.get("accuracy"), 4)],
        ["Majority-class baseline", _num(baselines.get("majorityClass"), 4)],
        ["Seasonal-persistence baseline", _num(baselines.get("seasonalPersistence"), 4)],
        ["Skill over majority class", _signed(skill.get("vsMajority"))],
        ["Skill over seasonal persistence", _signed(skill.get("vsSeasonal"))],
        ["ROC AUC", _num(performance.get("rocAuc"), 4)],
        ["Evaluation set size", str(performance.get("n", "--"))],
        ["Referred to analyst review", f"{performance.get('abstainPct', '--')}%"],
    ]
    return [
        _p(
            "The model predicts, for each monitored commodity series in a province, "
            "whether that quarter's production falls more than 10% below the series' "
            "own expanding seasonal baseline. The province figure is the share of its "
            "series flagged. Inputs are PSA OpenStat production volumes, PSA and BSP "
            "macroeconomic indicators, PAGASA climate variables, and a geocoded news "
            "corpus.", "body"),
        Spacer(1, 3 * mm),
        _data_table(["Measure", "Value"], rows, [78 * mm, 30 * mm], {1: "RIGHT"}),
        Spacer(1, 3 * mm),
        _p(
            "<b>Accuracy must be read against its baselines.</b> Always predicting the "
            "majority class already scores "
            f"{_num(baselines.get('majorityClass'), 3)}, and repeating what the same "
            "quarter did last year scores "
            f"{_num(baselines.get('seasonalPersistence'), 3)}. The model's advantage "
            "over seasonal persistence is small; its value is in flagging which "
            "series and which province, not in beating the calendar by a wide "
            "margin.", "small"),
    ]


def _limitations_block(limited_signal: bool, article_count: int,
                       includes_municipal: bool) -> list:
    items = [
        ("Same-quarter nowcast, not a forecast",
         "Figures cover a quarter whose production volumes PSA has already published. "
         "The system does not currently project forward: there is no forecast quarter."),
        ("Availability, not household hunger",
         "A production shortfall is a supply-side signal. It does not measure whether "
         "households went hungry, and it should not be read as a hunger estimate."),
    ]
    if limited_signal:
        items.append((
            "Limited news signal",
            f"Only {article_count} relevant articles were geocoded to this subject for "
            "the quarter, below the threshold for a dependable news component. The "
            "news-derived drivers carry correspondingly little weight."))
    if includes_municipal:
        items.append((
            "Municipal figures are disaggregated, not modelled",
            "Municipal values reweight the province figure by poverty incidence (60%) "
            "and population density (40%). Poverty values are largely province means "
            "rather than municipal Small Area Estimates, so within-province ordering "
            "is indicative only."))
    items.append((
        "Reviewed before publication",
        "Province-quarter figures pass through analyst review. A withheld figure is "
        "excluded from this report and from the regional average."))

    flow = []
    for title, text in items:
        flow.append(KeepTogether([_p(f"<b>{title}</b>", "h3"), Spacer(1, 0.8 * mm),
                                  _p(text, "small"), Spacer(1, 3 * mm)]))
    return flow


def _news_block(news: dict, width: float) -> list:
    if not news["articleCount"]:
        return [_p("No relevant articles were geocoded to this subject for the quarter.",
                   "small")]

    unlocated = news["articleCount"] - news["provinceAttributed"]
    intro = f"{news['articleCount']} relevant articles were analysed for this quarter."
    if unlocated > 0:
        intro += (
            f" Of these, {news['provinceAttributed']} could be located to a specific "
            f"province; the remaining {unlocated} mention the region without naming "
            "one, so they inform the regional picture but no individual province.")
    flow = [_p(intro, "body"), Spacer(1, 3 * mm)]

    if news["topics"]:
        flow.append(_bar_table(
            [(t["label"], t["pct"], f"{t['count']} ({t['pct']}%)", ACCENT)
             for t in news["topics"][:6]], width))
        flow.append(Spacer(1, 4 * mm))

    if news["data"]:
        # By classifier confidence, not recency. The corpus carries false
        # positives, and the most recent article is often a weak match -- which
        # in a DA-facing document reads as the system not knowing what food
        # insecurity is.
        ranked = sorted(
            news["data"],
            key=lambda a: (a.get("relevanceScore") or 0),
            reverse=True,
        )[:6]
        flow.append(_p("Strongest-signal headlines", "h3"))
        flow.append(Spacer(1, 1.5 * mm))
        rows = [[_p(a["date"] or "--", "cell"),
                 _p(f"<b>{_escape(a['title'])}</b><br/>"
                    f"<font color='#6A7269'>{_escape(a['source'])} · {_escape(a['topicLabel'])}</font>",
                    "cell"),
                 _p(_num(a.get("relevanceScore"), 2), "cell")]
                for a in ranked]
        flow.append(_data_table(
            ["Date", "Headline", "Score"], rows,
            [20 * mm, width - 44 * mm, 24 * mm], {2: "RIGHT"}))
        flow.append(Spacer(1, 2 * mm))
        flow.append(_p(
            "Score is the classifier's food-insecurity relevance for the article, "
            "0-1. The corpus is machine-selected and not hand-verified; treat "
            "individual headlines as leads rather than findings.", "small"))
    return flow


def _escape(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _cover(title: str, subject: str, quarter: str, width: float) -> list:
    generated = datetime.now(timezone.utc).astimezone().strftime("%d %B %Y")
    return [
        _p("DEPARTMENT OF AGRICULTURE · CALABARZON", "eyebrow"),
        Spacer(1, 2 * mm),
        _p(title, "title"),
        _p(f"{subject} · {_quarter_label(quarter)}", "subtitle"),
        Spacer(1, 3 * mm),
        _rule(1.2, INK, 0, 4),
        _p(f"Generated {generated} from aiPHeed model output.", "small"),
        Spacer(1, 5 * mm),
        _callout(
            "What this number is",
            "The share of this area's monitored PSA commodity series predicted to fall "
            "more than 10% below that series' own expanding seasonal baseline. "
            "<b>It is a food availability signal, not a household food-insecurity "
            "probability</b>, and it describes a quarter that has already occurred "
            "rather than one ahead."),
    ]


def build_province_report(province_id: str, quarter: str,
                          withheld: frozenset[str] = frozenset()) -> bytes:
    """A single province: headline, drivers, commodities, LGUs, news, method."""
    slug = province_id.lower()
    if slug not in ref.PROVINCES:
        raise svc.SubjectNotFound(f"Unknown province id '{province_id}'.")
    if slug in withheld:
        raise svc.ForecastWithheld(
            f"The {quarter} assessment for this province has been withheld from "
            "publication by a reviewer."
        )

    provinces = svc.province_summary(quarter, withheld)
    province = next((p for p in provinces if p["id"] == slug), None)
    if province is None:
        raise svc.SubjectNotFound(f"Unknown province id '{province_id}'.")

    code = province["provinceCode"]
    explain = svc.explainability(code, quarter)
    municipalities = svc.municipality_summary(quarter, province_id=slug)
    news = svc.news(code, quarter, 1, 60)

    published = [p for p in provinces if not p["withheld"]]
    rank = sorted(published, key=lambda p: p["riskScore"], reverse=True)
    position = next((i + 1 for i, p in enumerate(rank) if p["id"] == slug), None)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=20 * mm, bottomMargin=18 * mm,
        title=f"aiPHeed {province['name']} {quarter}",
        author="aiPHeed", subject="Food availability shock assessment",
    )
    width = doc.width

    flow = _cover("Food Availability Shock Assessment", province["name"], quarter, width)

    flow += _section("Headline")
    flow += _headline(
        province["name"], province["riskScore"], province["riskLevel"],
        province["qoqChange"],
        [
            ("Series flagged at risk",
             f"{province['seriesAtRisk']} of {province['seriesMonitored']}"),
            ("Rank in CALABARZON", f"{position} of {len(published)}" if position else "--"),
            ("Population (2020 CPH)", f"{province['population']:,}"),
            ("Poverty incidence (PSA 2021)", f"{province['povertyRate']}%"),
            ("Articles analysed", str(province["articleCount"])),
        ],
        width)

    if province["limitedSignal"]:
        flow += [Spacer(1, 4 * mm), _callout(
            "Limited signal",
            f"Only {province['articleCount']} relevant news articles were geocoded to "
            "this province for the quarter. News-derived drivers should be treated as "
            "weak evidence.", WARN_BG)]

    if province["topAtRiskCommodities"]:
        flow += _section("Commodity series at risk")
        commodities = [
            c for c in Predictor().forecast_commodities(quarter)
            if c["province_code"] == code and c["at_risk"]
        ]
        flow += [_p(
            f"{len(commodities)} of {province['seriesMonitored']} monitored series are "
            "predicted to fall below their seasonal baseline. The highest-probability "
            "series are listed first.", "body"), Spacer(1, 3 * mm)]
        rows = [[_p(_escape(c["commodity"]), "cellb"),
                 _p(c["commodity_group"].replace("_", " "), "cell"),
                 _p(_num(c["shock_probability"], 3), "cell")]
                for c in commodities[:18]]
        flow.append(_data_table(
            ["Commodity series", "Group", "Shock probability"],
            rows, [width - 62 * mm, 32 * mm, 30 * mm], {2: "RIGHT"}))

    flow.append(PageBreak())

    flow += _section("What is driving the figure")
    flow += _drivers_block(explain, province["name"], quarter, width)

    flow += _section("Municipal distribution")
    flow += [_p(
        "Municipal values reweight the province figure by poverty incidence (60%) and "
        "population density (40%). <b>They are not independent municipal forecasts</b> "
        "-- every municipality in a province rises and falls with it.", "body"),
        Spacer(1, 3 * mm)]
    muni_rows = [[_p(f"{i + 1}", "cell"), _p(_escape(m["name"]), "cellb"),
                  _p(m["classification"], "cell"),
                  _p(f"{m['population']:,}", "cell"),
                  _p("--" if m["povertyRate"] is None else f"{m['povertyRate']}%", "cell"),
                  _p(_num(m["riskIndex"], 4), "cell")]
                 for i, m in enumerate(municipalities)]
    flow.append(_data_table(
        ["#", "City / Municipality", "Class", "Population", "Poverty", "Index"],
        muni_rows,
        [8 * mm, width - 108 * mm, 24 * mm, 26 * mm, 22 * mm, 28 * mm],
        {3: "RIGHT", 4: "RIGHT", 5: "RIGHT"}))

    flow.append(PageBreak())
    flow += _section("News evidence")
    flow += _news_block(news, width)

    flow += _section("Method")
    flow += _method_block(svc.model_performance())

    flow += _section("Limitations")
    flow += _limitations_block(province["limitedSignal"], province["articleCount"], True)

    flow += [Spacer(1, 3 * mm), _callout("Scope of this assessment", ACTIONS_PLACEHOLDER)]

    doc.build(flow, onFirstPage=_Canvas(province["name"], quarter),
              onLaterPages=_Canvas(province["name"], quarter))
    return buf.getvalue()


def build_region_report(quarter: str, withheld: frozenset[str] = frozenset()) -> bytes:
    """CALABARZON: regional headline, province comparison, drivers, news, method."""
    region = svc.region_forecast(quarter, withheld)
    provinces = svc.province_summary(quarter, withheld)
    published = [p for p in provinces if not p["withheld"]]
    explain = svc.explainability_region(quarter)
    news = svc.news(None, quarter, 1, 60)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=20 * mm, bottomMargin=18 * mm,
        title=f"aiPHeed CALABARZON {quarter}",
        author="aiPHeed", subject="Food availability shock assessment",
    )
    width = doc.width

    flow = _cover("Regional Food Availability Assessment", ref.REGION_NAME, quarter, width)

    total_monitored = sum(p["seriesMonitored"] or 0 for p in published)
    total_at_risk = sum(p["seriesAtRisk"] or 0 for p in published)

    flow += _section("Headline")
    flow += _headline(
        ref.REGION_NAME, region["riskScore"], region["riskLevel"], region["qoqChange"],
        [
            ("Provinces covered", f"{region['provincesIncluded']} of {len(provinces)}"),
            ("Series flagged at risk", f"{total_at_risk} of {total_monitored}"),
            ("Provinces above cutoff", str(region["provinceCounts"]["high"])),
            ("Provinces withheld", str(region["provinceCounts"]["withheld"])),
            ("Articles analysed",
             f"{news['articleCount']} ({news['provinceAttributed']} located)"),
        ],
        width)
    flow += [Spacer(1, 3 * mm), _p(
        f"The regional figure is the mean of {region['provincesIncluded']} published "
        "province scores. The model has no region-level series of its own. A province "
        "withheld by a reviewer is excluded from the mean rather than counted as "
        "zero.", "small")]

    if region["provinceCounts"]["withheld"]:
        withheld_names = ", ".join(p["name"] for p in provinces if p["withheld"])
        flow += [Spacer(1, 4 * mm), _callout(
            "Withheld from this assessment",
            f"{withheld_names} -- excluded by analyst review for this quarter. Figures "
            "below cover the remaining provinces only.", WARN_BG)]

    flow += _section("Provinces compared")
    top = max((p["riskScore"] for p in published), default=0) or 1
    flow.append(_bar_table(
        [(p["name"], int(round(p["riskScore"] / top * 100)),
          f"{_num(p['riskScore'], 3)}  {p['riskLevel']}", ACCENT)
         for p in sorted(published, key=lambda x: x["riskScore"], reverse=True)],
        width))
    flow.append(Spacer(1, 4 * mm))

    rows = [[_p(_escape(p["name"]), "cellb"),
             _p(_num(p["riskScore"], 4), "cell"),
             _p((p["riskLevel"] or "--").upper(), "cell"),
             _p(f"{p['seriesAtRisk']}/{p['seriesMonitored']}", "cell"),
             _p(_signed(p["qoqChange"]), "cell"),
             _p(f"{p['povertyRate']}%", "cell"),
             _p(str(p["articleCount"]), "cell")]
            for p in sorted(published, key=lambda x: x["riskScore"], reverse=True)]
    flow.append(_data_table(
        ["Province", "Index", "Level", "At risk", "QoQ", "Poverty", "Articles"],
        rows,
        [width - 118 * mm, 20 * mm, 18 * mm, 20 * mm, 22 * mm, 20 * mm, 18 * mm],
        {1: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT", 6: "RIGHT"}))

    flow += [Spacer(1, 4 * mm), _p(
        "Poverty incidence and availability shock are different signals and do not "
        "track together. A province may hold the region's highest poverty rate and its "
        "lowest production shortfall in the same quarter.", "small")]

    flow.append(PageBreak())

    flow += _section("What is driving the regional figure")
    flow += _drivers_block(explain, ref.REGION_NAME, quarter, width)

    flow += _section("News evidence")
    flow += _news_block(news, width)

    flow += _section("Method")
    flow += _method_block(svc.model_performance())

    flow += _section("Limitations")
    limited = all(p["limitedSignal"] for p in published) if published else True
    flow += _limitations_block(limited, news["articleCount"], False)

    flow += [Spacer(1, 3 * mm), _callout("Scope of this assessment", ACTIONS_PLACEHOLDER)]

    doc.build(flow, onFirstPage=_Canvas(ref.REGION_NAME, quarter),
              onLaterPages=_Canvas(ref.REGION_NAME, quarter))
    return buf.getvalue()


def filename(scope: str, subject_id: str, quarter: str) -> str:
    stem = ref.REGION_ID if scope == "region" else subject_id.lower()
    return f"aipheed-{stem}-{quarter}.pdf"
