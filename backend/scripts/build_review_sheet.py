"""
build_review_sheet.py
---------------------
Generate the DOST Region IV-A expert-review spreadsheet from the curated
CALABARZON food-insecurity dataset.

Output: an .xlsx with
  * Sheet 1 "Instructions" — the 4 questions + A-F key (mirrors the one-page guide)
  * Sheet 2 "Review"       — one row per article: reference info + the machine's
                             proposed labels (grey) + empty reviewer columns
                             (yellow) with dropdown validation.

Reviewers only fill the yellow columns. The machine columns are shown so they
can confirm/correct rather than label from scratch.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "calabarzon_food_insecurity_dataset.parquet"
OUT = ROOT / "data" / "processed" / "DOST4A_review_sheet.xlsx"

# ---------------------------------------------------------------------------
# Palette / styles
# ---------------------------------------------------------------------------
NAVY = "1F3A5F"
GREY_FILL = PatternFill("solid", fgColor="EDEDED")      # machine columns
YELLOW_FILL = PatternFill("solid", fgColor="FFF3C4")    # reviewer columns
HEAD_FILL = PatternFill("solid", fgColor=NAVY)
HEAD_REV_FILL = PatternFill("solid", fgColor="B8860B")  # reviewer header (dark gold)
HEAD_FONT = Font(color="FFFFFF", bold=True, size=10)
THIN = Side(style="thin", color="C9C9C9")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP_TOP = Alignment(wrap_text=True, vertical="top")
CENTER = Alignment(horizontal="center", vertical="top")

# ---------------------------------------------------------------------------
# Load + order
# ---------------------------------------------------------------------------
df = pd.read_parquet(SRC)

# Sort so the softest / needs-review rows surface first for careful eyes, then
# by province and date for a tidy read.
df["needs_review"] = df.get("needs_review", False).fillna(False)
df = df.sort_values(
    ["needs_review", "province", "publication_date"],
    ascending=[False, True, True],
).reset_index(drop=True)


def clean(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


# Column plan: (header, kind)  kind in {ref, machine, reviewer}
COLUMNS = [
    ("#", "ref"),
    ("Article ID", "ref"),
    ("Date", "ref"),
    ("Province", "ref"),
    ("City / Municipality", "ref"),
    ("Headline", "ref"),
    ("Opening lines (lead)", "ref"),
    ("Source", "ref"),
    ("Link", "ref"),
    # --- machine's proposed labels (for reference) ---
    ("SYS: Relevant?", "machine"),
    ("SYS: Dimension", "machine"),
    ("SYS: Category", "machine"),
    ("SYS: Event type", "machine"),
    # --- reviewer fills these ---
    ("Q1  Keep / Drop", "reviewer"),
    ("Q2  Dimension (A-F)", "reviewer"),
    ("Commodity (optional)", "reviewer"),
    ("Needs review?", "reviewer"),
    ("Reviewer notes", "reviewer"),
    ("Reviewer initials", "reviewer"),
]

WIDTHS = {
    "#": 5, "Article ID": 14, "Date": 12, "Province": 12,
    "City / Municipality": 18, "Headline": 46, "Opening lines (lead)": 60,
    "Source": 16, "Link": 16, "SYS: Relevant?": 11, "SYS: Dimension": 12,
    "SYS: Category": 26, "SYS: Event type": 20,
    "Q1  Keep / Drop": 13, "Q2  Dimension (A-F)": 16, "Commodity (optional)": 16,
    "Needs review?": 13, "Reviewer notes": 34, "Reviewer initials": 12,
}

DIM_LETTER = {  # normalize machine dimension to a bare letter A-F when possible
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F",
}

wb = Workbook()

# ===========================================================================
# Sheet 1 — Instructions
# ===========================================================================
ws0 = wb.active
ws0.title = "Instructions"
ws0.sheet_view.showGridLines = False
ws0.column_dimensions["A"].width = 3
ws0.column_dimensions["B"].width = 110

lines = [
    ("h1", "DOST Region IV-A — Expert Review Sheet"),
    ("h2", "aiPHeed CALABARZON Food-Insecurity News Dataset"),
    ("sp", ""),
    ("p",  "The system already collected these news articles and tagged each one automatically. "
           "Your job is to CHECK those tags. You are grading pre-filled forms, not labeling from scratch. "
           "Your reviewed answers become the official answer key ('gold standard') used to measure the "
           "system's accuracy."),
    ("sp", ""),
    ("p",  "Go to the 'Review' tab. Read the Headline + Opening lines. The grey 'SYS:' columns show what "
           "the machine proposed. Fill only the YELLOW columns (Q1-Q2 + notes). Each yellow cell has a "
           "drop-down — click the cell and pick a value."),
    ("sp", ""),
    ("b",  "Q1  Keep / Drop  — Is this really about food insecurity?"),
    ("i",  "     Keep = real evidence about food access, supply, hunger, nutrition, farming, fisheries, or "
           "food aid.  Drop = incidental 'food' word in a non-food story (restaurant, recipe, 'solar farm', agri-tourism)."),
    ("b",  "Q2  Dimension (A-F)  — pick the ONE the story is most about. Ask yourself the question in bold:"),
    ("i",  "     A · Availability  —  \"Did something reduce the actual supply of food (growing/catching/raising)?\""),
    ("i",  "          Food at the source: farms, fishing, livestock.  e.g. typhoon flattens rice fields, fish kill, "
           "ASF wipes out hogs, poor harvest, pests.  Think: less food EXISTS because production got hit."),
    ("i",  "     B · Access / Affordability  —  \"Is the food there, but people can't afford it or reach it?\""),
    ("i",  "          The wallet and the market. e.g. rice prices jump, vegetables 'mahal', poverty limits buying. "
           "Think: food is available, but out of reach because of money."),
    ("i",  "     C · Utilization / Nutrition  —  \"Is it about health, nutrition, or malnutrition?\""),
    ("i",  "          What food does once eaten. e.g. child malnutrition/stunting, feeding program, nutrition advisory. "
           "Think: having food isn't enough — is it keeping people healthy?"),
    ("i",  "     D · Stability (shocks)  —  \"Did a disaster or disruption suddenly cut off food access?\""),
    ("i",  "          The interruption itself. e.g. flood displaces families, roads cut off, transport strike blocks "
           "deliveries, unrest.  Think: a sudden event broke the normal flow of food."),
    ("i",  "     E · Hunger / Assistance  —  \"Is it about actual hunger, or giving out food help?\""),
    ("i",  "          The outcome or the response. e.g. families going hungry, DSWD food-pack / ayuda distribution, "
           "relief feeding.  Think: hunger is happening, or someone is stepping in to feed people."),
    ("i",  "     F · Livelihood  —  \"Did lost income or jobs threaten families' ability to buy food?\""),
    ("i",  "          Earning power: jobs, wages, OFW remittances. e.g. remittances drop, mass layoffs, fisherfolk lose "
           "income.  Think: the money that buys food dried up because of lost work."),
    ("sp", ""),
    ("b",  "The 3 mix-ups to watch for"),
    ("i",  "     A vs B (supply vs price):  'Typhoon destroyed the rice crop' = A  ·  'Rice prices rose after the typhoon' = B"),
    ("i",  "     A vs D (the loss vs the disruption):  focus on crops/harvest destroyed = A  ·  focus on people "
           "evacuated / roads cut / deliveries stopped = D"),
    ("i",  "     B vs F (no money to buy vs no income earned):  'Food too expensive for families' = B  ·  'Families "
           "lost jobs / remittances fell' = F"),
    ("sp", ""),
    ("b",  "Ground rules"),
    ("i",  "  • When unsure, set 'Needs review?' = Yes rather than guessing — we adjudicate those together."),
    ("i",  "  • Judge the article as written, not from outside knowledge."),
    ("i",  "  • One primary dimension per article."),
    ("i",  "  • Your call overrides the machine. Disagreeing is the point — that's how we prove and improve accuracy."),
    ("sp", ""),
    ("i",  "Full one-page guide: backend/docs/DOST4A_annotation_guide.md   ·   Questions -> aiPHeed research team."),
]

r = 1
for kind, text in lines:
    cell = ws0.cell(row=r, column=2, value=text)
    if kind == "h1":
        cell.font = Font(bold=True, size=16, color=NAVY)
    elif kind == "h2":
        cell.font = Font(bold=True, size=12, color="B8860B")
    elif kind == "b":
        cell.font = Font(bold=True, size=11, color="222222")
    elif kind == "i":
        cell.font = Font(size=10, color="333333")
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    elif kind == "p":
        cell.font = Font(size=10, color="222222")
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws0.row_dimensions[r].height = 42
    r += 1

# ===========================================================================
# Sheet 2 — Review
# ===========================================================================
ws = wb.create_sheet("Review")
ws.sheet_view.showGridLines = False
ws.freeze_panes = "A2"

# Header row
for ci, (header, kind) in enumerate(COLUMNS, start=1):
    c = ws.cell(row=1, column=ci, value=header)
    c.font = HEAD_FONT
    c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    c.fill = HEAD_REV_FILL if kind == "reviewer" else HEAD_FILL
    c.border = BORDER
    ws.column_dimensions[get_column_letter(ci)].width = WIDTHS[header]
ws.row_dimensions[1].height = 30

# Data rows
for i, (_, row) in enumerate(df.iterrows(), start=1):
    excel_row = i + 1
    dim = clean(row.get("food_security_dimension"))
    values = [
        i,
        clean(row.get("article_id")),
        clean(row.get("publication_date"))[:10],
        clean(row.get("province")),
        clean(row.get("city_municipality")),
        clean(row.get("title")),
        clean(row.get("content_lead")),
        clean(row.get("news_source")),
        clean(row.get("url")),
        # machine proposals
        "Yes" if row.get("is_direct_food_insecurity") else "borderline",
        DIM_LETTER.get(dim, dim),
        clean(row.get("food_insecurity_category")),
        clean(row.get("event_type")),
        # reviewer blanks
        "", "", "", "", "", "",
    ]
    for ci, (val, (header, kind)) in enumerate(zip(values, COLUMNS), start=1):
        c = ws.cell(row=excel_row, column=ci, value=val)
        c.border = BORDER
        c.alignment = WRAP_TOP
        if kind == "machine":
            c.fill = GREY_FILL
            c.font = Font(size=9, color="555555")
            c.alignment = CENTER if header != "SYS: Category" else WRAP_TOP
        elif kind == "reviewer":
            c.fill = YELLOW_FILL
        else:
            c.font = Font(size=9)
        if header == "Link" and val:
            c.hyperlink = val
            c.value = "open"
            c.font = Font(size=9, color="0563C1", underline="single")
    ws.row_dimensions[excel_row].height = 54

# ---------------------------------------------------------------------------
# Drop-down validations on reviewer columns
# ---------------------------------------------------------------------------
last = len(df) + 1
col_idx = {h: i for i, (h, _) in enumerate(COLUMNS, start=1)}


def add_dv(header: str, options: list[str]):
    letter = get_column_letter(col_idx[header])
    dv = DataValidation(
        type="list",
        formula1='"' + ",".join(options) + '"',
        allow_blank=True,
        showDropDown=False,  # False => arrow IS shown (openpyxl quirk)
    )
    dv.error = "Pick a value from the list."
    dv.prompt = "Choose one"
    ws.add_data_validation(dv)
    dv.add(f"{letter}2:{letter}{last}")


add_dv("Q1  Keep / Drop", ["Keep", "Drop"])
add_dv("Q2  Dimension (A-F)", ["A", "B", "C", "D", "E", "F", "n/a"])
add_dv("Needs review?", ["Yes", "No"])

wb.save(OUT)
print(f"Wrote {OUT}  ({len(df)} articles)")
