"""
update_section_3_5.py
---------------------
Rewrite the prose paragraphs of section_3_5_with_diagrams_v2.docx to:
  1. Reflect the actual aiPHeed implementation (composite SWS+PSA stress label,
     FIES as FAO SDG 2.1.2 anchor only, 11 primary sources, 45 features, etc.)
  2. Add explicit conditions / decision branches to the system flowchart text
  3. Improve clarity and readability for a thesis defense audience
  4. Preserve all three embedded figures (use case, flowchart, state diagram)

Run:
    python scripts/update_section_3_5.py
"""
from pathlib import Path
from docx import Document

SRC = Path(r"C:\Users\admin\Downloads\section_3_5_with_diagrams_v2.docx")
DST = Path(r"C:\Users\admin\Downloads\section_3_5_with_diagrams_v3.docx")

# ─────────────────────────────────────────────────────────────────────────────
# Rewritten paragraphs (figure captions and image paragraphs are untouched)
# ─────────────────────────────────────────────────────────────────────────────

USE_CASE_TEXT = (
    "Who Uses aiPHeed. As shown in Figure 3.1, aiPHeed has three users: the Admin "
    "(reviewer), the DSWD Staff (end user), and the Auto Scheduler (system). The "
    "Admin can start a forecast manually, review new forecasts before they are "
    "shown to the public, and approve or reject early warnings. The DSWD Staff, "
    "who is the main end user, can view the food insecurity risk for each of the "
    "five CALABARZON provinces, see the risk for all 142 cities and towns of "
    "Region IV-A, download a PDF report each quarter, view active warnings that "
    "have already been approved, and view past forecasts for trend analysis. The "
    "Auto Scheduler runs the forecast on a fixed schedule without anyone pressing "
    "a button. No user inside the system declares an official food insecurity "
    "result, because aiPHeed only gives risk indicators and does not make final "
    "decisions for DSWD."
)

FLOWCHART_PART1 = (
    "How a Forecast Is Made. As shown in Figure 3.2, aiPHeed makes a forecast "
    "in ten steps, with four yes-or-no checks along the way. The pipeline first "
    "gets the latest government data, including prices, jobs, climate, and "
    "hunger surveys from PSA, BSP, DOE, PAGASA, SWS, and DOST-FNRI ENNS. It then "
    "collects new English and Filipino news articles from Philippine RSS feeds, "
    "Google News RSS, and the Wayback web archive."
)

FLOWCHART_PART2 = (
    "For each article, the system finds out which province the news is about "
    "(first check). If the article is about one of the five CALABARZON "
    "provinces, it is kept; if not, it is skipped. The system then groups the "
    "kept articles by province and quarter and weights them so that provinces "
    "with fewer articles are not under-represented. The second check asks "
    "whether there are at least five articles for that province-quarter. If "
    "yes, the cell is OK; if not, it is marked LOW DATA and the user sees a "
    "warning that the forecast for that cell is based on limited information."
)

FLOWCHART_PART3 = (
    "The system then reads each remaining article and scores how related it is "
    "to food problems (third check). Articles that clearly talk about food "
    "problems are used to compute the Food Stress Score for each province and "
    "quarter, together with a one-quarter and two-quarter look-back and a "
    "speed-of-change value. The articles are also tagged with five trigger "
    "types—market problems, climate, jobs, OFW remittances, and fish kill—and "
    "the share of articles per trigger type is computed for each province."
)

FLOWCHART_PART4 = (
    "All news signals and government data are combined into one big table "
    "(forty-five values per province per quarter). The trained model is loaded "
    "and used to predict the risk three months ahead for each province. The "
    "province-level risk is then broken down into city- and town-level risk "
    "for the 142 cities and towns of Region IV-A, with each result clearly "
    "labeled as a downscaled estimate (not a separate model output). The "
    "fourth and final check asks whether the province-level risk is above the "
    "warning level: if yes, a warning is sent to the Admin for review; if no, "
    "no warning is sent. All forecasts and warnings are reviewed by the Admin "
    "before they appear on the public dashboard."
)

STATE_PART1 = (
    "Life of a Forecast. As shown in Figure 3.3, every forecast goes through "
    "four stages. It starts in Waiting when the pipeline begins. Once the "
    "model produces a result, the forecast moves to For Review, where the Admin "
    "looks at it. The Admin then either approves it (the forecast moves to "
    "Published and becomes visible on the dashboard) or rejects it (the "
    "forecast goes back to Waiting for the next quarter's run). When a new "
    "forecast for the following quarter is published, the older one moves to "
    "Archived but stays accessible for historical trend analysis."
)

STATE_PART2 = (
    "Life of a Warning. A warning follows a similar path with one decision. "
    "When a province's risk goes above the warning level, a warning is created "
    "and placed in For Review. The Admin then either approves it, in which "
    "case it is shown on the dashboard, or rejects it, in which case it stays "
    "in the records for audit but is not shown publicly. This two-step review "
    "ensures that no forecast or warning reaches the public dashboard without "
    "a human decision, since aiPHeed is meant as an early-warning tool and not "
    "as an automatic decision-maker."
)

# Map original paragraph index → new text
REPLACEMENTS = {
    1:  USE_CASE_TEXT,
    5:  FLOWCHART_PART1,
    6:  FLOWCHART_PART2,
    7:  FLOWCHART_PART3,        # original had only 2 paragraphs of flowchart prose
    11: STATE_PART1,
    12: STATE_PART2,
}

# Original paragraph 7 will be replaced with FLOWCHART_PART3, and a new
# paragraph (FLOWCHART_PART4) needs to be inserted after it. We track that
# separately because python-docx makes new paragraph insertion fiddly.


def _set_paragraph_text(paragraph, new_text: str) -> None:
    """Replace a paragraph's text content while preserving its style."""
    # Remove existing runs
    for run in list(paragraph.runs):
        run.text = ""
    # Add new text in a single run inheriting the paragraph style
    if paragraph.runs:
        paragraph.runs[0].text = new_text
    else:
        paragraph.add_run(new_text)


def main() -> None:
    doc = Document(SRC)
    paragraphs = doc.paragraphs

    # Apply direct replacements
    for idx, new_text in REPLACEMENTS.items():
        if idx < len(paragraphs):
            _set_paragraph_text(paragraphs[idx], new_text)
            print(f"[updated] paragraph {idx}: {new_text[:80]}...")
        else:
            print(f"[skip]    paragraph {idx} out of range")

    # Insert FLOWCHART_PART4 immediately after paragraph 7 (now FLOWCHART_PART3)
    # and before paragraph 8 (originally a blank spacer).
    # python-docx insert pattern: copy element, place before target
    from copy import deepcopy
    from docx.oxml.ns import qn

    p7 = paragraphs[7]._element
    p8 = paragraphs[8]._element

    new_p = deepcopy(p7)
    # Clear the cloned text
    for t in new_p.findall(qn("w:r")):
        new_p.remove(t)

    p8.addprevious(new_p)

    # Set text on the new paragraph using a fresh python-docx Paragraph wrapper
    from docx.text.paragraph import Paragraph
    new_para = Paragraph(new_p, doc.paragraphs[8]._parent)
    new_para.add_run(FLOWCHART_PART4)
    print(f"[inserted] new flowchart paragraph after index 7")

    DST.parent.mkdir(parents=True, exist_ok=True)
    doc.save(DST)
    print(f"\nSaved -> {DST}")


if __name__ == "__main__":
    main()
