"""
generate_section_3_5_diagrams.py
---------------------------------
Generate three complete, thesis-quality diagrams for section 3.5:

  Figure 3.1 — Use Case Diagram   (3 actors, 11 use cases, association lines)
  Figure 3.2 — System Flowchart   (10 sequential steps + 4 decision diamonds)
  Figure 3.3 — State Diagram      (forecast 4 states + alert 3 states)

Output: data/figures/fig_3_1_usecase.png, fig_3_2_flowchart.png, fig_3_3_state.png

Then replace the 3 inline shapes in section_3_5_with_diagrams_v2.docx and write
section_3_5_with_diagrams_v3.docx in the user's Downloads folder.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse, Polygon

# ──────────────────────────────────────────────────────────────────────────────
# Output paths
# ──────────────────────────────────────────────────────────────────────────────
OUT_DIR = Path("data/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

USECASE_PNG   = OUT_DIR / "fig_3_1_usecase.png"
FLOWCHART_PNG = OUT_DIR / "fig_3_2_flowchart.png"
STATE_PNG     = OUT_DIR / "fig_3_3_state.png"

# Style constants
ACTOR_COLOR = "#2E5C8A"
USECASE_FILL = "#E8F0FA"
USECASE_EDGE = "#2E5C8A"
PROCESS_FILL = "#E8F0FA"
PROCESS_EDGE = "#2E5C8A"
DECISION_FILL = "#FFF4D6"
DECISION_EDGE = "#C49000"
STATE_FILL = "#E8F0FA"
STATE_EDGE = "#2E5C8A"
TERMINAL_FILL = "#D6E8D6"
TERMINAL_EDGE = "#3A7A3A"
ARROW_COLOR = "#444444"

FONT_TITLE = {"fontsize": 14, "fontweight": "bold"}
FONT_NODE  = {"fontsize": 9}
FONT_SMALL = {"fontsize": 8}


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3.1 — Use Case Diagram
# ══════════════════════════════════════════════════════════════════════════════

def draw_actor(ax, x, y, label):
    """Draw a stick-figure actor with a label below."""
    head_r = 0.18
    ax.add_patch(plt.Circle((x, y + 0.35), head_r, fill=False, lw=1.6, color=ACTOR_COLOR))
    ax.plot([x, x], [y + 0.17, y - 0.30], lw=1.6, color=ACTOR_COLOR)               # body
    ax.plot([x - 0.25, x + 0.25], [y + 0.05, y + 0.05], lw=1.6, color=ACTOR_COLOR) # arms
    ax.plot([x, x - 0.20], [y - 0.30, y - 0.65], lw=1.6, color=ACTOR_COLOR)        # left leg
    ax.plot([x, x + 0.20], [y - 0.30, y - 0.65], lw=1.6, color=ACTOR_COLOR)        # right leg
    ax.text(x, y - 0.85, label, ha="center", va="top",
            fontsize=10, fontweight="bold", color=ACTOR_COLOR)


def draw_usecase(ax, x, y, w, h, label):
    """Draw a usecase ellipse."""
    e = Ellipse((x, y), w, h, facecolor=USECASE_FILL,
                edgecolor=USECASE_EDGE, lw=1.4)
    ax.add_patch(e)
    ax.text(x, y, label, ha="center", va="center", **FONT_NODE)
    return (x, y, w, h)


def draw_assoc(ax, ax_pt, uc_pt, uc_w):
    """Draw a thin association line from actor to use case ellipse left edge."""
    x0, y0 = ax_pt
    x1, y1 = uc_pt
    # Connect to nearest edge of ellipse on the side facing the actor
    if x0 < x1:
        x1 -= uc_w / 2
    else:
        x1 += uc_w / 2
    ax.plot([x0, x1], [y0, y1], lw=1.0, color=ARROW_COLOR)


def make_usecase_diagram():
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(7, 10.5, "aiPHeed - Who Does What",
            ha="center", va="center", **FONT_TITLE)

    # System boundary
    boundary = FancyBboxPatch((4.0, 0.7), 6.0, 9.0,
                              boxstyle="round,pad=0.05,rounding_size=0.15",
                              facecolor="white", edgecolor="#888888",
                              lw=1.0, linestyle="--")
    ax.add_patch(boundary)
    ax.text(7, 9.5, "aiPHeed System", ha="center", va="center",
            fontsize=10, color="#666666", style="italic")

    # Actors (left = admin & dswd staff, right = scheduler)
    sys_admin = (1.5, 8.2)
    evaluator = (1.5, 3.5)
    scheduler = (12.5, 5.8)

    draw_actor(ax, *sys_admin, "System Admin\n(IT Unit)")
    draw_actor(ax, *evaluator, "DSWD Staff\n(DRRM Unit)")
    draw_actor(ax, *scheduler, "Auto Scheduler\n(System)")

    # Use cases (centered in boundary) — 10 use cases including SHAP
    uc_w, uc_h = 2.6, 0.7
    use_cases = {
        "uc1":  (7.0, 8.7, "Start the Forecast"),
        "uc2":  (7.0, 7.9, "Review New Forecasts"),
        "uc3":  (7.0, 7.1, "Confirm or Dismiss Alerts"),
        "uc4":  (7.0, 6.3, "Run Forecast Automatically"),
        "uc5":  (7.0, 5.5, "View Province Risk"),
        "uc6":  (7.0, 4.7, "View City and Town Risk"),
        "uc7":  (7.0, 3.9, "View Feature Explanations (SHAP)"),
        "uc8":  (7.0, 3.1, "Download Report (PDF)"),
        "uc9":  (7.0, 2.3, "View Active Warnings"),
        "uc10": (7.0, 1.5, "View Past Forecasts"),
    }
    for k, (x, y, label) in use_cases.items():
        draw_usecase(ax, x, y, uc_w, uc_h, label)

    # Associations
    # System Admin → uc1 only (technical startup)
    for k in ("uc1",):
        x, y, _ = use_cases[k]
        draw_assoc(ax, sys_admin, (x, y), uc_w)

    # Automated Scheduler → uc4
    for k in ("uc4",):
        x, y, _ = use_cases[k]
        draw_assoc(ax, scheduler, (x, y), uc_w)

    # DSWD Staff → reviews + all end-user use cases
    for k in ("uc2", "uc3", "uc5", "uc6", "uc7", "uc8", "uc9", "uc10"):
        x, y, _ = use_cases[k]
        draw_assoc(ax, evaluator, (x, y), uc_w)

    plt.tight_layout()
    plt.savefig(USECASE_PNG, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {USECASE_PNG}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3.2 — System Flowchart with 4 decision diamonds
# ══════════════════════════════════════════════════════════════════════════════

def _proc_box(ax, x, y, w, h, text):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=PROCESS_FILL, edgecolor=PROCESS_EDGE, lw=1.3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", **FONT_NODE)


def _decision_diamond(ax, x, y, w, h, text):
    pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    poly = Polygon(pts, closed=True, facecolor=DECISION_FILL,
                   edgecolor=DECISION_EDGE, lw=1.3)
    ax.add_patch(poly)
    ax.text(x, y, text, ha="center", va="center", **FONT_SMALL)


def _terminal(ax, x, y, w, h, text):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.4",
                         facecolor=TERMINAL_FILL, edgecolor=TERMINAL_EDGE, lw=1.4)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=10, fontweight="bold")


def _arrow(ax, p0, p1, label=None, label_offset=(0.0, 0.0), label_color="#333333"):
    arr = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                          color=ARROW_COLOR, lw=1.1)
    ax.add_patch(arr)
    if label:
        mx = (p0[0] + p1[0]) / 2 + label_offset[0]
        my = (p0[1] + p1[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=8, color=label_color,
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", edgecolor="none", alpha=0.9))


def make_flowchart():
    fig, ax = plt.subplots(figsize=(11, 21))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 29)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(6, 28.2, "aiPHeed - How a Forecast Is Made",
            ha="center", va="center", **FONT_TITLE)

    # Geometry helpers
    cx = 6  # main vertical column
    pw, ph = 4.6, 0.9   # process box size
    dw, dh = 4.0, 1.4   # decision diamond size

    # Step 0 — START
    _terminal(ax, cx, 27.2, 1.8, 0.7, "START")

    # Step 1 — Refresh primary data
    _proc_box(ax, cx, 26.0, pw, ph,
              "1. Get latest government data\n(prices, jobs, climate, hunger surveys)")
    _arrow(ax, (cx, 26.85), (cx, 26.45))

    # Step 2 — Ingest news
    _proc_box(ax, cx, 24.5, pw, ph,
              "2. Collect news articles\n(English and Filipino)")
    _arrow(ax, (cx, 25.55), (cx, 24.95))

    # Step 3 — Geocode
    _proc_box(ax, cx, 23.0, pw, ph,
              "3. Find which province each\narticle is talking about")
    _arrow(ax, (cx, 24.05), (cx, 23.45))

    # Decision 1 — geocode match?
    _decision_diamond(ax, cx, 21.4, dw, dh,
                      "Is the article about a\nCALABARZON province?")
    _arrow(ax, (cx, 22.55), (cx, 22.10))

    # No branch from D1 → discard
    _proc_box(ax, 10.5, 21.4, 2.4, 0.8, "Skip article")
    _arrow(ax, (cx + dw / 2, 21.4), (10.5 - 1.2, 21.4), label="No",
           label_offset=(0.0, 0.2), label_color="#B00000")

    # Yes branch from D1 → bias weighting
    _proc_box(ax, cx, 19.8, pw, ph,
              "4. Group articles by province\nand quarter, then weight them")
    _arrow(ax, (cx, 20.7), (cx, 20.25), label="Yes", label_offset=(0.4, 0.0),
           label_color="#0E7E0E")

    # Decision 2 — article count >= 5?
    _decision_diamond(ax, cx, 18.2, dw, dh,
                      "Are there at least\n5 articles for this\nprovince-quarter?")
    _arrow(ax, (cx, 19.35), (cx, 18.90))

    # No branch from D2 → flag LIMITED_SIGNAL
    _proc_box(ax, 10.5, 18.2, 2.6, 0.8, "Mark as LIMITED SIGNAL\n(warn the user)")
    _arrow(ax, (cx + dw / 2, 18.2), (10.5 - 1.3, 18.2), label="No",
           label_offset=(0.0, 0.2), label_color="#B00000")

    # Yes branch from D2 (and the LIMITED_SIGNAL branch rejoins)
    _arrow(ax, (cx, 17.50), (cx, 16.95), label="Yes",
           label_offset=(0.7, 0.0), label_color="#0E7E0E")

    # LIMITED_SIGNAL branch comes back into the main flow
    _arrow(ax, (10.5, 17.80), (cx + 0.4, 16.95))

    # Step 5 — NLP scoring
    _proc_box(ax, cx, 16.5, pw, ph,
              "5. Read each article and score how\nrelated it is to food problems")

    # Decision 3 — relevance threshold
    _decision_diamond(ax, cx, 14.9, dw, dh,
                      "Is the article clearly\nabout food problems?")
    _arrow(ax, (cx, 16.05), (cx, 15.60))

    # No branch from D3 → exclude
    _proc_box(ax, 10.5, 14.9, 2.4, 0.8, "Skip article")
    _arrow(ax, (cx + dw / 2, 14.9), (10.5 - 1.2, 14.9), label="No",
           label_offset=(0.0, 0.2), label_color="#B00000")

    # Yes branch from D3 → FSSI compute
    _proc_box(ax, cx, 13.3, pw, ph,
              "6. Compute the Food Stress Sentiment Index (FSSI)\nand the 5 trigger categories per province")
    _arrow(ax, (cx, 14.20), (cx, 13.75), label="Yes",
           label_offset=(0.4, 0.0), label_color="#0E7E0E")

    # Step 7 — Feature matrix
    _proc_box(ax, cx, 11.8, pw, ph,
              "7. Combine all signals into one table\n(news + economy + climate)")
    _arrow(ax, (cx, 12.85), (cx, 12.25))

    # Step 8 — Run inference
    _proc_box(ax, cx, 10.3, pw, ph,
              "8. Load the trained LightGBM model\nand predict the next-quarter risk")
    _arrow(ax, (cx, 11.35), (cx, 10.75))

    # Step 9 — SHAP explainability (NEW)
    _proc_box(ax, cx, 8.8, pw, ph,
              "9. Compute SHAP values to identify\nwhich features most influenced each prediction")
    _arrow(ax, (cx, 9.85), (cx, 9.25))

    # Step 10 — Disaggregate
    _proc_box(ax, cx, 7.3, pw, ph,
              "10. Break down province risk to\n142 cities and towns\n(weighted by poverty + population)")
    _arrow(ax, (cx, 8.35), (cx, 7.75))

    # Decision 4 — alert threshold
    _decision_diamond(ax, cx, 5.7, dw, dh,
                      "Is the risk above\nthe warning level?")
    _arrow(ax, (cx, 6.85), (cx, 6.40))

    # No branch from D4 → no alert
    _proc_box(ax, 10.5, 5.7, 2.4, 0.8, "No warning sent")
    _arrow(ax, (cx + dw / 2, 5.7), (10.5 - 1.2, 5.7), label="No",
           label_offset=(0.0, 0.2), label_color="#B00000")

    # Yes branch from D4 → stage alert
    _proc_box(ax, cx, 4.1, pw, ph,
              "11. Send forecast and warning\nto admin for review")
    _arrow(ax, (cx, 5.00), (cx, 4.55), label="Yes",
           label_offset=(0.4, 0.0), label_color="#0E7E0E")

    # Also from "no alert" branch → stage forecast (rejoin)
    _arrow(ax, (10.5, 5.30), (cx + 0.4, 4.55))

    # END
    _terminal(ax, cx, 2.5, 1.8, 0.7, "END")
    _arrow(ax, (cx, 3.65), (cx, 2.85))

    # Legend
    lx, ly = 0.4, 27.0
    ax.add_patch(FancyBboxPatch((lx, ly), 1.6, 0.5,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=PROCESS_FILL, edgecolor=PROCESS_EDGE, lw=1.0))
    ax.text(lx + 0.8, ly + 0.25, "Step", ha="center", va="center", fontsize=8)
    ly -= 0.7
    pts = [(lx + 0.8, ly + 0.5), (lx + 1.5, ly + 0.25),
           (lx + 0.8, ly), (lx + 0.1, ly + 0.25)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=DECISION_FILL,
                          edgecolor=DECISION_EDGE, lw=1.0))
    ax.text(lx + 0.8, ly + 0.25, "Yes/No", ha="center", va="center", fontsize=8)
    ly -= 0.7
    ax.add_patch(FancyBboxPatch((lx, ly), 1.6, 0.5,
                                boxstyle="round,pad=0.02,rounding_size=0.25",
                                facecolor=TERMINAL_FILL, edgecolor=TERMINAL_EDGE, lw=1.0))
    ax.text(lx + 0.8, ly + 0.25, "Start / End", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FLOWCHART_PNG, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {FLOWCHART_PNG}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3.3 — State Diagram (forecast + alert)
# ══════════════════════════════════════════════════════════════════════════════

def _state_box(ax, x, y, w, h, label, fill=STATE_FILL, edge=STATE_EDGE):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.18",
                         facecolor=fill, edgecolor=edge, lw=1.5)
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=11, fontweight="bold")


def _start_dot(ax, x, y):
    ax.add_patch(plt.Circle((x, y), 0.10, color=ARROW_COLOR))


def _end_dot(ax, x, y):
    ax.add_patch(plt.Circle((x, y), 0.18, fill=False, lw=1.2, color=ARROW_COLOR))
    ax.add_patch(plt.Circle((x, y), 0.10, color=ARROW_COLOR))


def _trans(ax, p0, p1, label, curve=0.0, lpos=0.5, lof=(0, 0)):
    style = "-|>"
    if abs(curve) > 0.01:
        connectionstyle = f"arc3,rad={curve}"
    else:
        connectionstyle = "arc3,rad=0"
    arr = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                          color=ARROW_COLOR, lw=1.1,
                          connectionstyle=connectionstyle)
    ax.add_patch(arr)
    mx = p0[0] + (p1[0] - p0[0]) * lpos + lof[0]
    my = p0[1] + (p1[1] - p0[1]) * lpos + lof[1]
    ax.text(mx, my, label, ha="center", va="center",
            fontsize=8.5, style="italic",
            bbox=dict(boxstyle="round,pad=0.18",
                      facecolor="white", edgecolor="none", alpha=0.92))


def make_state_diagram():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(8, 9.5, "aiPHeed - What Happens to a Forecast and a Warning",
            ha="center", va="center", **FONT_TITLE)

    # ────────── Forecast lane (top) ──────────
    ax.text(8, 8.8, "Life of a Forecast", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#2E5C8A")

    sw, sh = 1.95, 0.9
    y_fore = 7.0
    _start_dot(ax, 1.2, y_fore)
    _state_box(ax, 3.0, y_fore, sw, sh, "Waiting")
    _state_box(ax, 6.5, y_fore, sw, sh, "For Review")
    _state_box(ax, 10.0, y_fore, sw, sh, "Published")
    _state_box(ax, 13.5, y_fore, sw, sh, "Archived",
               fill="#EAEAEA", edge="#666666")
    _end_dot(ax, 15.4, y_fore)

    # Transitions
    _trans(ax, (1.35, y_fore), (3.0 - sw / 2, y_fore), "forecast starts",
           lof=(0, 0.32))
    _trans(ax, (3.0 + sw / 2, y_fore), (6.5 - sw / 2, y_fore),
           "forecast ready", lof=(0, 0.32))
    _trans(ax, (6.5 + sw / 2, y_fore), (10.0 - sw / 2, y_fore),
           "admin approves", lof=(0, 0.32))
    _trans(ax, (10.0 + sw / 2, y_fore), (13.5 - sw / 2, y_fore),
           "next forecast comes in", lof=(0, 0.32))
    _trans(ax, (13.5 + sw / 2, y_fore), (15.4 - 0.2, y_fore), "")

    # Reject loop: For Review → Waiting (curved back)
    _trans(ax, (6.5 - sw / 2 + 0.05, y_fore - sh / 2),
                (3.0 + sw / 2 - 0.05, y_fore - sh / 2),
                "admin rejects (try again)",
                curve=-0.35, lpos=0.5, lof=(0, -0.55))

    # Decision note
    ax.text(8.25, y_fore - 1.35,
            "Admin must approve or reject before\nthe forecast becomes Published",
            ha="center", va="center", fontsize=9, style="italic",
            color="#555555")

    # ────────── Separator line ──────────
    ax.plot([0.5, 15.5], [4.7, 4.7], lw=0.8, color="#bbbbbb", linestyle="--")

    # ────────── Alert lane (bottom) ──────────
    ax.text(8, 4.2, "Life of a Warning", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#2E5C8A")

    y_alert = 2.5
    _start_dot(ax, 1.2, y_alert)
    _state_box(ax, 3.0, y_alert, sw, sh, "For Review")
    _state_box(ax, 8.5, y_alert + 0.9, sw, sh, "Approved",
               fill="#D6E8D6", edge="#3A7A3A")
    _state_box(ax, 8.5, y_alert - 0.9, sw, sh, "Rejected",
               fill="#F5D6D6", edge="#A04040")
    _end_dot(ax, 11.0, y_alert + 0.9)
    _end_dot(ax, 11.0, y_alert - 0.9)

    _trans(ax, (1.35, y_alert), (3.0 - sw / 2, y_alert),
           "risk too high", lof=(0, 0.32))
    # Branch to Approved
    _trans(ax, (3.0 + sw / 2, y_alert), (8.5 - sw / 2, y_alert + 0.9),
           "admin approves", curve=0.18, lpos=0.55, lof=(0, 0.35))
    # Branch to Rejected
    _trans(ax, (3.0 + sw / 2, y_alert), (8.5 - sw / 2, y_alert - 0.9),
           "admin rejects", curve=-0.18, lpos=0.55, lof=(0, -0.35))

    _trans(ax, (8.5 + sw / 2, y_alert + 0.9), (11.0 - 0.2, y_alert + 0.9),
           "shown on dashboard", lof=(0, 0.32))
    _trans(ax, (8.5 + sw / 2, y_alert - 0.9), (11.0 - 0.2, y_alert - 0.9),
           "kept in records only", lof=(0, 0.32))

    # Decision note
    ax.text(13.5, y_alert,
            "Admin must approve\nor reject the warning\nbefore it is shown",
            ha="center", va="center", fontsize=9, style="italic",
            color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8E0",
                      edgecolor="#C49000", lw=0.8))

    plt.tight_layout()
    plt.savefig(STATE_PNG, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {STATE_PNG}")


# ══════════════════════════════════════════════════════════════════════════════
# Replace the 3 inline shapes in the docx with the new PNGs
# ══════════════════════════════════════════════════════════════════════════════

def replace_images_in_docx() -> None:
    from docx import Document
    from docx.shared import Inches
    from docx.oxml.ns import qn
    import shutil

    SRC = Path(r"C:\Users\admin\Downloads\section_3_5_with_diagrams_v3.docx")
    DST = Path(r"C:\Users\admin\Downloads\section_3_5_with_diagrams_v4.docx")

    if not SRC.exists():
        print(f"[error] source docx not found: {SRC}")
        return

    shutil.copy(SRC, DST)
    doc = Document(DST)

    # The three inline shapes appear in document order. The image data lives
    # in word/media/. Easiest reliable replacement: walk paragraphs in order,
    # find each one containing an inline image, drop the image run, and
    # insert a fresh image at its position.
    new_pngs = [USECASE_PNG, FLOWCHART_PNG, STATE_PNG]
    image_idx = 0

    for para in doc.paragraphs:
        # Find any image-bearing run in this paragraph
        has_image = False
        for run in para.runs:
            if run._element.findall(".//" + qn("w:drawing")):
                has_image = True
                break
        if not has_image:
            continue

        if image_idx >= len(new_pngs):
            break

        # Clear existing runs entirely (drops the old image)
        for run in list(para.runs):
            run._element.getparent().remove(run._element)

        # Add a new run with the replacement image, full width 6.5 inches
        run = para.add_run()
        run.add_picture(str(new_pngs[image_idx]), width=Inches(6.3))
        print(f"[replaced] image #{image_idx + 1} -> {new_pngs[image_idx].name}")
        image_idx += 1

    doc.save(DST)
    print(f"\n[saved] {DST}".replace("→", "->"))


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    make_usecase_diagram()
    make_flowchart()
    make_state_diagram()
    replace_images_in_docx()


if __name__ == "__main__":
    main()
