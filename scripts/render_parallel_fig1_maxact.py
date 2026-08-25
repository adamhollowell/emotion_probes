#!/usr/bin/env python3
"""
Figure 1 renderer.

Inputs (resolved relative to the project root):
  data/heldout/maxact_v3/fig1_render_data.json      per-panel tokens, per-token
      values, and the per-panel p90_floor + top1_value shading anchors
      (produced by build_fig1_render_data.py from the canonical sweep).
  data/heldout/maxact_v3/fig1_window_selection.json the displayed ranks per panel;
      the renderer draws exactly those rows, in rank order. Nothing is hard-coded.

Outputs:
  docs/e1_figures/E1_Figure_1_maxact.png
  docs/e1_figures/E1_Figure_1_maxact.pdf
  docs/e1_figures/E1_Figure_1_maxact_selection_audit.json

Design goals:
  - faithful parallel to Anthropic's first emotion-vector figure
  - 12 panels, Anthropic-style column ordering
  - 7 visible snippet rows per panel
  - frameless, dense excerpt blocks
  - exact token concatenation, no inserted spaces
  - per-panel activation floor/cap shading
  - true solid multi-stop heatmap colors, not alpha-faded single orange
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / "data" / "heldout" / "maxact_v3" / "fig1_render_data.json"
SELECTION_PATH = PROJECT_ROOT / "data" / "heldout" / "maxact_v3" / "fig1_window_selection.json"

OUT_DIR = PROJECT_ROOT / "docs" / "e1_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / "E1_Figure_1_maxact.png"
OUT_PDF = OUT_DIR / "E1_Figure_1_maxact.pdf"
OUT_SELECTION = OUT_DIR / "E1_Figure_1_maxact_selection_audit.json"


# ---------------------------------------------------------------------
# Panel order
# ---------------------------------------------------------------------
# Ordered row-wise so the visual columns read:
# Left column:  Afraid, Angry, Calm, Desperate, Guilty, Happy
# Right column: Inspired, Loving, Nervous, Proud, Sad, Surprised

PANEL_ORDER = [
    "afraid", "inspired",
    "angry", "loving",
    "calm", "nervous",
    "desperate", "proud",
    "guilty", "sad",
    "happy", "surprised",
]


# ---------------------------------------------------------------------
# Row selection
# ---------------------------------------------------------------------
# The displayed ranks per panel come entirely from the window-selection file;
# nothing is hard-coded here. Rows are drawn in ascending rank order.

def load_row_selection() -> Dict[str, List[int]]:
    with open(SELECTION_PATH, "r", encoding="utf-8") as f:
        sel = json.load(f)
    return {
        p["emotion"].strip().lower(): [int(r) for r in p["selected_ranks"]]
        for p in sel["panels"]
    }


# ---------------------------------------------------------------------
# Visual parameters
# ---------------------------------------------------------------------

DPI = 220

CANVAS_W = 1832
CANVAS_H = 1940

MARGIN_L = 70
MARGIN_R = 70
MARGIN_T = 45
MARGIN_B = 45

COL_GAP = 92
ROW_GAP = 22

# Actual excerpt window width. This is now also the layout slot width,
# so left and right columns use exactly the same visible row length.
BOX_W = 800
PANEL_W = BOX_W
PANEL_H = 288

HEADER_H = 31
INNER_PAD_X = -18
INNER_PAD_Y = 6

ROW_H = 40
ROW_GAP_INNER = 0

BG = (255, 255, 255)
PANEL_BG = (255, 255, 254)
PANEL_BORDER = (226, 226, 222)
TEXT = (18, 18, 18)
TITLE = (20, 20, 20)

BOX_RADIUS = 5


# ---------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------

def load_font(size: int, bold: bool = False):
    """
    Locked Figure 1 render requires Arial/Helvetica typography.
    Fail loudly rather than falling back to DejaVu Sans.
    """
    from pathlib import Path
    from PIL import ImageFont

    candidates_regular = [
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ]
    candidates_bold = [
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ]

    candidates = candidates_bold if bold else candidates_regular
    for fp in candidates:
        path = Path(fp)
        if path.exists():
            print(f"Using font: {path}")
            return ImageFont.truetype(str(path), size=size)

    raise RuntimeError(
        "Required Arial/Helvetica font not found. "
        "Install or enable Arial/Helvetica before rendering Figure 1."
    )


FONT_HEADER = load_font(30, bold=False)
FONT_ROW = load_font(23, bold=False)

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert '#rrggbb' to (r, g, b)."""
    h = hex_color.strip().lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected 6-digit hex color, got: {hex_color}")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


HEATMAP_STOPS = [
    (0.00, "#fdd8b3"),  # low activation, high-brightness peach/gold
    (0.17, "#fdd1a5"),
    (0.34, "#fdc38d"),
    (0.50, "#fd9b51"),  # mid activation, vibrant orange
    (0.66, "#f57623"),
    (0.80, "#f2701d"),
    (0.92, "#ed6614"),
    (1.00, "#e15509"),  # high activation, saturated burnt red-orange
]


def interpolate_rgb(c0: Tuple[int, int, int], c1: Tuple[int, int, int], u: float) -> Tuple[int, int, int]:
    """Linear interpolation between two RGB colors."""
    u = max(0.0, min(1.0, float(u)))
    return tuple(
        int(round(c0[i] + (c1[i] - c0[i]) * u))
        for i in range(3)
    )


def heatmap_rgba_for_value(value: float, floor: float, cap: float) -> Tuple[int, int, int, int]:
    """
    Locked data rule:
      - highlight iff value > floor
      - normalized strength = (value - floor) / (cap - floor), clamped to [0, 1]
      - floor/cap are per panel

    Visual rule:
      - use a true sequential multi-stop color ramp
      - use solid opacity, alpha = 255
      - apply a power curve so mid-range activations remain vibrant
    """
    if cap <= floor:
        return (255, 255, 255, 0)

    raw = (float(value) - float(floor)) / (float(cap) - float(floor))
    raw = max(0.0, min(1.0, raw))

    if raw <= 0:
        return (255, 255, 255, 0)

    # Visual contrast correction only. Does not change thresholding or ordering.
    t = raw ** 1.30

    for i in range(len(HEATMAP_STOPS) - 1):
        t0, h0 = HEATMAP_STOPS[i]
        t1, h1 = HEATMAP_STOPS[i + 1]

        if t <= t1:
            local_u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            rgb = interpolate_rgb(hex_to_rgb(h0), hex_to_rgb(h1), local_u)
            return (*rgb, 195)

    return (*hex_to_rgb(HEATMAP_STOPS[-1][1]), 195)


# ---------------------------------------------------------------------
# Text / drawing helpers
# ---------------------------------------------------------------------

def title_case_emotion(e: str) -> str:
    return e.strip().capitalize()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> float:
    if text == "":
        return 0.0
    try:
        return draw.textlength(text, font=font)
    except Exception:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]


def normalize_newlines(token: str) -> str:
    # Render embedded newlines as spaces (matches Anthropic / v12 curated).
    return str(token).replace("\n", " ").replace("\r", " ")


def draw_token_highlight(
    panel: Image.Image,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    fill_rgba: Tuple[int, int, int, int],
) -> None:
    if fill_rgba[3] <= 0:
        return

    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    od.rounded_rectangle(
        [int(x0), int(y0), int(x1), int(y1)],
        radius=BOX_RADIUS,
        fill=fill_rgba,
    )

    panel.alpha_composite(overlay)


def render_row(
    panel: Image.Image,
    y: int,
    tokens: List[str],
    values: List[float],
    p90_floor: float,
    top1_value: float,
    clipped_left: bool,
    clipped_right: bool,
) -> None:
    """
    Render one row into a clipped row image of width BOX_W.

    This makes both columns use the same visible excerpt-window width,
    independent of the larger layout slot. Text is not altered; it is simply
    drawn into a fixed-width viewport and clipped at the right edge.
    """
    row_img = Image.new("RGBA", (BOX_W, ROW_H), (255, 255, 255, 0))
    draw = ImageDraw.Draw(row_img)

    # Faint Anthropic-like row guide line, drawn before highlights/text.
    # Skip the sixth/final row so it does not double with the panel bottom border.
    if y < INNER_PAD_Y + 5 * ROW_H:
        draw.line(
            [(0, ROW_H - 3), (BOX_W, ROW_H - 3)],
            fill=(214, 214, 214, 153),
            width=1,
        )

    x = INNER_PAD_X

    # Do not render leading ellipses. clipped_left remains recorded in audit.
    if clipped_left:
        pass

    for tok, val in zip(tokens, values):
        t = normalize_newlines(tok)
        if t == "":
            continue

        w = text_width(draw, t, FONT_ROW)
        fill_rgba = heatmap_rgba_for_value(val, p90_floor, top1_value)

        if fill_rgba[3] > 0:
            # Bubble highlight: shade only the visible word glyphs, not the
            # leading/trailing whitespace, so adjacent shaded words render as
            # separate rounded pills (Anthropic look) instead of a solid bar.
            stripped = t.lstrip(" ")
            lead = t[: len(t) - len(stripped)]
            lead_w = text_width(draw, lead, FONT_ROW) if lead else 0.0
            word = stripped.rstrip(" ")
            word_w = text_width(draw, word, FONT_ROW)
            if word_w > 0:
                # v12-style activation geometry:
                # tighter rectangular highlights, modest corner radius,
                # less vertical/horizontal padding than the bubble renderer.
                bx0 = x + lead_w - 2.0
                bx1 = x + lead_w + word_w + 2.0
                overlay = Image.new("RGBA", row_img.size, (0, 0, 0, 0))
                od = ImageDraw.Draw(overlay)
                od.rounded_rectangle(
                    [int(bx0), 4, int(bx1), 29],
                    radius=3,
                    fill=fill_rgba,
                )
                row_img.alpha_composite(overlay)

        draw.text((x, 5), t, font=FONT_ROW, fill=TEXT)
        x += w

    # Do not render trailing ellipses. Fixed row viewport handles clipping.
    if clipped_right:
        pass

    panel.alpha_composite(row_img, dest=(0, y))


def render_panel(panel_data: Dict[str, Any], selected_ranks: List[int]) -> Tuple[Image.Image, Dict[str, Any]]:
    emotion = panel_data["emotion"]
    p90_floor = float(panel_data["p90_floor"])
    top1_value = float(panel_data["top1_value"])

    # Panel image includes both the external title and the boxed excerpt area.
    # The border should surround only the excerpt rows, not the emotion header.
    panel = Image.new("RGBA", (PANEL_W, PANEL_H), BG + (255,))
    draw = ImageDraw.Draw(panel)

    # Emotion title outside the box, matching Anthropic's visual structure.
    draw.text(
        (0, 0),
        title_case_emotion(emotion),
        font=FONT_HEADER,
        fill=TITLE,
    )

    rows_by_rank = {int(r["rank"]): r for r in panel_data["rows"]}
    selected_rows = []

    # Box begins below the title and tightly encloses the visible rows.
    box_top = HEADER_H + 14
    row_start_y = box_top + 4
    box_bottom = row_start_y + len(selected_ranks) * ROW_H - 2

    # Light excerpt box around the text only.
    draw.rectangle(
        [0, box_top, BOX_W - 1, box_bottom],
        outline=PANEL_BORDER,
        width=1,
        fill=PANEL_BG,
    )

    y = row_start_y

    for rank in selected_ranks:
        row = rows_by_rank[rank]
        selected_rows.append({
            "rank": rank,
            "doc_id": row.get("doc_id"),
            "source": row.get("source"),
            "contains_scrub_token": bool(row.get("contains_scrub_token", False)),
            "clipped_left": bool(row.get("clipped_left", False)),
            "clipped_right": bool(row.get("clipped_right", False)),
        })

        render_row(
            panel=panel,
            y=y,
            tokens=row["tokens"],
            values=row["values"],
            p90_floor=p90_floor,
            top1_value=top1_value,
            clipped_left=bool(row.get("clipped_left", False)),
            clipped_right=bool(row.get("clipped_right", False)),
        )
        y += ROW_H + ROW_GAP_INNER

    selection_info = {
        "emotion": emotion,
        "selected_ranks": selected_ranks,
        "selected_rows": selected_rows,
        "title_outside_box": True,
        "box_encloses_visible_rows_only": True,
    }

    return panel, selection_info


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing input JSON: {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "panels" in data:
        panels_raw = data["panels"]
    else:
        panels_raw = data

    # Support both possible extractor shapes:
    # 1. {"panels": [{"emotion": "afraid", ...}, ...]}
    # 2. {"panels": {"afraid": {...}, "angry": {...}, ...}}
    if isinstance(panels_raw, dict):
        panels_by_emotion = {}
        for emotion, panel in panels_raw.items():
            if not isinstance(panel, dict):
                raise TypeError(f"Panel for {emotion!r} is not a dict: {type(panel).__name__}")
            panel = dict(panel)
            panel.setdefault("emotion", emotion)
            panels_by_emotion[emotion.strip().lower()] = panel
    elif isinstance(panels_raw, list):
        panels_by_emotion = {
            p["emotion"].strip().lower(): p
            for p in panels_raw
        }
    else:
        raise TypeError(f"Unsupported panels container: {type(panels_raw).__name__}")

    missing = [e for e in PANEL_ORDER if e not in panels_by_emotion]
    if missing:
        raise ValueError(f"Missing panels in JSON: {missing}")

    row_selection = load_row_selection()
    missing_sel = [e for e in PANEL_ORDER if e not in row_selection]
    if missing_sel:
        raise ValueError(f"Missing panels in selection: {missing_sel}")

    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)

    selection_audit = {
        "figure": "Figure 1",
        "input": str(DATA_PATH),
        "selection": str(SELECTION_PATH),
        "output_png": str(OUT_PNG),
        "output_pdf": str(OUT_PDF),
        "layout": "2 columns x 6 rows, Anthropic-style column reading order, equal fixed-width excerpt boxes",
        "row_selection_rule": "displayed ranks per panel taken from fig1_window_selection.json, drawn in ascending rank order; clipped_left/right retained in audit but ellipses are not rendered",
        "highlight_rule": (
            "token highlighted iff value > panel p90_floor; normalized intensity = "
            "(value-p90_floor)/(top1_value-p90_floor), clamped to [0,1]; visual color "
            "uses solid multi-stop heatmap ramp after t=t**1.30"
        ),
        "panels": [],
    }

    for i, emotion in enumerate(PANEL_ORDER):
        row = i // 2
        col = i % 2

        x = MARGIN_L + col * (PANEL_W + COL_GAP)
        y = MARGIN_T + row * (PANEL_H + ROW_GAP)

        panel_img, panel_selection = render_panel(
            panels_by_emotion[emotion],
            row_selection[emotion],
        )

        canvas.paste(panel_img.convert("RGB"), (x, y))
        selection_audit["panels"].append(panel_selection)

    canvas.save(OUT_PNG, "PNG", dpi=(DPI, DPI))
    canvas.save(OUT_PDF, "PDF", resolution=DPI)

    with open(OUT_SELECTION, "w", encoding="utf-8") as f:
        json.dump(selection_audit, f, indent=2)

    print("Figure 1 render complete.")
    print(f"Input:  {DATA_PATH}")
    print(f"PNG:    {OUT_PNG}")
    print(f"PDF:    {OUT_PDF}")
    print(f"Audit:  {OUT_SELECTION}")
    print("")
    print("Panels:", len(PANEL_ORDER))
    print("Rows per panel:", {k: len(v) for k, v in row_selection.items()})


if __name__ == "__main__":
    main()
