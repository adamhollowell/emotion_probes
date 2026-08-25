#!/usr/bin/env python3
"""
Interactive label editor for Figure 3 (the UMAP cluster scatter).

Run from the PROJECT ROOT in your .venv (its matplotlib backend is macosx, i.e.
interactive):

    python3 repo/github_repo/scripts/render_parallel_fig6_interactive.py

- Reads the frozen coords + cluster assignment from
  geometry_out/fig3_coords_AUTH.csv (the ONE authoritative coordinate file).
  No UMAP / k-means is recomputed here, ever. The legend/key, palette, title, and
  axis limits are fixed in fig6_common.py and are NOT edited here — only the
  emotion word-labels and their leader lines move.
- Drag any emotion word to reposition it; its leader line follows.
- Press 'S' to save. Saving writes BOTH:
      geometry_out/fig3_label_layout_AUTH.json  <- the reproducible label
                                               layout (ships), stamped with the
                                               coords sha it was placed against.
      geometry_out/fig3_label_preview.png  <- rendered via the SAME code path
                                               the producer uses, so this preview is
                                               byte-identical to the final PNG.
  If a layout file already exists it is loaded as the starting point, so you can
  resume and refine across sessions.
- When happy, hand the layout JSON back; the producer (render_parallel_fig6_umap.py)
  regenerates the canonical PNG from it. Do NOT copy into locked_figures_and_tables/.

Headless self-test / initial seed (no window; auto-declutters and writes the layout):
    python3 repo/github_repo/scripts/render_parallel_fig6_interactive.py --export
"""
import sys

import matplotlib

EXPORT = "--export" in sys.argv
if EXPORT:
    matplotlib.use("Agg")  # no window for the headless seed/self-test

import matplotlib.pyplot as plt
from adjustText import adjust_text

import fig6_common as fc

PREVIEW_PNG = "geometry_out/fig3_label_preview.png"

# On-screen figure scale. At full size the figure is taller than the screen and the
# macOS backend gives no scrollbar, hiding the bottom cluster. This shrinks the whole
# figure uniformly (figsize + fonts + markers) so it fits, at normal dpi so picking
# still works. Placement geometry and the saved PNG are unaffected (producer renders
# at scale 1.0). Lower/raise with --scale F if it still doesn't fit or is too small.
EDITOR_SCALE = 0.58


def parse_scale(default=EDITOR_SCALE):
    if "--scale" in sys.argv:
        return float(sys.argv[sys.argv.index("--scale") + 1])
    return default


def initial_texts(ax, df, point_pos, fontsize):
    """Seed the draggable labels. Resume from an existing layout when present;
    otherwise start at the point positions and auto-declutter once (adjustText).
    The auto-declutter is a convenience starting point only — the reproducible
    truth is whatever gets captured to the layout file on save."""
    layout = fc.load_layout()
    texts = []
    if layout is not None:
        for w, L in layout["labels"].items():
            if w not in point_pos:
                continue
            t = ax.text(L["x"], L["y"], w, fontsize=fontsize,
                        ha=L.get("ha", "center"), va=L.get("va", "center"), color="black")
            texts.append(t)
        print("[resumed] starting from %s (%d labels)" % (fc.LAYOUT_JSON, len(texts)))
    else:
        for w in fc.DENSE_WORDS:
            if w not in point_pos:
                continue
            x0, y0 = point_pos[w]
            texts.append(ax.text(x0, y0, w, fontsize=fontsize, ha="center", va="center", color="black"))
        adjust_text(texts, x=df.x.values, y=df.y.values, expand=(1.4, 1.6), force_text=(0.4, 0.6))
        for t in texts:
            cx, _ = t.get_position()
            ox, _ = point_pos[t.get_text()]
            t.set_ha("left" if cx > ox else "right")
        print("[fresh] seeded %d labels with an adjustText declutter" % len(texts))
    return texts


def attach_leaders(ax, texts, point_pos):
    arrows = {}
    for t in texts:
        w = t.get_text()
        ox, oy = point_pos[w]
        cx, cy = t.get_position()
        arrows[w] = ax.annotate("", xy=(ox, oy), xytext=(cx, cy), arrowprops=dict(**fc.LEADER_KW))
    return arrows


def save_all(df, texts, point_pos):
    layout = fc.layout_from_artists(df, texts, point_pos)
    fc.save_layout(layout)
    fc.render_final(df, layout, PREVIEW_PNG)   # same path the producer uses
    print("[saved] %d labels -> %s  +  %s" % (len(layout["labels"]), fc.LAYOUT_JSON, PREVIEW_PNG))
    return layout


def main():
    df = fc.load_coords()
    fc.check_membership(df)
    scale = 1.0 if EXPORT else parse_scale()
    fig, ax, DENS, point_pos = fc.build_base(df, scale=scale)
    # Disable matplotlib's built-in 's'/'ctrl+s' save-dialog so pressing S runs ONLY
    # our layout-capturing save (the built-in dialog was silently bypassing capture).
    matplotlib.rcParams["keymap.save"] = []
    if not EXPORT:
        w, h = fig.get_size_inches()
        print("[window] on-screen figure scale=%g (~%dx%d pt); pass --scale F to "
              "resize. Placement and the saved PNG are unaffected."
              % (scale, round(w * 100), round(h * 100)))
    fontsize = DENS["point_label_fontsize"]

    texts = initial_texts(ax, df, point_pos, fontsize)
    arrows = attach_leaders(ax, texts, point_pos)

    if EXPORT:
        save_all(df, texts, point_pos)
        print("[export] headless seed complete; open without --export to hand-tune.")
        return

    class DragLabels:
        def __init__(self):
            self.sel = None
            for t in texts:
                t.set_picker(True)
            fig.canvas.mpl_connect("pick_event", self.press)
            fig.canvas.mpl_connect("motion_notify_event", self.drag)
            fig.canvas.mpl_connect("button_release_event", self.release)
            fig.canvas.mpl_connect("key_press_event", self.key)

        def press(self, e):
            if e.artist in texts:
                self.sel = e.artist

        def drag(self, e):
            if self.sel is not None and e.xdata is not None and e.ydata is not None:
                w = self.sel.get_text()
                ox, _ = point_pos[w]
                self.sel.set_ha("left" if e.xdata > ox else "right")
                self.sel.set_position((e.xdata, e.ydata))
                arrows[w].set_position((e.xdata, e.ydata))
                fig.canvas.draw_idle()

        def release(self, e):
            self.sel = None

        def key(self, e):
            if e.key and e.key.lower() == "s":
                layout = save_all(df, texts, point_pos)
                try:
                    fig.canvas.manager.set_window_title(
                        "Figure 3  —  SAVED %d labels to %s" % (len(layout["labels"]), fc.LAYOUT_JSON))
                except Exception:
                    pass

    print("=" * 66)
    print("DRAG each emotion word to reposition it; the leader line follows.")
    print("Press 'S' to save -> fig3_label_layout_AUTH.json + fig3_label_preview.png.")
    print("The preview is rendered by the producer's code path (WYSIWYG-exact).")
    print("=" * 66)
    _editor = DragLabels()  # noqa: F841 (keeps handlers alive)
    plt.show()


if __name__ == "__main__":
    main()
