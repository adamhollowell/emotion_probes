#!/usr/bin/env python3
"""
Build the extended (top-20 per emotion) Figure-1 render data from the durable
v2 held-out artifacts. Output schema matches fig1_render_data_v2.json, so the
v11 / curated renderers (and scripts/render_fig1_draft.py) consume it directly.

Run from the project root:
    python3 scripts/build_fig1_ext20.py

Inputs (all under data/heldout/maxact/ unless noted):
  maxactivating_v2.json      top-20 windows/emotion (171 emotions): doc_id, peak_pos,
                             peak_token, source. topk=20, layer=31.
  pertoken/<doc_id>.npz      per-token arrays for every top doc:
                               tokens[N]  (object array of token strings, includes leading spaces)
                               proj[N,171] (L31 projection of each token onto the 171 emotion vectors)
  fig1_render_data_v2.json   panel-level p90_floor + top1_value (reused UNCHANGED; these define
                             the shading normalization and are the locked P6 shading anchors)
  data/vectors/emotion_vectors_171.pt  emotions list -> gives the proj column order

Output:
  data/heldout/maxact/fig1_render_data_ext20.json

Window convention (per row):
  tokens = full_doc_tokens[peak_pos - LEAD : peak_pos + SPAN]
  values = proj[:, emotion_col] over the same slice
  peak_index_in_row = peak_pos - start
  LEAD = how many tokens of calm lead-in precede the peak. LARGER LEAD = airier panels
         (more sub-threshold words shown) BUT shifts the window left and re-exposes more
         surrounding text -> re-vet printability after any change.
  LEAD_OVERRIDE = per-(emotion, rank) lead-in trim, e.g. to drop a suggestive prefix while
         keeping the activation peak in view.

The emotion->column map is verified against maxactivating_v2 (proj[peak_pos, col] must equal
the stored rank-1 value); the script aborts if the mapping is wrong.
"""
# --- basis gate: refuse to run unless the vectors match the expected sha256.
# Identify the vectors by their hash, not the filename `emotion_vectors_171.pt`. ---
import hashlib as _hlg, pathlib as _plg
_VECG = _plg.Path("data/vectors/emotion_vectors_171.pt")
_EXPG = "6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b"
if _VECG.exists():
    _hg = _hlg.sha256(_VECG.read_bytes()).hexdigest()
    assert _hg == _EXPG, (
        "%s basis mismatch: %s != expected canonical %s -- refusing non-canonical basis"
        % (_VECG, _hg[:12], _EXPG[:12]))
# --- end basis gate ---

import sys, os, json
import numpy as np

# ---- tunables -------------------------------------------------------------
LEAD = 13
SPAN = 60
LEAD_OVERRIDE = {("proud", 5): 7}   # trims a suggestive numeric prefix; keeps the "pride" peak
# ---------------------------------------------------------------------------

EMOS = ['afraid', 'inspired', 'angry', 'loving', 'calm', 'nervous',
        'desperate', 'proud', 'guilty', 'sad', 'happy', 'surprised']

ROOT   = os.getcwd()
MAXACT = os.path.join(ROOT, 'data/heldout/maxact/maxactivating_v2.json')
RD_V2  = os.path.join(ROOT, 'data/heldout/maxact/fig1_render_data_v2.json')
PT_DIR = os.path.join(ROOT, 'data/heldout/maxact/pertoken')
VEC    = os.path.join(ROOT, 'data/vectors/emotion_vectors_171.pt')
OUT    = os.path.join(ROOT, 'data/heldout/maxact/fig1_render_data_ext20.json')


def main():
    mv = json.load(open(MAXACT))['by_emotion']
    rd = json.load(open(RD_V2))['panels']

    cache = {}
    def toks_proj(did):
        if did not in cache:
            z = np.load(os.path.join(PT_DIR, did + '.npz'), allow_pickle=True)
            cache[did] = (z['tokens'], z['proj'])
        return cache[did]

    # Self-calibrate emotion -> proj column with NO external dependency:
    # for each emotion, the correct column is the one whose projection at the
    # rank-1 peak equals the stored rank-1 value. (proj has 171 columns in the
    # 171-vector order; this recovers the mapping straight from the data.)
    col = {}
    for e in EMOS:
        r = mv[e][0]
        _, proj = toks_proj(r['doc_id'])
        c = int(np.argmin(np.abs(proj[int(r['peak_pos'])].astype(np.float64) - float(r['value']))))
        if abs(float(proj[int(r['peak_pos']), c]) - float(r['value'])) > max(2.0, abs(float(r['value'])) * 0.01):
            sys.exit(f"COLUMN CALIBRATION FAIL for {e}")
        col[e] = c

    out = {
        'figure': 'Figure 1 DRAFT (top-20 held-out, printable-selected)',
        'basis': 'L31 cleaned vectors; v2 durable held-out corpus (n=47,274)',
        'shading_rule': ('per-panel: token shaded iff value>p90_floor; '
                         'intensity=(value-p90_floor)/(top1_value-p90_floor), clamped; '
                         'p90_floor/top1_value copied from fig1_render_data_v2.json'),
        'window': {'LEAD': LEAD, 'SPAN': SPAN,
                   'LEAD_OVERRIDE': {f'{k[0]}#{k[1]}': v for k, v in LEAD_OVERRIDE.items()}},
        'panels': {},
    }

    for e in EMOS:
        floor = float(rd[e]['p90_floor'])
        top   = float(rd[e]['top1_value'])
        rows = []
        for i, r in enumerate(mv[e]):
            rank = i + 1
            did  = r['doc_id']
            pp   = int(r['peak_pos'])
            toks, proj = toks_proj(did)
            # verify column mapping on the rank-1 window
            if rank == 1:
                got = float(proj[pp, col[e]])
                if abs(got - float(r['value'])) > max(2.0, abs(float(r['value'])) * 0.01):
                    sys.exit(f"COLUMN MAP FAIL {e}: proj={got} vs stored={r['value']}")
            pj = proj[:, col[e]]
            lead = LEAD_OVERRIDE.get((e, rank), LEAD)
            a = max(0, pp - lead)
            b = min(len(toks), pp + SPAN)
            rows.append({
                'rank': rank,
                'doc_id': did,
                'source': r['source'],
                'peak_index_in_row': pp - a,
                'clipped_left': a > 0,
                'clipped_right': b < len(toks),
                'tokens': [str(t) for t in toks[a:b]],
                'values': [float(v) for v in pj[a:b]],
            })
        out['panels'][e] = {'emotion': e, 'p90_floor': floor, 'top1_value': top, 'rows': rows}

    json.dump(out, open(OUT, 'w'))
    print('wrote', OUT, '-', sum(len(v['rows']) for v in out['panels'].values()), 'rows')


if __name__ == '__main__':
    main()
