#!/usr/bin/env python3
"""GGE1-G4 IT-colon producer (pod B) — 27b-it final-token residuals on E1's
frozen fig2 12-scenario set, per IT_COLON_ANALYSIS_DESIGN.md (Amendment 1) and
coordination_100T/E1_baseside_protocol_for_GGE1.md.

Producer ONLY: extraction + gates; saves RESIDUALS (never projections — the
P7.1 lesson). All analysis (unitnorm projections, base pairing vs E1's
persisted scenario_acts.pt, the P-colon grade) is post-session Grid-side.

Three arms, all captured at resid_post, ALL 46 layers, f32, one position each:
  bare      PRIMARY — bare prompt, BOS-prepended, FINAL token (the fig2-gate
            winner; matched to E1's base side by construction)
  template  SENSITIVITY — Gemma chat template, final token of the tokenized
            generation-ready prompt (disclosed regime shift; never pooled)
  colon     TERTIARY — "Human: {p}\\n\\nAssistant:" raw format, final token
            (the colon); IT-side regime probe only, excluded from P-colon

Gates: fingerprint (VERIFIED pin, google/gemma-2-27b-it); prompt-set pairing
(stimulus file present, 12 prompts, base-side activations checked by hash);
per-arm BOS/final-token asserts. Pilot = 1 prompt -> SHOW ADAM.

RUN ORDER (pod B, at the 27b-it load):
    python gemma_grid/colon_it/run_it_colon.py --pilot
    python gemma_grid/colon_it/run_it_colon.py

Outputs:
    gemma_grid/colon_it/colon_analysis_it.pt      (~4 MB w/ all arms)
    gemma_grid/colon_it/run_it_colon_manifest.json     (SR.5)
(Filename keeps the historical "colon" label; the primary arm is bare-final —
recorded in provenance per Amendment 1.)
"""
from __future__ import annotations
import argparse, hashlib, json, platform, re, sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

MODEL = "google/gemma-2-27b-it"
N_LAYERS, D_MODEL = 46, 4608
VERIFIED_PINS = REPO / "gemma_grid/handoff/hf_model_revisions_VERIFIED.json"
STIMULUS = REPO / "geometry_out/scenario_matrix_layer31.json"   # the replication study's released copy: the same 12 prompts and emotion order as the copy the run read
SCENARIO_ACTS = REPO / "data/vectors/scenario_acts.pt"           # the replication study's base-side scenario activations
SCENARIO_ACTS_SHA = "94244b237e36b8d5f2d3b28eabf17dd46e754c11e3c928e5cebfe32169345e12"
OUT = REPO / "gemma_grid/colon_it/colon_analysis_it.pt"

MANIFEST: dict[str, str] = {}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    MANIFEST[str(p.relative_to(REPO))] = h.hexdigest()
    return h.hexdigest()


def fingerprint_gate() -> str:
    from huggingface_hub import HfApi
    sha256(VERIFIED_PINS)
    expected = json.loads(VERIFIED_PINS.read_text())[MODEL]["current_main_sha"]
    resolved = HfApi().model_info(MODEL).sha
    print(f"[fingerprint] resolved={resolved}\n[fingerprint] expected={expected}")
    if resolved != expected:
        print("[fingerprint] MISMATCH — STOP. Report to Adam.")
        sys.exit(2)
    print("[fingerprint] PASS — pinning explicitly.")
    return resolved


def load_prompts() -> tuple[list, dict]:
    stim_sha = sha256(STIMULUS)
    acts_sha = sha256(SCENARIO_ACTS)
    prompts = json.loads(STIMULUS.read_text())["prompts"]
    assert len(prompts) == 12, f"expected 12 scenarios, got {len(prompts)} — STOP"
    assert acts_sha == SCENARIO_ACTS_SHA, \
        "data/vectors/scenario_acts.pt is not the banked base-side activations — pairing unverifiable, STOP"
    pairing = {"stimulus_matrix_sha256": stim_sha,
               "scenario_acts_sha256": acts_sha,
               "scenario_acts_sha256_banked": SCENARIO_ACTS_SHA,
               "prompt_order_note": "matrix rows align to this order (protocol §2)"}
    print(f"[pairing gate] PASS — 12 prompts; stimulus sha {stim_sha[:12]}; banked act sha verified")
    return prompts, pairing


def load_model_pinned(revision: str):
    from transformer_lens.model_bridge import TransformerBridge
    bridge = TransformerBridge.boot_transformers(
        MODEL, device="cuda", dtype=torch.bfloat16, revision=revision,
    )
    # tl_compatibility_mode stays FALSE — grid regime (gemma_grid/CLAUDE.md).
    bridge.eval()
    assert int(bridge.cfg.n_layers) == N_LAYERS and int(bridge.cfg.d_model) == D_MODEL
    return bridge


def resid_post_at(bridge, tokens: torch.Tensor, pos: int) -> torch.Tensor:
    """(N_LAYERS, D_MODEL) f32 residuals at one position, via run_with_cache."""
    with torch.no_grad():
        _, cache = bridge.run_with_cache(
            tokens, names_filter=lambda n: n.endswith("hook_resid_post"))
    keyed = {}
    for name, act in cache.items():
        m = re.search(r"\.(\d+)\.", name)
        if m is not None and name.endswith("hook_resid_post"):
            keyed[int(m.group(1))] = act[0, pos, :].float().cpu()
    assert sorted(keyed) == list(range(N_LAYERS)), \
        f"resid_post layers captured {sorted(keyed)[:3]}…{sorted(keyed)[-3:]} != 0..{N_LAYERS-1} — STOP"
    return torch.stack([keyed[i] for i in range(N_LAYERS)], dim=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", action="store_true", help="1 prompt, all arms — show Adam")
    a = ap.parse_args()

    revision = fingerprint_gate()
    prompts, pairing = load_prompts()
    if a.pilot:
        prompts = prompts[:1]

    print(f"Loading {MODEL} (compat=False, bf16, revision pinned)...")
    bridge = load_model_pinned(revision)
    tok = bridge.tokenizer
    bos = tok.bos_token_id

    arms: dict[str, dict] = {k: {"residuals": [], "final_token_id": [], "final_token_str": [],
                                 "n_tokens": []} for k in ("bare", "template", "colon")}
    arm_defs = {
        "bare": "to_tokens(prompt), BOS-prepended, measurement at FINAL token "
                "(fig2-gate winner P1_bare_final; PRIMARY, matched to E1 base side)",
        "template": "tokenizer.apply_chat_template(user=prompt, add_generation_prompt=True), "
                    "measurement at final token (SENSITIVITY; disclosed regime shift)",
        "colon": '"Human: {p}\\n\\nAssistant:" raw format, measurement at final token '
                 "(TERTIARY IT-side regime probe; excluded from P-colon)",
    }
    template_render = None
    for p in prompts:
        variants = {}
        t_bare = bridge.to_tokens(p)
        variants["bare"] = t_bare
        chat_out = tok.apply_chat_template([{"role": "user", "content": p}],
                                           add_generation_prompt=True, tokenize=True,
                                           return_tensors="pt")
        # transformers 5.15.0: returns BatchEncoding (dict-like), not a tensor —
        # index input_ids (the 5.15.0-correct call); tolerant of older tensor returns.
        chat_ids = chat_out if torch.is_tensor(chat_out) else chat_out["input_ids"]
        variants["template"] = chat_ids.to(t_bare.device)
        if template_render is None:
            template_render = tok.apply_chat_template([{"role": "user", "content": "X"}],
                                                      add_generation_prompt=True, tokenize=False)
        variants["colon"] = bridge.to_tokens(f"Human: {p}\n\nAssistant:")
        for arm, toks in variants.items():
            assert toks.shape[0] == 1
            if arm != "template":
                assert int(toks[0, 0]) == bos, f"{arm}: first token != BOS — STOP"
            pos = toks.shape[1] - 1
            r = resid_post_at(bridge, toks, pos)
            arms[arm]["residuals"].append(r)
            fid = int(toks[0, pos]); fstr = tok.decode([fid])
            arms[arm]["final_token_id"].append(fid)
            arms[arm]["final_token_str"].append(fstr)
            arms[arm]["n_tokens"].append(int(toks.shape[1]))
        colon_str = arms["colon"]["final_token_str"][-1]
        assert ":" in colon_str, f"colon arm final token {colon_str!r} has no colon — STOP"
    for arm in arms:
        arms[arm]["residuals"] = torch.stack(arms[arm]["residuals"], dim=0)  # (n,46,4608)
        print(f"[{arm}] residuals {tuple(arms[arm]['residuals'].shape)}; "
              f"final tokens {arms[arm]['final_token_str']}")

    payload = {
        "prompts": prompts, "arms": arms, "arm_definitions": arm_defs,
        "pairing": pairing,
        "provenance": {
            "job": "GGE1-G4 / IT colon analysis (pod B)",
            "model_name": MODEL, "hf_revision_resolved": revision,
            "torch": torch.__version__, "tl_compatibility_mode": False,
            "dtype": "bfloat16 (resid_post -> f32)",
            "template_render_example": template_render,
            "pod_gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
            "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "host": platform.node(), "pilot": bool(a.pilot),
            "naming_note": "filename keeps the historical 'colon' label; PRIMARY arm is "
                           "bare-final per Amendment 1 / the frozen fig2 protocol",
        },
    }
    out = OUT.with_name("colon_analysis_it_PILOT.pt") if a.pilot else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp"); torch.save(payload, tmp); tmp.rename(out)
    (out.parent / f"run_it_colon_manifest{'_PILOT' if a.pilot else ''}.json").write_text(
        json.dumps({"rule": "SR.5 — every input hashed at run time",
                    "inputs_sha256": MANIFEST}, indent=1))
    print(f"wrote {out.relative_to(REPO)} + SR.5 manifest")
    if a.pilot:
        print("\n[PILOT] one prompt, three arms captured. SHOW ADAM before the full 12.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
