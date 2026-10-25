"""
Load a grid-cell YAML into a validated (Config, CellSpec) pair.

One cell = one YAML under gemma_grid/configs/. This is the config-driven
front end the PLAN asks for: it reuses emotion_probes.config.Config unchanged
and adds only the grid-specific bookkeeping (depth-fraction rule, per-cell
shape assertions) that Config does not carry.

The loader itself does NO model loading and touches NO GPU. It produces:
  - a Config the existing pipeline already knows how to run, and
  - a CellSpec of the values to ASSERT once the model is actually loaded.

Layer derivation follows the grid invariant (gemma_grid/CLAUDE.md):
    analysis_layer = round(depth_fraction * n_layers)
computed from the model's OWN n_layers at load time, never trusted from YAML.
The YAML's expect_* fields are assertion targets, not sources of truth.

Usage:
    from gemma_grid.scripts.cell_config import load_cell, assert_model_matches
    cfg, spec = load_cell("gemma_grid/configs/gemma-2-2b-base.yaml", project_root)
    bridge = load_model(cfg)                 # existing emotion_probes loader
    layer = assert_model_matches(bridge, spec)   # raises on any mismatch
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from emotion_probes.config import Config


@dataclass(frozen=True)
class CellSpec:
    """Grid bookkeeping for one cell — everything Config does not carry."""
    cell_id: str
    tuning: str                 # "base" | "it"
    status: str
    depth_fraction: float
    expect_n_layers: int
    expect_d_model: int
    expect_analysis_layer: int
    stories: str                # repo-relative path
    neutral: str                # repo-relative path
    source_yaml: Path

    def resolved_layer(self, n_layers: int) -> int:
        """The grid layer rule, applied to a model's actual layer count."""
        return round(self.depth_fraction * n_layers)


_REQUIRED = (
    "cell_id", "model_id", "tuning", "tl_compatibility_mode", "dtype",
    "device", "skip_first_n_tokens", "pca_variance_target", "emotion_subset",
    "stories", "neutral", "depth_fraction", "expect_n_layers",
    "expect_d_model", "analysis_layer_expected",
)


def load_cell(yaml_path: str | Path, project_root: str | Path) -> tuple[Config, CellSpec]:
    """
    Parse a cell YAML into (Config, CellSpec). Pure — no model, no GPU.

    Enforces the grid's hard invariants at parse time so a bad cell cannot
    reach a GPU run:
      * tl_compatibility_mode MUST be False (match the anchor; see CLAUDE.md).
      * emotion_subset MUST be null (full 171).
      * the YAML's own arithmetic must be self-consistent:
        round(depth_fraction * expect_n_layers) == analysis_layer_expected.
    """
    yaml_path = Path(yaml_path)
    raw = yaml.safe_load(yaml_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{yaml_path}: expected a mapping, got {type(raw).__name__}")

    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"{yaml_path}: missing required keys {missing}")

    # --- Hard invariants (fail loud, fail early) ---
    if raw["tl_compatibility_mode"] is not False:
        raise ValueError(
            f"{yaml_path}: tl_compatibility_mode must be false to match the "
            f"anchor (got {raw['tl_compatibility_mode']!r}). See gemma_grid/CLAUDE.md."
        )
    if raw["emotion_subset"] is not None:
        raise ValueError(
            f"{yaml_path}: emotion_subset must be null (full 171); "
            f"got {raw['emotion_subset']!r}."
        )
    if raw["tuning"] not in ("base", "it"):
        raise ValueError(f"{yaml_path}: tuning must be 'base' or 'it', got {raw['tuning']!r}")

    depth_fraction = float(raw["depth_fraction"])
    expect_n_layers = int(raw["expect_n_layers"])
    expect_d_model = int(raw["expect_d_model"])
    expect_layer = int(raw["analysis_layer_expected"])

    # The YAML must be internally consistent: its stated expected layer has to
    # be what the grid rule produces from its stated layer count. This catches
    # a hand-edited table typo before the model is ever loaded.
    rule_layer = round(depth_fraction * expect_n_layers)
    if rule_layer != expect_layer:
        raise ValueError(
            f"{yaml_path}: analysis_layer_expected={expect_layer} but "
            f"round({depth_fraction} * {expect_n_layers}) = {rule_layer}. "
            f"Fix the YAML so its own arithmetic agrees."
        )

    cfg = Config(
        model_name=raw["model_id"],
        device=raw["device"],
        dtype=raw["dtype"],
        tl_compatibility_mode=False,
        emotion_subset=None,
        skip_first_n_tokens=int(raw["skip_first_n_tokens"]),
        pca_variance_target=float(raw["pca_variance_target"]),
        primary_layer=None,  # resolved from the real model at load, then asserted
        project_root=Path(project_root),
    )
    spec = CellSpec(
        cell_id=raw["cell_id"],
        tuning=raw["tuning"],
        status=str(raw.get("status", "")),
        depth_fraction=depth_fraction,
        expect_n_layers=expect_n_layers,
        expect_d_model=expect_d_model,
        expect_analysis_layer=expect_layer,
        stories=raw["stories"],
        neutral=raw["neutral"],
        source_yaml=yaml_path,
    )
    return cfg, spec


def assert_model_matches(bridge, spec: CellSpec) -> int:
    """
    After the model is loaded, confirm it IS the cell the YAML claims, and
    return the analysis layer derived from the model's own layer count.

    Raises AssertionError on any mismatch — a wrong checkpoint, an unexpected
    layer count, or a depth-fraction resolution that disagrees with the YAML.
    Nothing downstream (extraction, analysis) runs unless this passes.
    """
    n_layers = int(bridge.cfg.n_layers)
    d_model = int(bridge.cfg.d_model)
    assert n_layers == spec.expect_n_layers, (
        f"{spec.cell_id}: model reports n_layers={n_layers}, "
        f"YAML expects {spec.expect_n_layers}"
    )
    assert d_model == spec.expect_d_model, (
        f"{spec.cell_id}: model reports d_model={d_model}, "
        f"YAML expects {spec.expect_d_model}"
    )
    layer = spec.resolved_layer(n_layers)
    assert layer == spec.expect_analysis_layer, (
        f"{spec.cell_id}: round({spec.depth_fraction} * {n_layers}) = {layer}, "
        f"YAML expects analysis layer {spec.expect_analysis_layer}"
    )
    return layer
