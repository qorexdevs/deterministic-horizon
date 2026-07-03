"""Snapshot test pinning MODEL_HORIZONS to the published Table 3 values."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deterministic_horizon.policy import MODEL_HORIZONS

SNAPSHOT = Path(__file__).parent / "snapshots" / "model_horizons.json"


def test_model_horizons_match_paper_table3():
    """MODEL_HORIZONS must stay verbatim to Table 3 unless the snapshot is refreshed."""
    loaded = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["horizons"]
    assert set(loaded) == set(MODEL_HORIZONS), (
        "MODEL_HORIZONS keys drifted from the Table 3 snapshot. "
        "Updated paper values? Refresh tests/snapshots/model_horizons.json."
    )
    for model, params in MODEL_HORIZONS.items():
        assert params == pytest.approx(loaded[model]), (
            f"{model} diverged from paper Table 3. Updated paper values? "
            "Refresh the snapshot and cite the replication in the PR."
        )
