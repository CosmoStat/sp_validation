"""Diagnostics use the recipe's map and labels and preserve percentage text."""

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text

from sp_validation import nz_diagnostics


def test_recipe_map_labels_and_percentages(tmp_path, monkeypatch):
    grid = np.linspace(0, 2, 21)
    cell = np.tile(np.arange(4), 2)
    weights = np.tile([1.0, 2.0, 3.0, 4.0], (4, 2))
    paths = {
        key: tmp_path / f"{key}.npz"
        for key in [
            "calibration_arrays",
            "model",
        ]
    }
    np.savez(
        paths["calibration_arrays"],
        z=np.linspace(0.15, 1.1, 8),
        cal_photoz=np.linspace(0.15, 1.1, 8),
        cal_group=cell,
        cal_cell=cell,
        edges=[0.1, 0.37, 0.55, 0.76, 1.2],
        gold=np.ones((4, 4), bool),
        cal_weights=weights,
        grid=grid,
        bin_hists=np.ones((4, len(grid))),
        source_weight_cell=np.arange(1.0, 5.0),
        source_photoz_cell=np.arange(1.0, 5.0) * np.linspace(0.2, 1.0, 4),
    )
    np.savez(paths["model"], group_map=np.arange(4))
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "bins": [
                    {"gold_fraction": 0.9, "weight_fraction": 0.25} for _ in range(4)
                ]
            }
        )
    )
    comparison = tmp_path / "nz.txt"
    np.savetxt(comparison, np.column_stack([grid, np.ones((len(grid), 4))]))
    config = dict(
        paths,
        summary=summary,
        side=2,
        label="Full sample",
        reference_label="Small sample",
        published_label="Published sample",
        reference_nz=comparison,
        published_nz=comparison,
    )
    texts = []

    def inspect_and_close(fig, path):
        labels = fig.findobj(Text)
        assert all(not label.get_usetex() for label in labels)
        texts.extend(label.get_text() for label in labels)
        plt.close(fig)

    monkeypatch.setattr(nz_diagnostics, "_save", inspect_and_close)
    with matplotlib.rc_context({"text.usetex": True}):
        nz_diagnostics.render(config, tmp_path / "figures")
        assert matplotlib.rcParams["text.usetex"]
    assert any("% weight lost" in text for text in texts)
    assert any("outliers" in text and "%" in text for text in texts)
    assert "Full sample" in texts
    assert "Small sample" in texts
    assert any(text.startswith("Published sample, mean") for text in texts)
    assert "Source shear weight per cell" in texts
