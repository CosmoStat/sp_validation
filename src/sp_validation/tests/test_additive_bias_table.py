"""The additive-bias table of CosmologyValidation, on a stub without catalogues."""

import os

from sp_validation.cosmo_val.catalog_characterization import (
    CatalogCharacterizationMixin,
)


class _Stub(CatalogCharacterizationMixin):
    def __init__(self, output_dir):
        self.versions = ["A_v1", "B"]
        self.cc = {"A_v1": {}, "B": {"label": "Bee"}, "paths": {"output": output_dir}}
        self._c1 = {"A_v1": -1.6e-4, "B": 1e-4}
        self._c2 = {"A_v1": 2.3e-4, "B": 2e-4}
        self._c_err = {
            "A_v1": {"sn": (3e-5, 3e-5), "jk": (8e-5, 6e-5)},
            "B": {"sn": (3e-5, 3e-5), "jk": None},
        }

    def _output_path(self, *parts):
        return os.path.join(self.cc["paths"]["output"], *parts)

    def print_done(self, msg):
        pass


def test_print_additive_bias_tables(tmp_path):
    stub = _Stub(str(tmp_path))

    stub.print_additive_bias(labels={"A_v1": "DES weights"})

    txt = (tmp_path / "c_non_tomographic.txt").read_text().splitlines()
    assert txt[1].split() == [
        "A_v1",
        "-0.00016+/-0.00003",
        "0.00023+/-0.00003",
        "-0.00016+/-0.00008",
        "0.00023+/-0.00006",
    ]
    # Without jackknife only the shape-noise columns
    assert len(txt[2].split()) == 3

    tex = (tmp_path / "c_non_tomographic.tex").read_text()
    # Jackknife errors where available, else shape noise; labels from the
    # argument, then the config
    assert r"DES weights & $-1.6 \pm 0.8$ & $2.3 \pm 0.6$ \\" in tex
    assert r"Bee & $1.0 \pm 0.3$ & $2.0 \pm 0.3$ \\" in tex
