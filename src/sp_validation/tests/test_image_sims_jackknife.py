"""Shared-tile, response-recalibrating m/c covariance."""

import numpy as np
import pytest

from sp_validation.image_sims import ImageSimMBias


@pytest.mark.parametrize("refit", [False, True])
def test_shared_tile_deletions_recalibrate_both_branches(refit):
    estimator = ImageSimMBias(
        dict(
            shear_amplitude=0.025,
            match_radius_deg=0.0002,
            pair_match=False,
            w_cols=["w_des"],
            n_bootstrap=5,
            bootstrap_seed=42,
        )
    )
    tiles = np.array(["a", "a", "b", "c"])
    labels = np.array(["c", "a", "b"])
    expected = []
    fixed_response = []
    for p, m, comp in estimator.pairs:
        for name, sign in ((p, 1), (m, -1)):
            raw = np.array([[0.1, -0.2, 0.03, 0.2], [0.2, -0.1, 0.3, -0.2]])
            raw[comp] += sign * 0.025
            # Off-diagonal terms exercise matrix orientation and coupling.
            response = np.array([[0.8, 0.03], [-0.02, 0.7]])
            deleted_R = np.array(
                [response + np.eye(2) * sign * x for x in (0.04, -0.02, 0.01)]
            )
            g = np.linalg.solve(response, raw)
            estimator.cats[name] = dict(
                raw=raw,
                e1=g[0],
                e2=g[1],
                tiles=tiles,
                jk_labels=labels.copy(),
                jk_R=deleted_R,
                w={"w_des": np.array([1.0, 2.0, 3.0, 4.0])},
                weight_scheme="des",
                W_policy="refit" if refit else "fixed",
                jk_g=np.array(
                    [
                        np.linalg.solve(
                            deleted_R[j],
                            np.average(
                                raw[:, tiles != label],
                                axis=1,
                                weights=np.array([1.0, 2.0, 3.0, 4.0])[tiles != label]
                                ** 2,
                            ),
                        )
                        for j, label in enumerate(labels)
                    ]
                ),
            )
        draws, held_R = [], []
        for label in sorted(labels):
            means, fixed = [], []
            for name in (p, m):
                cat = estimator.cats[name]
                keep = tiles != label
                mean = np.average(
                    cat["raw"][:, keep], axis=1, weights=cat["w"]["w_des"][keep]
                )
                index = list(labels).index(label)
                means.append(
                    cat["jk_g"][index, comp]
                    if refit
                    else np.linalg.solve(cat["jk_R"][index], mean)[comp]
                )
                fixed.append(
                    np.average(
                        cat[f"e{comp + 1}"][keep], weights=cat["w"]["w_des"][keep]
                    )
                )
            draws.append([(means[0] - means[1]) / 0.05 - 1, (means[0] + means[1]) / 2])
            held_R.append((fixed[0] - fixed[1]) / 0.05 - 1)
        expected.extend(np.array(draws).T)
        fixed_response.append(held_R)
    result = estimator.run(False)
    expected = np.array(expected).T
    delta = expected - expected.mean(axis=0)
    covariance = 2 / 3 * delta.T @ delta
    np.testing.assert_allclose(result["mc_jackknife"], expected)
    np.testing.assert_allclose(result["mc_covariance"], covariance)
    np.testing.assert_allclose(
        [result[k + "_err"] for k in result["covariance_order"]],
        np.sqrt(covariance.diagonal()),
    )
    assert abs(covariance[0, 1]) > 1e-6
    assert not np.allclose(expected[:, 0], fixed_response[0])
    assert result["uncertainty"] == "shared_tile_response_jackknife"
    assert "conditional_object_bootstrap_fixed_response" not in result
    estimator.cats[estimator.pairs[0][1]]["jk_labels"][0] = "d"
    with pytest.raises(ValueError, match="identical parent"):
        estimator.run(False)
