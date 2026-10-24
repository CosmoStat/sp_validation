"""Paper readers accept catalogue-policy params under the host launcher."""

import shutil
from pathlib import Path

import yaml
from conftest import REPO, VERSIONS, parse_jobs


def test_bmodes_readers_resolve(toy):
    paper = toy.root / "papers" / "bmodes"
    shutil.copytree(REPO / "papers" / "bmodes", paper,
                    ignore=shutil.ignore_patterns("results", ".snakemake", "__pycache__"))
    shutil.copytree(toy.rundir / "results" / "cosmology", paper / "results" / "cosmology")
    path = paper / "config" / "config.yaml"
    config = yaml.safe_load(path.read_text())
    config["versions"] = list(VERSIONS)
    fid = config["fiducial"]
    fid.update(version=VERSIONS[1], mock_version=VERSIONS[0],
               nz_realisations={"A": VERSIONS[1]})
    config["plotting"]["version_labels"] = {VERSIONS[1]: "Toy"}
    path.write_text(yaml.safe_dump(config))

    # Supply input parts/covariances: this tests ingestion rules, not measurement.
    for version in VERSIONS:
        paths = [toy.cosmo_val / (
            f"{version}_xi_minsep={fid['min_sep_int']}_maxsep={fid['max_sep_int']}"
            f"_nbins={fid['nbins_int']}_npatch=1.sacc"
        )]
        paths.append(Path(toy.common.covariance_path(
            version, "g", fid["min_sep_int"], fid["max_sep_int"], fid["nbins_int"],
            "_masked",
        )))
        for bins in (32, 96):
            paths.extend([
                toy.cosmo_val / f"pseudo_cl_{version}_powspace_nbins={bins}.sacc",
                toy.cosmo_val / f"pseudo_cl_cov_{version}_powspace_nbins={bins}.fits",
            ])
        for item in paths:
            item.parent.mkdir(parents=True, exist_ok=True)
            item.touch()

    rules = ["cosebis_version_comparison", "cosebis_data_vector", "cl_data_vector",
             "cl_version_comparison", "harmonic_space_pte_matrices",
             "cosebis_filter_overlay"]
    result = toy.snakemake(
        "-n", *rules,
        f"results/paper_plots/intermediate/{VERSIONS[1]}_pure_eb.npz",
        f"results/tapestry/cosebis_pte_matrix/pte_values/{VERSIONS[1]}/pte_000_006.json",
        "results/tapestry/harmonic_config_cosebis_comparison_full/evidence.json",
        cwd=paper,
    )
    assert result.returncode == 0, result.stdout
    scheduled = {job.rule for job in parse_jobs(result.stdout)}
    assert set(rules + ["pure_eb_modes", "compute_cosebis_pte",
                        "harmonic_config_cosebis_comparison"]) <= scheduled
