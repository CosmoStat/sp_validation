"""Snakemake rules for Moriond 2026 talk figures.

Converts paper-repo PDFs to PNGs for the reveal.js deck. Whenever a
collaborator updates a figure in their paper directory, re-running
``snakemake talk_figures`` picks up the change automatically.
"""

TALK_DIR = "docs/talks/26_Moriond_UNIONS"
PAPER_III = "docs/unions_release/unions_bmodes/Figures"
PAPER_IV  = "docs/unions_release/unions_2d_shear_xi/Figures"
PAPER_V   = "docs/unions_release/unions_harmonic/Figures"

# output name (without .png) → source PDF
TALK_FIGURES = {
    # Paper III — B-mode data vectors
    "pure_eb_data_vector":       f"{PAPER_III}/pure_eb_data_vector.pdf",
    "cosebis_data_vector":       f"{PAPER_III}/cosebis_data_vector.pdf",
    "cl_data_vector":            f"{PAPER_III}/cl_data_vector.pdf",
    "config_space_pte_fiducial": f"{PAPER_III}/config_space_pte_fiducial.pdf",
    "harmonic_config_cosebis_full": f"{PAPER_III}/harmonic_config_cosebis_full.pdf",
    "harmonic_config_cosebis_fiducial": f"{PAPER_III}/harmonic_config_cosebis_fiducial.pdf",
    # Paper IV — configuration-space cosmology
    "best_fit_xipm":                    f"{PAPER_IV}/best_fit_xipm_SP_v1.4.6.3_B.pdf",
    "contour_survey_comparison":        f"{PAPER_IV}/SP_v1.4.6.3_B_fiducial_config_contour_plot.pdf",
    "whisker_plot":                     f"{PAPER_IV}/S8_whisker_plot.pdf",
    "Omega_m_sigma_8_joint_config":     f"{PAPER_IV}/Omega_m_sigma_8_joint_config.pdf",
    "S8_comparison_config_harm":        f"{PAPER_IV}/S8_comparison_config_harm.pdf",
    # Paper V — harmonic-space cosmology
    "Cell_EE_bestfit":              f"{PAPER_V}/paperplot_Cell_EE_and_best_fit.pdf",
    "S8_joint_config_harm":         f"{PAPER_V}/S8_joint_config_harm.pdf",
    "S8_difference_config_harm":    f"{PAPER_V}/S8_difference_config_harm_map_2D.pdf",
    "contours_blind_harmonic":      f"{PAPER_V}/cosmological_constraints_blind/contours_s8_omegam_cell.pdf",
    "contours_config_vs_harmonic":  f"{PAPER_V}/cosmological_constraints_cl_vs_xi/contours_s8_omegam_cell.pdf",
    "contours_weak_lensing_surveys": f"{PAPER_V}/cosmological_constraints_weak_lensing/contours_OMEGA_M_S8_2D_weak_lensing_all_surveys.pdf",
}


localrules: talk_figure, talk_figure_preview, talk_figures, talk_previews, presentation_blind_nz_plot, presentation_s8_convergence, presentation_s8_with_unions, presentation_omega_m_difference, presentation_s8_scatter_mocks, presentation_pte_cosebis

ruleorder: presentation_blind_nz_plot > talk_figure


rule talk_figures:
    """Build all talk PNGs from paper-repo PDFs."""
    input:
        expand(f"{TALK_DIR}/images/{{name}}.png", name=TALK_FIGURES.keys()),
        f"{TALK_DIR}/images/blind_nz_ABC.png",
        f"{TALK_DIR}/images/s8_convergence_whisker.png",
        f"{TALK_DIR}/images/omega_m_difference_config_harm.png",
        f"{TALK_DIR}/images/s8_scatter_config_vs_harmonic.png",
        f"{TALK_DIR}/images/pte_cosebis_talk.png",
        f"{TALK_DIR}/images/s8_convergence_with_unions.png",


rule talk_previews:
    """Build low-res preview PNGs (< 1800px) safe for Claude to read."""
    input:
        expand(f"{TALK_DIR}/images/preview/{{name}}.png", name=TALK_FIGURES.keys()),


rule talk_figure:
    """Convert a single paper PDF to high-res PNG for the talk."""
    input:
        pdf=lambda w: TALK_FIGURES[w.name],
    output:
        f"{TALK_DIR}/images/{{name}}.png",
    wildcard_constraints:
        name="|".join(TALK_FIGURES.keys()),
    container:
        None
    shell:
        "convert -density 300 {input.pdf} -quality 95 {output}"


rule talk_figure_preview:
    """Downscale a talk figure to < 1800px for safe AI reading."""
    input:
        f"{TALK_DIR}/images/{{name}}.png",
    output:
        f"{TALK_DIR}/images/preview/{{name}}.png",
    wildcard_constraints:
        name="|".join(TALK_FIGURES.keys()),
    container:
        None
    shell:
        "convert {input} -resize '1800x>' {output}"


rule presentation_s8_convergence:
    """S8 convergence whisker plot for Moriond slide 2."""
    output:
        f"{TALK_DIR}/images/s8_convergence_whisker.png",
    shell:
        "python {TALK_DIR}/plot_s8_convergence.py"


rule presentation_s8_with_unions:
    """S8 convergence whisker plot WITH UNIONS constraints for slide 14 animation."""
    output:
        f"{TALK_DIR}/images/s8_convergence_with_unions.png",
    shell:
        "python {TALK_DIR}/plot_s8_with_unions.py"


rule presentation_blind_nz_plot:
    """Plot all three blinded n(z) curves for Moriond presentation."""
    input:
        nz_A=lambda w: build_redshift_path(FIDUCIAL["version"], "A"),
        nz_B=lambda w: build_redshift_path(FIDUCIAL["version"], "B"),
        nz_C=lambda w: build_redshift_path(FIDUCIAL["version"], "C"),
    output:
        f"{TALK_DIR}/images/blind_nz_ABC.png",
    script:
        "../scripts/plot_presentation_blind_nz.py"


rule presentation_omega_m_difference:
    """Delta Omega_m histogram from 350 GLASS mocks (config vs harmonic)."""
    input:
        mock_summary="/n09data/guerrini/glass_mock_chains/summary_parameter_constraints_merged_v6.txt",
    output:
        f"{TALK_DIR}/images/omega_m_difference_config_harm.png",
    container:
        None
    shell:
        "python {TALK_DIR}/plot_omega_m_difference.py"


rule presentation_s8_scatter_mocks:
    """S8(config) vs S8(harmonic) scatter from 350 GLASS mocks with data crosshair."""
    input:
        mock_summary="/n09data/guerrini/glass_mock_chains/summary_parameter_constraints_merged_v6.txt",
    output:
        f"{TALK_DIR}/images/s8_scatter_config_vs_harmonic.png",
    container:
        None
    shell:
        "python {TALK_DIR}/plot_s8_scatter_mocks.py"


rule presentation_pte_cosebis:
    """COSEBIS B_n PTE heatmap for Moriond talk (single panel, talk-sized)."""
    input:
        pte_files=[
            f"{TAPESTRY_DIR}/cosebis_pte_matrix/pte_values/{FIDUCIAL['version']}/{FIDUCIAL['blind']}/pte_{i:03d}_{j:03d}.json"
            for i, j in PTE_SCALE_CUT_PAIRS
        ],
    output:
        f"{TALK_DIR}/images/pte_cosebis_talk.png",
    container:
        None
    shell:
        """
        apptainer exec --bind /home,/scratch,/automnt,/n17data,/n23data1,/n09data \
            /n17data/cdaley/containers/containers/ \
            python {TALK_DIR}/plot_pte_cosebis_talk.py
        """
