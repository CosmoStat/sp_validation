"""
B-mode analysis rules for PureEB and COSEBIS calculations
"""

def bmodes_versions():
    """Return ordered unique catalog versions configured for B-mode analysis."""
    return list(dict.fromkeys(config["versions"]))

rule plot_eb:
    input:
        xi_reporting="/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/output/{version}_xi_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.txt",
        xi_integration="/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/output/{version}_xi_minsep={min_sep_int}_maxsep={max_sep_int}_nbins={nbins_int}_npatch={npatch}.txt",
        cov_integration=lambda w: f"/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/data/covariance/covariance_{w.version}_{config['fiducial']['blind']}_g_minsep={{min_sep_int}}_maxsep={{max_sep_int}}_nbins={{nbins_int}}/covariance_{w.version}_{config['fiducial']['blind']}_g_minsep={{min_sep_int}}_maxsep={{max_sep_int}}_nbins={{nbins_int}}_processed.txt",
        # Use version-specific n(z) from new location
        redshift_file=lambda w: build_redshift_path(w.version, config['fiducial']['blind']),
    output:
        int_vs_rep="/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/output/{version}_eb_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_minsepint={min_sep_int}_maxsepint={max_sep_int}_nbinsint={nbins_int}_npatch={npatch}_varmethod=semi-analytic_integration_vs_reporting.png",
        xis="/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/output/{version}_eb_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_minsepint={min_sep_int}_maxsepint={max_sep_int}_nbinsint={nbins_int}_npatch={npatch}_varmethod=semi-analytic_xis.png",
        ptes="/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/output/{version}_eb_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_minsepint={min_sep_int}_maxsepint={max_sep_int}_nbinsint={nbins_int}_npatch={npatch}_varmethod=semi-analytic_ptes.png",
        cov="/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/output/{version}_eb_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_minsepint={min_sep_int}_maxsepint={max_sep_int}_nbinsint={nbins_int}_npatch={npatch}_varmethod=semi-analytic_covariance.png",
    params:
        version="{version}",
        min_sep="{min_sep}",
        max_sep="{max_sep}",
        nbins="{nbins}",
        min_sep_int="{min_sep_int}",
        max_sep_int="{max_sep_int}",
        nbins_int="{nbins_int}",
        npatch="{npatch}",
    threads: 1
    script:
        "../scripts/plot_eb.py"


rule plot_cosebis:
    input:
        xi_integration=f"/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/output/{{version}}_xi_minsep={config['fiducial']['min_sep_int']}_maxsep={config['fiducial']['max_sep_int']}_nbins={config['fiducial']['nbins_int']}_npatch={{npatch}}.txt",
        cov_integration=lambda w: f"/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/data/covariance/covariance_{w.version}_{config['fiducial']['blind']}_g_minsep={config['fiducial']['min_sep_int']}_maxsep={config['fiducial']['max_sep_int']}_nbins={config['fiducial']['nbins_int']}/covariance_{w.version}_{config['fiducial']['blind']}_g_minsep={config['fiducial']['min_sep_int']}_maxsep={config['fiducial']['max_sep_int']}_nbins={config['fiducial']['nbins_int']}_processed.txt",
    output:
        modes=f"/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/output/{{version}}_cosebis_minsep={config['fiducial']['min_sep_int']}_maxsep={config['fiducial']['max_sep_int']}_nbins={config['fiducial']['nbins_int']}_npatch={{npatch}}_varmethod=analytic_nmodes={config['fiducial']['nmodes']}_scalecut={config['fiducial']['fiducial_min_scale']}-{config['fiducial']['fiducial_max_scale']}_cosebis.png",
        ptes=f"/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/output/{{version}}_cosebis_minsep={config['fiducial']['min_sep_int']}_maxsep={config['fiducial']['max_sep_int']}_nbins={config['fiducial']['nbins_int']}_npatch={{npatch}}_varmethod=analytic_nmodes={config['fiducial']['nmodes']}_scalecut={config['fiducial']['fiducial_min_scale']}-{config['fiducial']['fiducial_max_scale']}_scalecut_ptes.png", 
        cov=f"/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/output/{{version}}_cosebis_minsep={config['fiducial']['min_sep_int']}_maxsep={config['fiducial']['max_sep_int']}_nbins={config['fiducial']['nbins_int']}_npatch={{npatch}}_varmethod=analytic_nmodes={config['fiducial']['nmodes']}_scalecut={config['fiducial']['fiducial_min_scale']}-{config['fiducial']['fiducial_max_scale']}_covariance.png",
    params:
        version="{version}",
        min_sep=config['fiducial']['min_sep'],
        max_sep=config['fiducial']['max_sep'],
        nbins=config['fiducial']['nbins'],
        nmodes=config['fiducial']['nmodes'],
        min_sep_int=config['fiducial']['min_sep_int'],
        max_sep_int=config['fiducial']['max_sep_int'],
        nbins_int=config['fiducial']['nbins_int'],
        npatch="{npatch}",
        fiducial_min_scale=config['fiducial']['fiducial_min_scale'],
        fiducial_max_scale=config['fiducial']['fiducial_max_scale'],
    threads: 1
    script:
        "../scripts/plot_cosebis.py"


rule plot_eb_fiducial:
    input:
        rules.plot_eb.output.xis.format(**config["fiducial"]),


rule plot_cosebis_fiducial:
    input:
        rules.plot_cosebis.output.modes.format(**config["fiducial"]),


rule bmodes_all:
    """Calculate both PureEB and COSEBIS modes for all versions using fiducial parameters"""
    input:
        # PureEB plots for all versions
        [rules.plot_eb.output.xis.format(**{**config["fiducial"], "version": version}) for version in bmodes_versions()],
        # COSEBIS plots for all versions  
        [rules.plot_cosebis.output.modes.format(**{**config["fiducial"], "version": version}) for version in bmodes_versions()],


rule bmodes_pdf:
    """Combine all PureEB and COSEBIS plot variants into a single PDF deck"""
    input:
        pure_eb_int_vs_rep=[
            rules.plot_eb.output.int_vs_rep.format(**{**config["fiducial"], "version": version})
            for version in bmodes_versions()
        ],
        pure_eb_xis=[
            rules.plot_eb.output.xis.format(**{**config["fiducial"], "version": version})
            for version in bmodes_versions()
        ],
        pure_eb_ptes=[
            rules.plot_eb.output.ptes.format(**{**config["fiducial"], "version": version})
            for version in bmodes_versions()
        ],
        pure_eb_covariances=[
            rules.plot_eb.output.cov.format(**{**config["fiducial"], "version": version})
            for version in bmodes_versions()
        ],
        cosebis_modes=[
            rules.plot_cosebis.output.modes.format(**{**config["fiducial"], "version": version})
            for version in bmodes_versions()
        ],
        cosebis_ptes=[
            rules.plot_cosebis.output.ptes.format(**{**config["fiducial"], "version": version})
            for version in bmodes_versions()
        ],
        cosebis_covariances=[
            rules.plot_cosebis.output.cov.format(**{**config["fiducial"], "version": version})
            for version in bmodes_versions()
        ],
    output:
        pdf="results/bmodes_all/bmodes_plots_leak_corr.pdf",
    params:
        ordered_paths=[
            *[
                rules.plot_eb.output.int_vs_rep.format(**{**config["fiducial"], "version": version})
                for version in bmodes_versions()
            ],
            *[
                rules.plot_eb.output.xis.format(**{**config["fiducial"], "version": version})
                for version in bmodes_versions()
            ],
            *[
                rules.plot_eb.output.ptes.format(**{**config["fiducial"], "version": version})
                for version in bmodes_versions()
            ],
            *[
                rules.plot_eb.output.cov.format(**{**config["fiducial"], "version": version})
                for version in bmodes_versions()
            ],
            *[
                rules.plot_cosebis.output.modes.format(**{**config["fiducial"], "version": version})
                for version in bmodes_versions()
            ],
            *[
                rules.plot_cosebis.output.ptes.format(**{**config["fiducial"], "version": version})
                for version in bmodes_versions()
            ],
            *[
                rules.plot_cosebis.output.cov.format(**{**config["fiducial"], "version": version})
                for version in bmodes_versions()
            ],
        ],
    threads: 1
    script:
        "../scripts/assemble_bmodes_pdf.py"


localrules:
    plot_eb_fiducial,
    plot_cosebis_fiducial,
    bmodes_all,
    bmodes_pdf,
    # Note: plot_eb and plot_cosebis removed from localrules - they now run on compute nodes
