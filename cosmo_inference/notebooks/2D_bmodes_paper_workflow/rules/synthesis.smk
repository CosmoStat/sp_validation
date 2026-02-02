# workflow/rules/synthesis.smk
"""
Synthesis — paper specs, dashboard, and paper integration.
Synthesis rules aggregate claims into papers and generate outputs for publication.
"""

import os
import socket
import subprocess
import sys

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Variables from included files: CONFIG_DIR, CLAIMS_DIR (Snakefile); SKILL_PATH, METHOD_SPECS (specs.smk)

# Claim rules that produce evidence.json — single source of truth for all_claims and claims_dashboard
# Each entry is a rule name; we access rules.X.output to get all outputs
CLAIM_RULES = [
    "cosebis_version_comparison",
    "cosebis_data_vector",
    "pure_eb_data_vector",
    "pure_eb_version_comparison",
    "pure_eb_covariance",
    "cl_data_vector",
    "cl_version_comparison",
    "config_space_pte_matrices",
    "harmonic_space_pte_matrices",
    "bb_covariance_blind_independence",
    "covariance_blind_consistency",
]


def _claim_outputs():
    """Get all outputs from claim rules."""
    return {name: getattr(rules, name).output for name in CLAIM_RULES}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Paper Macros
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

localrules: xi_cosmology_paper, paper_macros, bmodes_paper_spec, all_claims, spec_dependencies, claims_dashboard, serve_claims

rule xi_cosmology_paper:
    """Spec for B-mode reporting in configuration-space paper (Goh et al.).

    Depends on the two B-mode claims plus covariance consistency, produces macros for Paper II.
    Reports fiducial version, n=6 COSEBIS, joint pure-mode PTEs at both full and fiducial scales.
    Also generates evidence.json for dashboard dependency tracking.
    """
    input:
        spec=f"{CONFIG_DIR}/xi_cosmology_paper.md",
        cosebis_evidence=rules.cosebis_version_comparison.output.evidence,
        pure_eb_evidence=rules.pure_eb_data_vector.output.evidence,
        covariance_evidence=rules.covariance_blind_consistency.output.evidence,
    output:
        macros="docs/unions_release/unions_2d_shear_xi/claims_macros.tex",
        evidence=f"{CLAIMS_DIR}/xi_cosmology_paper/evidence.json",
    params:
        claims_dir=CLAIMS_DIR,
    script:
        "../scripts/generate_paper_macros.py"


rule paper_macros:
    """Generate LaTeX macros and tables for B-modes paper (Daley et al.)."""
    input:
        cosebis_evidence=rules.cosebis_version_comparison.output.evidence,
        pure_eb_evidence=rules.pure_eb_data_vector.output.evidence,
        # PTE composite evidence for table generation
        config_space_pte=rules.config_space_pte_matrices.output.evidence,
        harmonic_space_pte=rules.harmonic_space_pte_matrices.output.evidence,
    output:
        bmodes="docs/unions_release/unions_bmodes/claims_macros.tex",
        pte_table_results="docs/unions_release/unions_bmodes/pte_table_results.tex",
        pte_table_appendix="docs/unions_release/unions_bmodes/pte_table_appendix.tex",
    params:
        claims_dir=CLAIMS_DIR,
    script:
        "../scripts/generate_paper_macros.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Paper Specs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rule bmodes_paper_spec:
    """Generate evidence.json for bmodes_paper spec.

    Dependencies include both evidence files and figure outputs to ensure
    dashboard regenerates all stale plots.
    """
    input:
        spec=f"{CONFIG_DIR}/bmodes_paper.md",
        # Upstream evidence (using rules.X.output for single source of truth)
        pure_eb_covariance=rules.pure_eb_covariance.output.evidence,
        pure_eb_data_vector=rules.pure_eb_data_vector.output.evidence,
        cosebis_version_comparison=rules.cosebis_version_comparison.output.evidence,
        cl_data_vector=rules.cl_data_vector.output.evidence,
        config_space_pte=rules.config_space_pte_matrices.output.evidence,
        harmonic_space_pte=rules.harmonic_space_pte_matrices.output.evidence,
        # Paper figure dependencies (ensures dashboard regenerates version comparison plots)
        pure_eb_version_comparison=rules.pure_eb_version_comparison.output.evidence,
        cosebis_bmode_stacked=rules.cosebis_version_comparison.output.paper_stacked,
        # Consistency checks
        bb_covariance_blind=rules.bb_covariance_blind_independence.output.evidence,
        covariance_blind_consistency=rules.covariance_blind_consistency.output.evidence,
    output:
        evidence=f"{CLAIMS_DIR}/bmodes_paper/evidence.json",
    script:
        f"{SKILL_PATH}/scripts/generate_spec_evidence.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Aggregate Targets
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rule all_claims:
    """Aggregate target for all claim evidence (used by spec_dependencies)."""
    input:
        method_specs=expand(f"{CLAIMS_DIR}/{{spec}}/evidence.json", spec=METHOD_SPECS),
        bmodes_paper=rules.bmodes_paper_spec.output,
        xi_cosmology_paper=rules.xi_cosmology_paper.output,
        **_claim_outputs(),


rule spec_dependencies:
    """Extract spec dependency graph from snakemake DAG.

    Queries snakemake's own detailed-summary to derive which specs depend
    on which, based on actual input file declarations. Single source of truth.
    """
    output:
        deps=f"{CLAIMS_DIR}/deps.json",
    run:
        import json
        from pathlib import Path

        # Use --dry-run instead of --forceall to avoid timestamp pollution
        r = subprocess.run(
            ["snakemake", "--dry-run", "--detailed-summary", "all_claims"],
            capture_output=True, text=True
        )

        specs = {}
        for line in r.stdout.split("\n"):
            p = line.split("\t")
            if len(p) < 7 or "evidence.json" not in p[0]:
                continue
            sid = Path(p[0]).parent.name
            deps = []
            for x in p[4].split(","):
                x = x.strip()
                if x.endswith(".md") and Path(x).stem != sid:
                    deps.append(Path(x).stem)
                elif x.endswith("evidence.json") and Path(x).parent.name != sid:
                    deps.append(Path(x).parent.name)
            specs[sid] = {"deps": sorted(set(deps)), "date": p[1], "status": p[6]}

        with open(output.deps, "w") as f:
            json.dump(specs, f, indent=2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dashboard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

rule claims_dashboard:
    """Render claims dashboard with specs and evidence.

    Dashboard reads specs from felt fibers (foundation/claim/synthesis kinds).
    Evidence links to results/claims/{fiber_id}/ directories.
    """
    input:
        config=f"{CONFIG_DIR}/config.yaml",
        # Method specs (foundational, no dependencies)
        method_specs=expand(f"{CLAIMS_DIR}/{{spec}}/evidence.json", spec=METHOD_SPECS),
        # Paper specs (B-modes paper only — xi_cosmology_paper needs covariance_blind_consistency)
        bmodes_paper=rules.bmodes_paper_spec.output,
        paper_macros=rules.paper_macros.output,
        # All claim rules (using shared CLAIM_RULES list)
        **_claim_outputs(),
    output:
        html=f"{CLAIMS_DIR}/index.html",
    params:
        project_name="UNIONS B-modes",
        tagline="Spec-driven validation",
        claims_dir=CLAIMS_DIR,
        skill_path=SKILL_PATH,
    shell:
        """
        python {params.skill_path}/scripts/generate_claims_dashboard.py \
            {output.html} \
            --project-name "{params.project_name}" \
            --tagline "{params.tagline}" \
            --claims-dir {params.claims_dir} \
            --config-file {input.config}
        """


rule serve_claims:
    """Serve the claims dashboard."""
    input:
        html=f"{CLAIMS_DIR}/index.html",
    params:
        skill_path=SKILL_PATH,
        config_dir=CONFIG_DIR,
        claims_dir=CLAIMS_DIR,
        port_start=8000,
    run:
        def find_open_port(start_port, max_attempts=100):
            for port in range(start_port, start_port + max_attempts):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(('', port))
                        return port
                    except OSError:
                        continue
            raise RuntimeError(f"No open port found in range {start_port}-{start_port + max_attempts}")

        port = find_open_port(params.port_start)
        print(f"Starting dashboard on port {port}")

        script_path = os.path.join(params.skill_path, "scripts", "claims_server.py")
        workflow_root = os.path.abspath(os.path.join(workflow.basedir, ".."))
        config_dir_abs = os.path.join(workflow_root, params.config_dir)
        claims_dir_abs = os.path.join(workflow_root, params.claims_dir)
        print(f"Config: {config_dir_abs}")
        print(f"Claims: {claims_dir_abs}")

        # Run server — catch KeyboardInterrupt so Snakemake doesn't report failure
        try:
            result = subprocess.run([sys.executable, script_path,
                            "--claims-dir", claims_dir_abs,
                            "--specs-dir", config_dir_abs,
                            "--port", str(port)])
            if result.returncode > 0:
                print(f"Server exited with code {result.returncode}")
            else:
                print("Dashboard server stopped")
        except KeyboardInterrupt:
            # Ctrl+C is the expected exit path for a server
            print("\nDashboard server stopped")
