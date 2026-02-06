# workflow/rules/specs.smk
"""
Method specs — foundational techniques that inform claims.
Methods have no dependencies; they describe how things work.
"""

import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CONFIG_DIR, CLAIMS_DIR defined in Snakefile
SKILL_PATH = os.path.expanduser("~/.claude/skills/conducting-research/templates")

# Method specs — foundational techniques, plot types, analysis methods
METHOD_SPECS = ["cosebis", "pure_eb", "covariance", "cl", "1d_plots", "2d_plots"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Method Evidence Rules
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

localrules: method_spec_evidence, all_method_specs

rule method_spec_evidence:
    """Generate evidence.json for method specs (no dependencies)."""
    input:
        spec=f"{CONFIG_DIR}/{{method_spec}}.md",
    output:
        evidence=f"{CLAIMS_DIR}/{{method_spec}}/evidence.json",
    wildcard_constraints:
        method_spec="|".join(METHOD_SPECS),
    script:
        f"{SKILL_PATH}/scripts/generate_spec_evidence.py"


rule all_method_specs:
    """Build all method spec evidence files."""
    input:
        expand(f"{CLAIMS_DIR}/{{spec}}/evidence.json", spec=METHOD_SPECS),
