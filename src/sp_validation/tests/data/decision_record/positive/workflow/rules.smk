# @sc [decision:workflow_paragraph]
SCALE = 5e-4
UNRELATED = 1000

# @sc [decision:workflow_rule] rule-contract
# This whole rule implements the decision.
rule validate:
    threads: 2
    params:
        nmodes=20,
        angular_limits=[1, 250]

    output: 'results/catalogue.fits'
    shell: 'touch {output}'

rule unrelated:
    threads: 99
