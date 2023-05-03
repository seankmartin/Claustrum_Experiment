rule analyse_coherence:
    input:
        "results/index.csv"
    output:
        "results/coherence.csv"
    log:
        "logs/coherence_analysis.log"
    conda:
        "../../envs/nwb.yaml"
    script:
        "../scripts/analyse_lfp_coherence.py"

rule analyse_behaviour:
    input:
        "results/index.csv"
    output:
        "results/behaviour.csv",
        "results/levers.csv"
    log:
        "logs/behaviour.log"
    conda:
        "../../envs/nwb.yaml"
    script:
        "../scripts/analyse_behaviour.py"

rule split_coherence:
    input:
        "results/coherence.csv"
    output:
        "results/coherence_CLA RSC.csv"
    log:
        "logs/split_coherence.log"
    conda:
        "../../envs/nwb.yaml"
    script:
        "../scripts/split_coherence.py"
