rule analyse_coherence:
    input:
        "results/processed_data.csv"
    output:
        "results/coherence.csv"
    log:
        "logs/coherence_analysis.log"
    conda:
        "../../envs/nwb.yaml"
    script:
        "../scripts/analyse_lfp_coherence.py"