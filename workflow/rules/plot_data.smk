rule plot_coherence:
    input: 
        "results/coherence.csv"
    output:
        "results/png/coherence.png"
    log:
        "logs/plot_coherence.log"
    conda:
        "../../envs/nwb.yaml"
    script:
        "../scripts/plot_coherence.py"