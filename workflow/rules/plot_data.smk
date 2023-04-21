rule plot_coherence:
    input: 
        "results/coherence.csv"
    output:
        directory("results/coherence")
    log:
        "logs/plot_coherence.log"
    conda:
        "../../envs/nwb.yaml"
    script:
        "../scripts/plot_coherence.py"

rule plot_behaviour:
    input:
        "results/behaviour.csv",
        "results/levers.csv"
    output: 
        directory("results/behaviour")
    log:
        "logs/plot_behaviour.log"
    conda:
        "../../envs/nwb.yaml"
    script:
        "../scripts/plot_behaviour.py"