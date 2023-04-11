rule index_files:
    output:
        "results/axona_file_index.csv"
    log:
        "logs/axona_file_index.log"
    conda:
        "../../envs/axona.yaml"
    script:
        "../scripts/index_files.py"