rule index_files:
    output:
        "results/axona_file_index.csv"
    log:
        "logs/axona_file_index.log"
    conda:
        "../../envs/nwb.yaml"
    script:
        "../scripts/index_files.py"

rule process_metadata:
    input:
        "results/axona_file_index.csv"
    output:
        "results/metadata_parsed.csv"
    log:
        "logs/metadata_parsed.log"
    conda:
        "../../envs/nwb.yaml"
    script:
        "../scripts/parse_metadata.py"
    
rule convert_from_axona:
    input:
        "results/metadata_parsed.csv"
    output:
        "results/converted_data.csv"
    log:
        "logs/convert_to_nwb.log"
    conda:
        "../../envs/nwb.yaml"
    script:
        "../scripts/convert_to_nwb.py"

rule process_nwb:
    input:
        "results/converted_data.csv"
    output:
        "results/index.csv"
    log:
        "logs/process_nwb.log"
    conda:
        "../../envs/nwb.yaml"
    script:
        "../scripts/process_lfp.py"
