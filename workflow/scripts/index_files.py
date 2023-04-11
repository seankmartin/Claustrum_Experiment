import simuran

from simuran.loaders.neurochat_loader import NeuroChatLoader
from skm_pyutils.table import df_to_file


def main(path_to_files: str, output_path: str) -> None:
    loader = NeuroChatLoader(system="Axona", pos_extension=".pos")
    df = loader.index_files(path_to_files)
    # df = clean_data(df)
    df_to_file(df, output_path)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        use_snakemake = False
    else:
        use_snakemake = True
    if use_snakemake:
        simuran.set_only_log_to_file(snakemake.log[0])
        main(snakemake.config["data_directory"], snakemake.output[0])
    else:
        main(r"H:\Ham_Data", r"results\axona_file_index.csv")
