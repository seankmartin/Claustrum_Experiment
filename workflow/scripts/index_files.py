import simuran
import os

from simuran.loaders.neurochat_loader import NeurochatLoader
from skm_pyutils.table import df_to_file
from skm_pyutils.path import get_all_files_in_dir


def main(path_to_files: str, output_path: str) -> None:
    loader = NeurochatLoader(system="Axona", pos_extension=".pos")
    df = loader.index_files(path_to_files)
    for i, row in df.iterrows():
        file_ = os.path.join(row["directory"], row["filename"])
        avi_files = get_all_files_in_dir(row["directory"], ".avi")
        if os.path.exists(file_[:-3] + "avi"):
            df.at[i, "has_video"] = True
            df.at[i, "video_file"] = file_[:-3] + "avi"
        elif len(avi_files) > 0:
            if len(avi_files) > 1:
                print("More than one avi file found for {}".format(file_))
            df.at[i, "has_video"] = True
            df.at[i, "video_file"] = avi_files[0]
        else:
            df.at[i, "has_video"] = False
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
