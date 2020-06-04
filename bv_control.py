"""Control script for MEDPC behaviour analysis."""
import os
import json
from datetime import date, timedelta, datetime

import bvmpc.bv_batch
import bvmpc.bv_plot
import bvmpc.bv_analyse
import bvmpc.bv_file
import bvmpc.bv_utils


def main(config_name):
    """Main control for batch process."""
    here = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(here, "Configs", "Behaviour", config_name)
    config = bvmpc.bv_utils.read_cfg(config_path)

    in_dir = config.get("Setup", "in_dir")
    if in_dir[0] == '"':
        in_dir = in_dir[1:-1]
    out_main_dir = config.get("Setup", "out_dir")
    if out_main_dir == "":
        out_main_dir = in_dir
    analysis_flags = json.loads(config.get("Setup", "analysis_flags"))

    # Batch processing of sessions in folder
    if analysis_flags[0]:  # Convert new MEDPC files to neo
        out_dir = os.path.join(out_main_dir, "hdf5")
        in_files = bvmpc.bv_utils.get_all_files_in_dir(in_dir, return_absolute=True)

        # Check if we are using Axona files
        for filename in in_files:
            if os.path.splitext(filename)[-1] == ".inp":
                using_axona = True
                break
            else:
                using_axona = False

        if not using_axona:
            for filename in in_files:
                try:
                    bvmpc.bv_file.convert_to_neo(
                        filename, out_dir, remove_existing=False
                    )
                except Exception as e:
                    bvmpc.bv_utils.log_exception(e, "Error during coversion to neo")
        else:
            for filename in in_files:
                try:
                    if os.path.splitext(filename)[-1] == ".inp":
                        bvmpc.bv_file.convert_axona_to_neo(
                            filename, out_dir, remove_existing=False
                        )
                except Exception as e:
                    bvmpc.bv_utils.log_exception(e, "Error during coversion to neo")

    if analysis_flags[1]:  # plot_batch_sessions
        sub = config.get("BatchPlot", "subjects")
        sub = sub.replace(" ", "").split(",")
        sub = [str(sub_val) for sub_val in sub]

        sub_colors = config.get("BatchPlot", "sub_colors").replace(" ", "").split(",")
        if len(sub_colors) == 0:
            sub_colors_dict = None
        else:
            sub_colors_dict = dict(zip(sub, sub_colors))

        end_date_parsed = config.get("BatchPlot", "end_date")
        if "_" not in end_date_parsed:
            end_date = date.today() + timedelta(days=int(end_date_parsed))
        else:
            Y, M, D = [int(x) for x in end_date_parsed.split("_")]
            end_date = date(Y, M, D)
        start_date_parsed = config.get("BatchPlot", "start_date")
        if "_" not in start_date_parsed:
            start_date = end_date + timedelta(days=int(start_date_parsed))
        else:
            Y, M, D = [int(x) for x in start_date_parsed.split("_")]
            start_date = date(Y, M, D)

        plt_flags = config._sections["BatchPlotOpts"]
        for k, v in plt_flags.items():  # Converts dict variables in .config to int
            plt_flags[k] = int(v)
        bvmpc.bv_batch.plot_batch_sessions(
            out_main_dir, sub, start_date, end_date, plt_flags, sub_colors_dict
        )

    if analysis_flags[2]:
        # TODO turn this into batch if using it
        from bvmpc.bv_session import Session

        h5_loc = r"C:\Users\smartin5\OneDrive - TCDUD.onmicrosoft.com\Claustrum\hdf5\1_08-29-19_16-58_7_RandomisedBlocksExtended_p.nix"
        s = Session(neo_file=h5_loc)
        bvmpc.bv_analyse.trial_clustering(s)


if __name__ == "__main__":
    config_name = "Batch3_Rec.cfg"
    main(config_name)
