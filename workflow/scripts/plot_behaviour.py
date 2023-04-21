from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import simuran as smr


def main(input_df_path, out_dir, config_path):
    config = smr.config_from_file(config_path)
    coherence_df = smr.load_table(input_df_path)


def plot_trial_length(coherence_df, out_dir):
    smr.set_plot_style()
    coherence_df = coherence_df[coherence_df["Trial type"] != "Fail"]

    for t in ["Trial type", "Estimated trial type"]:
        fig, ax = plt.subplots()
        sns.displot(
            data=coherence_df,
            x="Trial length (s)",
            ax=ax,
            hue=t,
            kde=True,
        )
        smr.despine()
        filename = out_dir / f"trial_length_{t}"
        fig = smr.SimuranFigure(fig, filename)
        fig.save()


def plot_trial_press_number(coherence_df, out_dir):
    smr.set_plot_style()
    coherence_df = coherence_df[coherence_df["Trial type"] != "Fail"]

    for t in ["Trial type", "Estimated trial type"]:
        fig, ax = plt.subplots()
        sns.displot(
            data=coherence_df,
            x="Number of lever presses",
            ax=ax,
            hue=t,
            kde=True,
        )
        smr.despine()
        filename = out_dir / f"trial_press_number_{t}"
        fig = smr.SimuranFigure(fig, filename)
        fig.save()

def plot_trial_press_dist(dist_df, out_dir):
    smr.set_plot_style()
    dist_df = dist_df[dist_df["Trial type"] != "Fail"]

    for t in ["Trial type", "Estimated trial type"]:
        fig, ax = plt.subplots()
        sns.displot(
            data=dist_df,
            x="Lever press time (s)",
            ax=ax,
            hue=t,
            kde=True,
        )
        smr.despine()
        filename = out_dir / f"trial_press_dist_{t}"
        fig = smr.SimuranFigure(fig, filename)
        fig.save()

if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input[0],
        Path(snakemake.output[0]).parent.parent,
        snakemake.config["simuran_config"],
    )
