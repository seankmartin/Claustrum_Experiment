from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import simuran as smr


def main(input_df_path, input_df_path2, out_dir, config_path):
    config = smr.config_from_file(config_path)
    coherence_df = smr.load_table(input_df_path)
    coherence_df = coherence_df[coherence_df["Trial type"] != "Fail"]
    coherence_df = coherence_df[coherence_df["Estimated trial type"] != "Fail"]
    press_df = smr.load_table(input_df_path2)
    press_df = press_df[press_df["Trial type"] != "Fail"]
    press_df = press_df[press_df["Estimated trial type"] != "Fail"]
    plot_trial_length(coherence_df, out_dir)
    plot_trial_press_number(coherence_df, out_dir)
    plot_trial_press_dist(press_df, out_dir)


def plot_trial_length(coherence_df, out_dir):
    smr.set_plot_style()

    for t in ["Trial type", "Estimated trial type"]:
        fig, ax = plt.subplots()
        sns.histplot(
            data=coherence_df[coherence_df["Trial length (s)"] < 100],
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
        sns.histplot(
            data=coherence_df[coherence_df["Number of lever presses"] < 10],
            x="Number of lever presses",
            ax=ax,
            hue=t,
        )
        smr.despine()
        filename = out_dir / f"trial_press_number_{t}"
        fig = smr.SimuranFigure(fig, filename)
        fig.save()


def plot_trial_press_dist(dist_df, out_dir):
    smr.set_plot_style()
    dist_df = dist_df[dist_df["Trial type"] != "Fail"]
    dist_df = dist_df[dist_df["Lever press time (s)"] < 100]

    for t in ["Trial type", "Estimated trial type"]:
        fig, ax = plt.subplots()
        sns.histplot(data=dist_df, x="Lever press time (s)", ax=ax, hue=t, kde=True)
        smr.despine()
        filename = out_dir / f"trial_press_dist_{t}"
        fig = smr.SimuranFigure(fig, filename)
        fig.save()


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input[0],
        snakemake.input[1],
        Path(snakemake.output[0]),
        snakemake.config["simuran_config"],
    )
