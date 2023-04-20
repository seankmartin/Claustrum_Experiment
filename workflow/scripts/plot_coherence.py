from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import simuran as smr
from skm_pyutils.table import df_from_file, list_to_df


def convert_to_long_form(df):
    new_res = []
    columns = df.columns
    for i, row in df.iterrows():
        coherence = row["Coherence"]
        freqs = row["Frequency (Hz)"]
        for f, c in zip(freqs, coherence):
            new_res.append(f)
            new_res.append(c)
            for col in columns:
                if col not in ["Coherence", "Frequency (Hz)"]:
                    new_res.append(row[col])

    headers = ["Frequency (Hz)", "Coherence"] + [
        c for c in columns if c not in ["Coherence", "Frequency (Hz)"]
    ]
    new_df = list_to_df(new_res, headers)
    return new_df


def fix_notch_freqs(df, freqs_to_fix):
    fnames = sorted(list(set(df["Fname"])))
    for fname in fnames:
        fname_bit = df["Fname"] == fname
        df_bit = df[fname_bit]
        if len(df_bit) == 0:
            continue
        for f in freqs_to_fix:
            start_val = (
                df_bit[df_bit["Frequency (Hz)"].between(f - 5, f - 4)]["Coherence"]
                .iloc[0]
                .data
            )
            end_val = (
                df_bit[df_bit["Frequency (Hz)"].between(f + 4, f + 5)]["Coherence"]
                .iloc[-1]
                .data
            )
            freqs = df_bit["Frequency (Hz)"].between(f - 5, f + 5)
            interp = np.linspace(
                start_val, end_val, np.count_nonzero(freqs), endpoint=True
            )
            df.loc[freqs & fname_bit, "Coherence"] = interp


def plot_coherence(df, out_dir, max_frequency=40):
    smr.set_plot_style()

    fig, ax = plt.subplots()
    sns.lineplot(
        data=df[df["Frequency (Hz)"] <= max_frequency],
        x="Frequency (Hz)",
        y="Coherence",
        style="Trial Type",
        hue="Brain regions",
        # estimator="median",
        estimator="mean",
        errorbar=("ci", 95),
        n_boot=10000,
        ax=ax,
    )

    plt.ylim(0, 1)
    smr.despine()
    filename = out_dir / "coherence"
    fig = smr.SimuranFigure(fig, filename)
    fig.save()


def plot_band_coherence(input_df, output_dir):
    smr.set_plot_style()

    new_list = []
    for i, row in input_df.iterrows():
        for val in ["Delta", "Theta", "Beta", "Low Gamma", "High Gamma"]:
            new_list.append(row[val], row["Brain regions"], row["Trial type"], val)
    headers = ["Coherence", "Brain regions", "Trial type", "Band"]
    new_df = list_to_df(new_list, headers)

    for br in ["CLA RSC", "CLA ACC", "RSC ACC"]:
        sub_df = new_df[new_df["Brain regions"] == br]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.boxplot(
            data=sub_df,
            y="Coherence",
            x="Band",
            hue="Trial type",
            palette="pastel",
            ax=ax,
            showfliers=False,
            width=0.9,
            # palette=["0.65", "r"],
        )
        sns.stripplot(
            data=sub_df,
            y="Coherence",
            x="Band",
            hue="Trial type",
            ax=ax,
            # palette="dark:grey",
            palette=["0.4", "0.75"],
            alpha=0.95,
            dodge=True,
            edgecolor="k",
            linewidth=1,
            size=4.5,
            legend=False,
        )
        smr.despine()
        smr_fig = smr.SimuranFigure(fig, output_dir / f"coherence_{br}_")
        smr_fig.save()


def main(input_df_path, out_dir, config_path):
    config = smr.config_from_file(config_path)
    coherence_df = df_from_file(input_df_path)
    coherence_df = convert_to_long_form(coherence_df)
    fix_notch_freqs(coherence_df, config["notch_freqs"])
    plot_coherence(coherence_df, out_dir, config["max_psd_freq"])


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input[0],
        Path(snakemake.output[0]).parent.parent,
        snakemake.config["simuran_config"],
    )