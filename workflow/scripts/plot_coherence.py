import itertools
from pathlib import Path
import ast
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import simuran as smr
from skm_pyutils.table import df_from_file, list_to_df

TO_PLOT = "Trial type"


def convert_to_long_form(df, max_frequency=40):
    new_res = []
    columns = df.columns
    for i, row in df.iterrows():
        coherence = row["Coherence"]
        coherence = re.sub(" +", ",", coherence[1:-1].strip())
        coherence = f"[{coherence}]"
        coherence = np.array(ast.literal_eval(coherence))
        freqs = row["Frequency (Hz)"]
        freqs = re.sub(" +", ",", freqs[1:-1].strip())
        freqs = f"[{freqs}]"
        freqs = np.array(ast.literal_eval(freqs))
        for f, c in zip(freqs[freqs < max_frequency], coherence[freqs < max_frequency]):
            res2 = []
            res2.append(f)
            res2.append(c)
            for col in columns:
                if col in [
                    "Trial type",
                    "Estimated trial type",
                    "Brain regions",
                    "Source file",
                ]:
                    res2.append(row[col])
            new_res.append(res2)

    headers = [
        "Frequency (Hz)",
        "Coherence",
        "Brain regions",
        "Trial type",
        "Estimated trial type",
        "Source file",
    ]
    new_df = list_to_df(new_res, headers)
    new_df = new_df[new_df["Trial type"] != "Fail"]
    new_df = new_df[new_df["Estimated trial type"] != "Fail"]
    return new_df


def fix_notch_freqs(df, freqs_to_fix):
    fnames = sorted(list(set(df["Source file"])))
    for fname in fnames:
        fname_bit = df["Source file"] == fname
        file_df = df[fname_bit]
        if len(file_df) == 0:
            continue
        regions = list(file_df["Brain regions"].unique())
        for region_pair, f, tt in itertools.product(regions, freqs_to_fix, ["FI", "FR"]):
            region_bit = file_df["Brain regions"] == region_pair
            df_bit = file_df[region_bit]
            trial_bit = df_bit["Trial type"] == tt
            df_bit = df_bit[trial_bit]
            start_val = np.mean(
                df_bit[df_bit["Frequency (Hz)"].between(f - 3, f - 2)]["Coherence"]
            )
            end_val = np.mean(
                df_bit[df_bit["Frequency (Hz)"].between(f + 2, f + 3)]["Coherence"]
            )
            freqs = df_bit["Frequency (Hz)"].between(f - 3, f + 3)
            s = len(df.loc[freqs & fname_bit & region_bit & trial_bit, "Coherence"])
            interp = np.linspace(
                start_val, end_val, s, endpoint=True
            ) + np.random.normal(0, 0.05, s)
            df.loc[freqs & fname_bit & region_bit & trial_bit, "Coherence"] = interp


def plot_coherence(df, out_dir, max_frequency=40):
    smr.set_plot_style()

    df = df[df["Frequency (Hz)"] <= max_frequency]
    regions = df["Brain regions"].unique()
    for r in regions:
        fig, ax = plt.subplots()
        sns.lineplot(
            data=df[df["Brain regions"] == r],
            x="Frequency (Hz)",
            y="Coherence",
            style=TO_PLOT,
            hue=TO_PLOT,
            estimator="median",
            # estimator="mean",
            errorbar=("ci", 95),
            ax=ax,
        )

        smr.despine()
        filename = out_dir / f"coherence_{r}"
        fig = smr.SimuranFigure(fig, filename)
        fig.save()

        fig, ax = plt.subplots()
        sns.lineplot(
            data=df[df["Brain regions"] == r],
            x="Frequency (Hz)",
            y="Coherence",
            style="Estimated trial type",
            hue="Estimated trial type",
            estimator="median",
            # estimator="mean",
            errorbar=("ci", 95),
            ax=ax,
        )

        smr.despine()
        filename = out_dir / f"coherence_estimated_{r}"
        fig = smr.SimuranFigure(fig, filename)
        fig.save()


def plot_band_coherence(input_df, output_dir):
    smr.set_plot_style()

    new_list = []
    for i, row in input_df.iterrows():
        for val in ["Delta", "Theta", "Beta", "Low gamma", "High gamma"]:
            new_list.append(
                [
                    row[val],
                    row["Brain regions"],
                    row["Trial type"],
                    val,
                    row["Estimated trial type"],
                ]
            )
    headers = [
        "Coherence",
        "Brain regions",
        "Trial type",
        "Band",
        "Estimated trial type",
    ]
    new_df = list_to_df(new_list, headers)

    for br in ["CLA RSC", "CLA ACC", "RSC ACC"]:
        sub_df = new_df[new_df["Brain regions"] == br]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.boxplot(
            data=sub_df,
            y="Coherence",
            x="Band",
            hue=TO_PLOT,
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
            hue=TO_PLOT,
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
    long_df = convert_to_long_form(coherence_df, config["max_psd_freq"])
    fix_notch_freqs(long_df, config["notch_freqs"])
    plot_coherence(long_df, out_dir, config["max_psd_freq"])
    plot_band_coherence(coherence_df, out_dir)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input[0],
        Path(snakemake.output[0]),
        snakemake.config["simuran_config"],
    )
