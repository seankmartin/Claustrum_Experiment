"""Plotting functions for the bvmpc module."""

import os
import colorsys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import OrderedDict
import seaborn as sns


class GridFig:
    """Handles gridded figures."""

    def __init__(
            self, rows, cols=4,
            size_multiplier_x=5, size_multiplier_y=5, wspace=0.3, hspace=0.3,
            tight_layout=False):
        """
        Set up the grid specifications.

        size_multiplier, wspace, and hspace are used for spacing

        """
        self.fig = plt.figure(
            figsize=(cols * size_multiplier_x,
                     rows * size_multiplier_y),
            tight_layout=tight_layout)
        self.gs = gridspec.GridSpec(rows, cols, wspace=wspace, hspace=hspace)
        self.idx = 0
        self.rows = rows
        self.cols = cols

    def get_ax(self, row_idx, col_idx):
        """Add subplot with standard 1x1 gs -> returns ax."""
        ax = self.fig.add_subplot(self.gs[row_idx, col_idx])
        return ax

    def get_multi_ax(self, row_start, row_end, col_start, col_end):
        """Add subplot with custom gs sizes -> returns ax."""
        ax = self.fig.add_subplot(
            self.gs[row_start:row_end, col_start:col_end])
        plt.subplots_adjust(top=0.85)
        return ax

    def save_fig(self, df_date, df_sub, df_stage, plot_type, out_dir, j=None):
        """
        Names and saves figure.

        """
        date_p = sorted(set(df_date))
        sub_p = sorted(set(df_sub))
        stage_p = sorted(set(df_stage))
        out_name = plot_type + \
            str(date_p) + '_' + str(sub_p) + \
            '_' + str(stage_p)
        if j is not None:
            out_name += '_' + str(j)
        out_name += ".png"
        print("Saved figure to {}".format(
            os.path.join(out_dir, out_name)))
        self.fig.savefig(os.path.join(out_dir, out_name), dpi=400)
        plt.close()
        return

    def get_fig(self):
        """Return the figure object in this class."""
        return self.fig

    def get_next(self, along_rows=True):
        """
        Get next index along rows or columns.

        along rows:
        1   2   3   4   5   6
        7   8   9   10  11  12  ...

        else:
        1   3   5   7   9   11
        2   4   6   8   10  12  ...

        """
        if along_rows:
            row_idx = (self.idx // self.cols)
            col_idx = (self.idx % self.cols)

        else:
            row_idx = (self.idx % self.rows)
            col_idx = (self.idx // self.rows)

        # print(row_idx, col_idx)
        ax = self.get_ax(row_idx, col_idx)
        self._increment()
        return ax

    def get_next_snake(self):
        """
        Get the next index in a snake like pattern.

        1   2   5   6   9   10  ...
        3   4   7   8   11  12  ..

        """
        if self.rows != 2:
            print("ERROR: Can't get snake unless there are two rows")
        i = self.idx
        row_idx = (i // 2) % 2
        col_idx = (i // 2) + (i % 2) - row_idx
        ax = self.get_ax(row_idx, col_idx)
        self._increment()
        return ax

    def _increment(self):
        """Private function to increase the internal idx counter."""
        self.idx = self.idx + 1
        if self.idx == self.rows * self.cols:
            # print("Looping")
            self.idx = 0


class GroupManager:

    def __init__(self, group_list):
        self.group_list = group_list
        # print(self.group_list)
        self.info_dict = OrderedDict()
        self.index = 0
        self.color_list = [
            "Blues", "Oranges", "Greens", "Purples", "PuRd", "Greys"]
        set_vals = sorted(set(group_list), key=group_list.index)
        # print(set_vals)
        import numpy as np
        if len(set_vals) > len(self.color_list):
            start_vals = np.arange(0.0, 2.5, 2.5 / (len(set_vals) - 0.99))
            for set_v, start_v in zip(set_vals, start_vals):
                self.info_dict[set_v] = ColorManager(
                    group_list.count(set_v), "sns_helix", start=start_v)
        else:
            start_vals = self.color_list[:len(set_vals)]
            for set_v, start_v in zip(set_vals, start_vals):
                self.info_dict[set_v] = ColorManager(
                    group_list.count(set_v), "sns", sns_style=start_v)
        # print(self.info_dict)

    def get_next_color(self):
        out = self.info_dict[self.group_list[self.index]].get_next_color()
        self._increment()
        return out

    def _increment(self):
        self.index += 1
        if self.index == len(self.group_list):
            self.index = 0

    def test_plot(self):
        from scipy.stats import norm
        import numpy as np
        fig, ax = plt.subplots()
        x_axis = np.arange(-15, 5, 0.001)
        std_devs = np.arange(0.8, 3, 2.20 / len(self.group_list))
        for sd in std_devs:
            ax.plot(
                x_axis, norm.pdf(x_axis, -sd * 2, sd),
                color=self.get_next_color())
        sns.despine(top=True, bottom=True, right=True, left=True)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.savefig("test.png", dpi=400)


class ColorManager:
    def __init__(self, num_colors, method="sns", **kwargs):
        self.num_colors = num_colors
        if method == "sns":
            sns_style = kwargs.get("sns_style", None)
            self.create_sns_palette(s_type=sns_style)
        elif method == "rgb":
            self.create_rgb()
        elif method == "sns_helix":
            start = kwargs.get("start", 0)
            self.create_sns_helix(start)
        else:
            raise ValueError("{} not recognised method".format(
                method))
        self.idx = 0

    def create_rgb(self):
        N = self.num_colors
        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        self.colors = list(RGB_tuples)

    def create_sns_palette(self, s_type=None):
        self.colors = sns.color_palette(s_type, self.num_colors)

    def create_sns_helix(self, start):
        self.colors = sns.cubehelix_palette(self.num_colors, start)

    def get_next_color(self):
        result = self.colors[self.idx]
        self._increment()
        return result

    def get_color(self, index):
        return self.colors[index]

    def test_plot(self):
        from scipy.stats import norm
        import numpy as np
        fig, ax = plt.subplots()
        x_axis = np.arange(-15, 5, 0.001)
        std_devs = np.arange(0.8, 3, 2.20 / self.num_colors)
        for sd in std_devs:
            ax.plot(
                x_axis, norm.pdf(x_axis, -sd * 2, sd),
                color=self.get_next_color())
        sns.despine(top=True, bottom=True, right=True, left=True)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.savefig("test.png", dpi=400)

    def _increment(self):
        """Private function to increase the internal idx counter."""
        self.idx = self.idx + 1
        if self.idx == self.num_colors:
            self.idx = 0


def behav_vlines(ax, s, behav_plot, lw=1):
    """
    Plots vlines based on desired behaviour related timestamps

    Parameters
    ----------
    ax : plt.axe, default None
        ax object to plot into.
    s: object
        Behavioural session object
    behav_plot: list[bool]
        bool list denoting timestamps to plot.
            # 0 - plot levers
            # 1 - plot rewards
            # 2 - plot pellets
            # 3 - plot double pells
    lw: float
        Sets linewidth of vlines

    Returns
    -------
    ax : plt.axe, default None
    legends: list[handles]
        list of legend handles

    """
    from matplotlib import lines

    legends = []
    rw_ts = s.get_rw_ts()
    lev_ts = s.get_lever_ts()
    pell_exd_ts, d_pell_ts = s.split_pell_ts()
    if behav_plot[0]:
        for lev in lev_ts:  # vline demarcating lev presses
            ax.axvline(lev, linestyle='-',
                       color='blue', linewidth=lw/2)
        label = lines.Line2D([], [], color='blue', marker='|', linestyle='None',
                             markersize=10, markeredgewidth=lw, label='Lever Press')
        legends.append(label)
    if behav_plot[1]:
        for rw in rw_ts:    # vline demarcating reward point/end of trial
            ax.axvline(rw, linestyle='-',
                       color='orange', linewidth=lw)
        label = lines.Line2D([], [], color='orange', marker='|', linestyle='None',
                             markersize=10, markeredgewidth=lw, label='Reward')
        legends.append(label)
    if behav_plot[2]:
        for pell in pell_exd_ts:  # vline demarcating pells
            ax.axvline(pell, linestyle='-',
                       color='w', linewidth=lw)
        label = lines.Line2D([], [], color='w', marker='|', linestyle='None',
                             markersize=10, markeredgewidth=lw, label='Pell')
        legends.append(label)
    if behav_plot[3]:
        for d in d_pell_ts:  # vline demarcating double pells
            ax.axvline(d, linestyle='-',
                       color='magenta', linewidth=lw)
        label = lines.Line2D([], [], color='magenta', marker='|', linestyle='None',
                             markersize=10, markeredgewidth=lw, label='Double Pell')
        legends.append(label)
    return ax, legends


def savefig(fig, name):
    """
    Saves figure using the custom settings:
        dpi=400 
        bbox_inches='tight'
        pad_inches=0.5
    """
    print("Saving result to {}".format(name))
    fig.savefig(name, dpi=400, bbox_inches='tight', pad_inches=0.5)
