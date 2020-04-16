"""This module holds functions related to running batch operations."""
import os
import numpy as np
import pandas as pd
import math

import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from bvmpc.bv_file import extract_sessions
import bvmpc.bv_analyse as bv_an
import bvmpc.bv_plot as bv_plot
import bvmpc.bv_utils


def struc_timeline(sub_list, in_dir):
    """ Structure sessions into a pandas dataframe based on trials
    Returns 2 outputs: grp_timeline_df, time_df_sub
        grp_timeline_df - array of pandas dataframe across time based on session
        grp_timeline_df_sub     - list denoting subject corresponding to each df
    """
    # Initialize group variables
    grp_timeline_df = []
    grp_timeline_df_sub = []

    for sub in sub_list:
        s_grp = extract_sessions(in_dir, sub)
        # Initialize column arrays
        d_list, t_list, sub_list, s_idx_list, r_list, s_list = [], [], [], [], [], []
        err_FR_list, err_FI_list = [], []
        rw_FR_list, rw_FI_list, dr_list = [], [], []
        ratio_list, interval_list = [], []
        sch_list, sch_rw_list, sch_err_list, sch_dr_list = [], [], [], []

        for i, s in enumerate(s_grp):
            subject = s.get_metadata('subject')
            date = s.get_metadata("start_date").replace("/", "-")[:5]
            session_type = s.get_metadata('name')
            time = s.get_metadata('start_time')
            stage = session_type[:2].replace(
                '_', '')  # Obtain stage number w/o _
            timestamps = s.get_arrays()
            pell_ts = timestamps["Reward"]
            dpell_bool = np.diff(pell_ts) < 0.5
            dpell_idx = np.nonzero(dpell_bool)[0]
            reward_times = s.get_rw_ts()

            # Initialize variables used per session
            ratio, interval, sch, sch_err, sch_rw, sch_dr = [], [], [], [], [], []
            err_FI, err_FR, rw_FI, rw_FR = [], [], [], []

            ratio = s.get_ratio()
            interval = s.get_interval()
            # Stage specific variables
            if stage == '5a':
                s_name = 'R' + str(ratio)
                ratio = s_name
                rw_FR = len(reward_times)
            elif stage == '5b':
                s_name = 'I' + str(interval)
                interval = s_name
                rw_FI = len(reward_times)
            else:
                s_name = stage.replace('_', '').replace('2', 'M').replace(
                    '3', 'Lh').replace('4', 'Lt').replace(
                    '6', 'B1').replace('7', 'B2')
            if 'B' in s_name:
                rw_FI, rw_FR = 0, 0
                ratio = 'R' + str(ratio)
                interval = 'I' + str(interval)
                norm_r_ts, _, norm_err_ts, norm_dr_ts, _ = s.split_sess(
                    all_levers=True)
                sch_type = s.get_arrays('Trial Type')
                # Error related variables
                if s_name == 'B2':
                    err_FI, err_FR = 0, 0
                    for i, err in enumerate(norm_err_ts):
                        if sch_type[i] == 1:
                            err_FR = + len(err)
                        elif sch_type[i] == 0:
                            err_FI += len(err)
                        sch_err.append(len(err))
                else:
                    err_FR = []
                    err_FI = []

                # Reward related variables
                for i, (rw, dr) in enumerate(zip(norm_r_ts, norm_dr_ts)):
                    if sch_type[i] == 1:
                        rw_FR += len(rw)
                        sch.append('FR')
                        sch_dr.append([])
                    elif sch_type[i] == 0:
                        rw_FI += len(rw)
                        sch.append('FI')
                        sch_dr.append(len(dr))
                    sch_rw.append(len(rw))

            # Update list arrays with new session
            d_list.append(date)
            t_list.append(time)
            s_idx_list.append('S' + stage[0])
            s_list.append(s_name)
            r_list.append(len(pell_ts))
            dr_list.append(len(dpell_idx))
            interval_list.append(interval)
            ratio_list.append(ratio)
            sch_list.append(sch)
            sch_rw_list.append(sch_rw)
            sch_err_list.append(sch_err)
            sch_dr_list.append(sch_dr)
            rw_FI_list.append(rw_FI)
            rw_FR_list.append(rw_FR)
            err_FI_list.append(err_FI)
            err_FR_list.append(err_FR)
        sub_list = np.full(len(d_list), sub, dtype=int)

        timeline_dict = {
            'Subject': sub_list,
            'Date': d_list,
            'Start Time': t_list,
            'Stage Idx': s_idx_list,  # eg. S2, S5, S7 ...
            'Stage': s_list,  # eg. M, Lt, R8, I30, B1
            'Total Rewards': r_list,
            'Double Rewards': dr_list,
            'Interval': interval_list,
            'Ratio': ratio_list,
            'Schedule Blocks': sch_list,
            'Schedule Rw': sch_rw_list,
            'Schedule Err': sch_err_list,
            'Schedule DRw': sch_dr_list,
            'FI Corr': rw_FI_list,
            'FR Corr': rw_FR_list,
            'FI Err': err_FI_list,
            'FR Err': err_FR_list
        }
        timeline_df = pd.DataFrame(timeline_dict)
        grp_timeline_df.append(timeline_df)
        grp_timeline_df_sub.append(subject)

    return grp_timeline_df, grp_timeline_df_sub


def plot_batch_sessions(start_dir, sub_list, start_date, end_date, plt_flags, sub_colors_dict):
    out_dir = os.path.join(start_dir, "Plots", "Current")
    bvmpc.bv_utils.make_dir_if_not_exists(out_dir)

    if plt_flags["raster"] == 1 or plt_flags["hist"] == 1:
        # Path join only present in plot_sessions
        in_dir = os.path.join(start_dir, "hdf5")

        # Default conversion of date based on start_date and end_date range
        d = []
        for single_date in bvmpc.bv_utils.daterange(start_date, end_date):
            d.append(single_date.isoformat()[-5:])

        print('Extracting sessions on {} for subjects {}...'.format(d, sub_list))
        s_grp = extract_sessions(in_dir, sub_list, d_list=d)

        # splits s_grp into groups of 4
        plot_set = bvmpc.bv_utils.chunks(s_grp, 4)

        for j, plot_grp in enumerate(plot_set):
            # Figure Initialization
            n = len(plot_grp)
            if n > 4:
                print('Too many plots')
                quit()
            elif n > 2:
                rows, cols = [4, 4 * math.ceil(n / 2)]
            else:
                rows, cols = [2 * n, 4 * math.ceil(n / 2)]

            # Initializes GridFig Obj
            gf = bv_plot.GridFig(rows, cols, wspace=0.5, hspace=0.5)
            # fig = gf.get_fig()
            # size_multiplier = 5
            # fig = plt.figure(
            #     figsize=(cols * size_multiplier, rows * size_multiplier),
            #     tight_layout=False)
            # gs = gridspec.GridSpec(rows, cols, wspace=0.5, hspace=0.5)

            df_sub, df_date, df_stage = [], [], []  # Initialize plot naming parameters

            plotting_sub = []
            for i, s in enumerate(plot_grp):  # Iterate through groups to plot rasters
                # plot naming parameters
                df_sub.append(s.get_metadata('subject'))
                df_date.append(s.get_metadata(
                    'start_date').replace('/', '_')[:5])
                df_stage.append(s.get_stage())

                # 2x2 plotting axes
                k = (i % 2) * 2
                if plt_flags["raster"] == 1:
                    ax = gf.get_multi_ax(
                        k, k + 2, 4 * int(i / 2), 4 * math.ceil((i + 1) / 2))
                    bv_an.plot_raster_trials(
                        s, ax, sub_colors_dict, align=[0, 0, 1, 0])
                    plot_type = 'Raster_'  # Plot name for saving

                if plt_flags["hist"] == 1:
                    if not plotting_sub == s.get_metadata('subject'):
                        ax = gf.get_multi_ax(
                            k, k + 2, 4 * int(i / 2), 4 * math.ceil((i + 1) / 2))
                        plotting_sub = s.get_metadata('subject')
                        loop = 1
                    bv_an.trial_length_hist(s, ax, loop, sub_colors_dict)
                    plot_type = 'Hist_'  # Plot name for saving
                    loop += 1

            # Save Figure
            # plt.subplots_adjust(top=0.85)
            # fig.suptitle(('Subject ' + subject + ' Performance'),
            #                 color=mycolors(subject), fontsize=30)

            # date_p = sorted(set(df_date))
            # sub_p = sorted(set(df_sub))
            # stage_p = sorted(set(df_stage))
            # out_name = plot_type + str(date_p) + '_' + str(sub_p) + '_' + str(stage_p) + '_' + str(j)
            # out_name += ".png"
            # print("Saved figure to {}".format(
            #     os.path.join(out_dir, out_name)))
            # bv_plot.savefig(fig, os.path.join(out_dir, out_name))
            # plt.close()
            gf.save_fig(df_date, df_sub, df_stage, plot_type, out_dir, j)

    # plot cumulative response graphs
    if plt_flags["summary"] == 1:
        for single_date in bvmpc.bv_utils.daterange(start_date, end_date):
            d = [single_date.isoformat()[-5:]]
            # plot_sessions(start_dir, d, sub, summary=True, single=True,
            #               corr_only=True, scd=sub_colors_dict)  # Single animal breakdown
            # Group with corr_only breakdown
            plot_sessions(start_dir, d, sub_list, summary=True,
                          single=False, corr_only=True, scd=sub_colors_dict)
            # plot_sessions(start_dir, d, sub, summary=True, single=False, corr_only=False, scd=sub_colors_dict)  # Group with complete breakdown

        # plot all 4 timeline types
    if plt_flags["timeline"] == 1:
        print("Plotting Timeline...")
        d = [end_date.isoformat()[-5:]]
        in_dir = os.path.join(start_dir, "hdf5")
        single = False  # plots seperate graphs for each animal if True
        show_date = True  # Sets x-axis as dates if True
        details = False
        recent = False
        if not single:
            plot_limit = 4
            sub_list = bvmpc.bv_utils.split_list(sub_list, plot_limit)
            for l in sub_list:
                timeline_plot(l, in_dir, out_dir, single_plot=single, det_err=False, det_corr=False,
                              recent=recent, show_date=show_date, details=details, sub_colors_dict=sub_colors_dict)
        else:
            # Plots timeline for specified subjects
            timeline_plot(sub_list, in_dir, out_dir, single_plot=single, det_err=False, det_corr=False,
                          recent=recent, show_date=show_date, details=details, sub_colors_dict=sub_colors_dict)


def plot_sessions(
        start_dir, d_list, sub_list,
        summary=False, single=False, timeline=False,
        details=False, det_err=False, det_corr=False,
        recent=False, show_date=False, int_only=False,
        corr_only=False, scd=None):  # TODO Split timeline and plotting into seperate functions
    ''' Plots session summaries
    summary : bool, False
        Optional. Plots all sessions in a single plot, up to 6
    single : bool, False
        Optional. Plots single session summaries with breakdown of single blocks
    int_only : bool, False
        Optional. Plots only interval trials in zoomed schedule plot
    corr_only : bool, False
        Optional. Plots seperate summary plot with correct only trials
    scd : dict, None
        dict of Sub : color
        Used to assign color to plot title.
    '''
    s_list = ['4', '5a', '5b', '6', '7']

    in_dir = os.path.join(start_dir, "hdf5")
    out_dir = os.path.join(start_dir, "Plots", "Current")
    bvmpc.bv_utils.make_dir_if_not_exists(out_dir)

    if summary and not corr_only:
        #  extracts hdf5 session based on specification
        max_plot = 4  # Set max plots per figure
        s_grp = extract_sessions(in_dir, sub_list, s_list, d_list)
        if s_grp == []:
            return print("***No Files Extracted***")
        idx = 0
        if len(s_grp) > max_plot:
            j = 0
            s_grp_split = []
            s_grp_idx = np.arange(len(s_grp))
            for i in s_grp_idx[max_plot - 1::max_plot]:
                s_grp_split.append(s_grp[j:i + 1])
                j = i + 1

            mv = len(s_grp) % max_plot
            if mv != 0:
                s_grp_split.append(s_grp[-mv:])
            for s_grp in s_grp_split:
                idx += 1
                sum_plot(s_grp, idx, out_dir)
        else:
            sum_plot(s_grp, idx, out_dir)

    if summary and corr_only:
        # plots corr_only plots
        max_plot = 4  # Set max plots per figure
        s_grp = extract_sessions(in_dir, sub_list, s_list, d_list)
        if s_grp == []:
            return print("***No Files Extracted***")

        idx = 0
        if len(s_grp) > max_plot:
            j = 0
            s_grp_split = []
            s_grp_idx = np.arange(len(s_grp))
            for i in s_grp_idx[max_plot - 1::max_plot]:
                s_grp_split.append(s_grp[j:i + 1])
                j = i + 1

            mv = len(s_grp) % max_plot
            if mv != 0:
                s_grp_split.append(s_grp[-mv:])
            for s_grp in s_grp_split:
                idx += 1
                sum_plot(s_grp, idx, out_dir, corr_only=True)
        else:
            sum_plot(s_grp, idx, out_dir, corr_only=True)

    if single and summary:
        # Single Subject Plots
        idx = 0
        for sub in sub_list:
            s_grp = extract_sessions(in_dir, sub, s_list, d_list)
            if s_grp == []:
                return print("***No Files Extracted***")
            s_passed = []
            d_passed = []
            for s in s_grp:
                stage = s.get_metadata('name')[:2].replace('_', '')
                s_passed.append(stage)
                date = s.get_metadata("start_date").replace("/", "-")
                d_passed.append(date[:5])
                subject = s.get_metadata('subject')
            if '7' in s_passed:
                gf = bv_plot.GridFig(len(s_grp), 4)
                fig = gf.get_fig()
                if len(s_grp) == 1:
                    fig.tight_layout(rect=[0, 0.03, 0.8, 0.95])
                for i, s in enumerate(s_grp):
                    ax1 = gf.get_next()
                    bv_an.cumplot(s, out_dir, ax1, int_only,
                                  zoom=False, zoom_sch=False)
                    ax2 = gf.get_next()
                    bv_an.cumplot(s, out_dir, ax2, int_only, zoom=False, zoom_sch=True,
                                  plot_error=False, plot_all=True)
                    ax3 = gf.get_next()
                    bv_an.cumplot(s, out_dir, ax3, int_only, zoom=False, zoom_sch=True,
                                  plot_error=False, plot_all=False)
                    ax4 = gf.get_next()
                    bv_an.cumplot(s, out_dir, ax4, int_only, zoom=False, zoom_sch=True,
                                  plot_error=True, plot_all=False)
                plt.subplots_adjust(top=0.85)
                if scd == None:
                    color = "k"
                else:
                    color = bvmpc.bv_utils.mycolors(sub, scd)
                fig.suptitle(('Subject ' + subject + ' Performance'),
                             color=color, fontsize=30)

                # # Seperate plots w line
                # ax1.hlines(1.13, -0, 4.9, clip_on=False,
                #             transform=ax1.transAxes, linewidth=0.7)
                s_print = np.array_str(np.unique(np.array(s_passed)))
                d_print = np.array_str(np.unique(np.array(d_passed)))
                out_name = "Sum_" + subject + "_" + d_print + "_" + s_print + ".png"
                bv_plot.savefig(fig, os.path.join(out_dir, out_name))
            else:
                sum_plot(s_grp, idx, out_dir, single=single)


def sum_plot(s_grp, idx, out_dir, zoom=True, single=False,
             int_only=False, corr_only=False):
    """ zoom:   if True, divides session into blocks and plots each block as individual lines.

    """
    # Plots summary of day
    if zoom:
        print('zoomed')
        if len(s_grp) > 2:
            cols = 2 * math.ceil(len(s_grp) / 2)
            rows = 2
        else:
            rows = len(s_grp)
            cols = 2
    else:
        if len(s_grp) > 4:
            rows = math.ceil(len(s_grp) / 4)
            cols = 4
        else:
            cols = len(s_grp)
            rows = 1
    gf = bv_plot.GridFig(rows, cols)
    fig = gf.get_fig()
    s_passed = []
    d_passed = []

    for _, s in enumerate(s_grp):
        subject = s.get_metadata('subject')
        stage = s.get_metadata('name')[:2].replace('_', '')
        date = s.get_metadata("start_date").replace("/", "-")
        s_passed.append(stage)
        d_passed.append(date[:5])

        if zoom:
            ax1 = gf.get_next_snake()
        else:
            ax1 = gf.get_next()

        if corr_only and stage == '7':
            bv_an.cumplot(s, out_dir, ax1, int_only, zoom=False,
                          zoom_sch=False, plot_all=False)
        else:
            bv_an.cumplot(s, out_dir, ax1, int_only, zoom=False,
                          zoom_sch=False)

        if stage == '2' or stage == '3' or stage == '4':
            IRT = True
        elif stage == '5a' or stage == '5b':
            IRT = True  # Change to False for zoomed plot instead of IRT
        else:
            IRT = False

        if IRT:
            ax2 = gf.get_next_snake()
            bv_an.IRT(s, out_dir, ax2)
        elif zoom:
            ax2 = gf.get_next_snake()
            if corr_only and stage == '7':
                bv_an.cumplot(s, out_dir, ax2, int_only, zoom=False, zoom_sch=True,
                              plot_error=False, plot_all=False)
            else:
                bv_an.cumplot(s, out_dir, ax2, int_only, zoom=False, zoom_sch=True,
                              plot_error=False, plot_all=True)
        plt.subplots_adjust(top=0.85)
    d_print = np.array_str(np.unique(np.array(d_passed)))
    d_title = np.array2string(np.unique(np.array(d_passed)))
    s_print = np.array_str(np.unique(np.array(s_passed)))

    if single:
        fig.suptitle(('Subject ' + subject + ' Performance'), fontsize=30)
        out_name = "Sum_" + subject + "_" + d_print + "_" + s_print + ".png"
    elif corr_only and stage == '7':
        if idx == 0:
            fig.suptitle(('Summary across animals ' + d_title +
                          '_Correct Only'), fontsize=30)
            out_name = "Sum_" + d_print + "_" + s_print + "_Corr.png"
        else:
            fig.suptitle(('Summary across animals ' + d_title +
                          '_Correct Only' + " p" + str(idx)), fontsize=30)
            out_name = "Sum_" + d_print + "_" + \
                s_print + "_" + str(idx) + "_Corr.png"
    else:
        if idx == 0:
            fig.suptitle(('Summary across animals ' + d_title), fontsize=30)
            out_name = "Sum_" + d_print + "_" + s_print + ".png"
        else:
            fig.suptitle(('Summary across animals ' +
                          d_title + " p" + str(idx)), fontsize=30)
            out_name = "Sum_" + d_print + "_" + \
                s_print + "_" + str(idx) + ".png"
    print("Saved figure to {}".format(
        os.path.join(out_dir, out_name)))
    bv_plot.savefig(fig, os.path.join(out_dir, out_name))
    plt.close()


def timeline_plot(
        sub_list, in_dir, out_dir, single_plot=False,
        det_err=False, det_corr=False, recent=False,
        show_date=True, details=False, sub_colors_dict=None):
    """

    Plots total rewards from beginining of first session  

    Arguments:
    det_err - plots error lever presses
    det_cor - plots correct lever presses
    recent - plots recent (determined in code - static) datapoints only
    show_date - plots date as x axis instead of session type
    """
    # Plot size
    rows, cols = [len(sub_list), 4]
    size_multiplier = 5
    fig = plt.figure(
        figsize=(cols * size_multiplier, rows * size_multiplier),
        tight_layout=False)
    gs = gridspec.GridSpec(rows, cols, wspace=0.4, hspace=0.5)

    for c, sub in enumerate(sub_list):
        # Plot total pellets across sessions
        s_grp = extract_sessions(in_dir, [sub])
        s_list, r_list, type_list, d_list, box_list, time_list = [], [], [], [], [], []
        err_FR_list, err_FI_list = [], []
        rw_FR_list, rw_FI_list, rw_double_list = [], [], []
        changes, stage_change, dpell_change = [], [], []
        change_idx = []
        prev_ratio, prev_interval, c_ratio, c_interval = [], [], [], []
        dpell_old = []
        prev_name = '2'
        if recent:
            number_sessions_ago = -31  # change value to set number of sessions ago
            s_grp = s_grp[number_sessions_ago:]
        else:
            pass

        for i, s in enumerate(s_grp):
            ratio = s.get_ratio()
            interval = s.get_interval()
            s_type = s.get_metadata('name')[:2]
            timestamps = s.get_arrays()

            date = s.get_metadata('start_date')  # Display mm/dd/yy
            date = s.get_metadata('start_date')[:-3]  # Display mm/dd only
            # date = s.get_metadata('start_date').split("/")[1]  # Display dd only
            subject = s.get_metadata('subject')
            pell_ts = timestamps["Reward"]
            pell_double = np.nonzero(np.diff(pell_ts) < 0.5)[0]
            reward_times = s.get_rw_ts()
            box = s.get_metadata('box')
            # time format - "%H:%M:%S"
            exptime = s.get_metadata('start_time')[:2]

            d_list.append(date)
            box_list.append(box)
            time_list.append(exptime)

            if len(pell_double):
                dpell_change = 1
            if s_type == '5a':
                s_name = 'R' + str(ratio)
                c_ratio = s_name
            elif s_type == '5b':
                s_name = 'I' + str(interval)
                c_interval = s_name
            else:
                s_name = s_type.replace('_', '').replace('2', 'M').replace(
                    '3', 'Lh').replace('4', 'Lt').replace(
                    '6', 'B1').replace('7', 'B2')
            if 'B' in s_name:
                c_ratio = 'R' + str(ratio)
                c_interval = 'I' + str(interval)
            if not prev_name[0] == s_type[0]:
                stage_change.append(1)
                changes.append(0)
                change_idx.append(0)
            else:
                stage_change.append(0)
                if not c_ratio == prev_ratio and not c_interval == prev_interval:
                    changes.append([c_ratio, c_interval])
                    change_idx.append(1)
                elif not c_ratio == prev_ratio:
                    changes.append(c_ratio)
                    change_idx.append(1)
                elif not c_interval == prev_interval:
                    changes.append(c_interval)
                    change_idx.append(1)
                elif not dpell_change == dpell_old:
                    changes.append("DPell")
                    change_idx.append(1)
                else:
                    changes.append(0)
                    change_idx.append(0)
            # Calculates total reward (y axis variable)
            r_list.append(len(pell_ts))

            # Calculates FR & FI rewards and errors (alternative y axis variables)
            err_FI = 0
            err_FR = 0
            rw_FR = 0
            rw_FI = 0
            rw_double = 0
            err_plotted = 0
            corr_plotted = 0
            if s_type == '7_' or s_type == '6_':
                norm_r_ts, _, norm_err_ts, norm_dr_ts, _ = s.split_sess(
                    all_levers=True)
                sch_type = s.get_arrays('Trial Type')
                if s_type == '7_':
                    for i, _ in enumerate(norm_err_ts):
                        if sch_type[i] == 1:
                            err_FR = err_FR + len(norm_err_ts[i])
                        elif sch_type[i] == 0:
                            err_FI = err_FI + len(norm_err_ts[i])
                else:
                    err_FR = None
                    err_FI = None
                for i, _ in enumerate(norm_r_ts):
                    rw_double += len(norm_dr_ts[i])
                    if sch_type[i] == 1:
                        rw_FR = rw_FR + len(norm_r_ts[i])
                    elif sch_type[i] == 0:
                        rw_FI = rw_FI + len(norm_r_ts[i])
            elif s_type == '5a':
                err_FI = None
                err_FR = None
                rw_FR = len(reward_times)
                rw_FI = None
                rw_double = None
            elif s_type == '5b':
                err_FI = None
                err_FR = None
                rw_FR = None
                rw_FI = len(reward_times)
                rw_double = len(pell_double)

            else:
                err_FI = None
                err_FR = None
                rw_FR = None
                rw_FI = None
                rw_double = None

            # Updates list arrays with new session
            rw_FR_list.append(rw_FR)
            rw_FI_list.append(rw_FI)
            err_FR_list.append(err_FR)
            err_FI_list.append(err_FI)
            rw_double_list.append(rw_double)
            s_list.append(s_name)
            type_list.append('S-' + s_type[0])

            # Updates current iteration variables for next loop
            dpell_old = dpell_change
            prev_ratio = c_ratio
            prev_interval = c_interval
            prev_name = s_type

        if single_plot:
            rows, cols = [1, 4]
            size_multiplier = 5
            fig = plt.figure(
                figsize=(cols * size_multiplier, rows * size_multiplier),
                tight_layout=False)
            gs = gridspec.GridSpec(rows, cols, wspace=0.2, hspace=0.3)
            ax = fig.add_subplot(gs[0, :])
            out_name = "Timeline_" + subject
            if recent:
                out_name += "_recent"
            if details:
                out_name += "_details"
        else:
            ax = fig.add_subplot(gs[int(c), :])

        s_idx = np.arange(0, len(s_list))

        if details:  # Plots average sessions variables i.e. FR_corr, FI_corr, FR_err, FI_err, double_rw
            ratio_c = plt.cm.get_cmap('Wistia')
            interval_c = plt.cm.get_cmap('winter')
            # Change value to increase height of annotation
            # note_height = 0
            # y_axis = np.zeros((1, len(s_idx)))[0] + note_height
            y_axis = []
            # Sets line on which annotations appear
            for i, l in enumerate(rw_FR_list):
                if l is None:
                    y_axis.append(0)  # Hides non-stage 7 annotations
                else:
                    y_axis.append(l)
            plots = []
            labels = []
            # Dict controls lines to be plot. Set keys to 1 in plot_dict to include line in plot
            if det_err:
                plot_styles = {
                    "FR_Corr": [rw_FR_list, '*-', ratio_c(3 * 45), 0],
                    "FR_Err": [err_FR_list, 'x-', ratio_c(10 * 45), 1],
                    "FI_Corr": [rw_FI_list, '*-', interval_c(2 * 45), 0],
                    "FI_Err": [err_FI_list, 'x-', interval_c(4 * 45), 1],
                    "FI_DoubleR": [rw_double_list, '*-', 'hotpink', 0]}
                err_plotted = 1
            elif det_corr:
                plot_styles = {
                    "FR_Corr": [rw_FR_list, '*-', ratio_c(3 * 45), 1],
                    "FR_Err": [err_FR_list, 'x-', ratio_c(10 * 45), 0],
                    "FI_Corr": [rw_FI_list, '*-', interval_c(2 * 45), 1],
                    "FI_Err": [err_FI_list, 'x-', interval_c(4 * 45), 0],
                    "FI_DoubleR": [rw_double_list, '*-', 'hotpink', 1]}
                corr_plotted = 1
            else:
                plot_styles = {
                    "FR_Corr": [rw_FR_list, '*-', ratio_c(3 * 45), 1],
                    "FR_Err": [err_FR_list, 'x-', ratio_c(10 * 45), 1],
                    "FI_Corr": [rw_FI_list, '*-', interval_c(2 * 45), 1],
                    "FI_Err": [err_FI_list, 'x-', interval_c(4 * 45), 1],
                    "FI_DoubleR": [rw_double_list, '*-', 'hotpink', 1]}
                err_plotted = 1
                corr_plotted = 1

            plot_dict = {"FR_Corr": 1, "FR_Err": 1,
                         "FI_Corr": 1, "FI_Err": 1, "FI_DoubleR": 1}
            ax2 = ax.twinx()
            for k, val in plot_dict.items():
                if val:
                    s = plot_styles[k]
                    if k[-3:] == "Err":
                        ax_used = ax2
                    else:
                        ax_used = ax
                    h, = ax_used.plot(s_idx, s[0], s[1], label=k, linewidth='2',
                                      markersize=10, color=s[2], alpha=s[3])
                    plots.append(h)
                    labels.append(h.get_label())
            ax.set_title('\nSubject {} Timeline_Details'.format(
                subject), y=1.05, fontsize=25,
                color=bvmpc.bv_utils.mycolors(subject, sub_colors_dict))
        else:
            # Only plots total rewards
            y_axis = r_list
            h1, = plt.plot(
                s_idx, y_axis, label='Animal' + subject, linewidth='4',
                color=bvmpc.bv_utils.mycolors(subject, sub_colors_dict))

            ax.set_title(
                '\nSubject {} Timeline'.format(subject), y=1.05, fontsize=25,
                color=bvmpc.bv_utils.mycolors(subject, sub_colors_dict))

            # # Plot experiment time w rewards
            # ax2 = ax.twinx()
            # y_axis_2 = time_list
            # ax2.set_ylim([0, 24])
            # ax2.plot(s_idx, y_axis_2, label='Animal'+subject, linewidth='4',
            #                color=mycolors(subject))

        # Annotated changes in protocol
        annotate_fontsize = 12
        h2 = None
        h3 = None
        for i, c in enumerate(changes):
            if stage_change[i] == 1:
                h2 = ax.annotate(type_list[i], xy=(s_idx[i], y_axis[i]),
                                 ha='center', xytext=(0, (.2 * max(y_axis))),
                                 textcoords='offset points',
                                 arrowprops=dict(facecolor='blue', shrink=0.05), size=annotate_fontsize)
            elif change_idx[i] == 1:
                h3 = ax.annotate(str(c), xy=(s_idx[i], y_axis[i]),
                                 ha='center', xytext=(0, (.2 * max(y_axis))),
                                 textcoords='offset points',
                                 arrowprops=dict(facecolor='Red', shrink=0.05), size=annotate_fontsize)
        ax.set_xlim(0, len(s_idx))
        if show_date:
            # plots x-axis tics as dates
            plt.xticks(s_idx, d_list, fontsize=10)
            # plt.xticks(s_idx, box_list, fontsize=10)    # Plots x-tics as box
            ax.set_xlabel('Sessions (Dates)', fontsize=20)
        else:
            # plots x-axis ticks as stages
            plt.xticks(s_idx, s_list, fontsize=13)
            ax.set_xlabel('Sessions (Type)', fontsize=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=15)
        if details:
            if 'S-6' in type_list:
                ax.axhline(45, xmax=(type_list.index('S-6') / len(type_list)), color=interval_c(2 * 45),
                           linestyle='-.', linewidth='.5')
                ax.axhline(30, xmin=(type_list.index('S-6') / len(type_list)), color=interval_c(2 * 45),
                           linestyle='-.', linewidth='.5')
                ax.text(type_list.index('S-6'), 31, ' Max FI', fontsize=8,
                        color=interval_c(2 * 45), ha='left', va='bottom')
                ax.text(s_idx[0], 46, ' Max FI', fontsize=8,
                        color=interval_c(2 * 45), ha='left', va='bottom')
            else:
                ax.axhline(30, color=interval_c(2 * 45),
                           linestyle='-.', linewidth='.5')
                ax.text(s_idx[0], 31, ' Max FI', fontsize=8,
                        color=interval_c(2 * 45), ha='left', va='bottom')
            loc = 'top left'
            ax.set_ylabel('Correct Trials', fontsize=20)
            # set second y-axis labels
            ax2.tick_params(axis='y', labelsize=15)
            ax2.set_ylabel('Error Presses', fontsize=20)
        else:
            # plt.axhline(45, color='g', linestyle='-.', linewidth='.5')
            plt.axhline(60, color='r', linestyle='-.',
                        linewidth='.5')  # Marks max reward
            ax.set_ylabel('Total Rewards', fontsize=20)
            plots = [h1]
            labels = [h1.get_label()]
            loc = 'lower right'
        if h2 is not None and h3 is not None:
            plots.extend([h2.arrow_patch, h3.arrow_patch])
            labels.extend(['Stage Changes', 'Protocol Mod.'])
        elif h2 is not None:
            plots.append(h2.arrow_patch)
            labels.append('Stage Changes')
        elif h3 is not None:
            plots.append(h3.arrow_patch)
            labels.append('Protocol Mod.')
        plt.legend(plots, labels, loc=loc, ncol=2)
        if single_plot:
            out_name += ".png"
            print("Saved figure to {}".format(
                os.path.join(out_dir, out_name)))
            bv_plot.savefig(fig, os.path.join(out_dir, out_name))
            plt.close()

    if not single_plot:
        out_name = "Timeline_Sum_" + "-".join(sub_list)
        if recent:
            out_name += "_recent"
        if details:
            out_name += "_details"
            if corr_plotted == 1 and err_plotted == 1:
                pass
            elif corr_plotted == 1:
                out_name += "_corr"
            elif err_plotted == 1:
                out_name += "_err"
        out_name += ".png"
        print("Saved figure to {}".format(
            os.path.join(out_dir, out_name)))
        bv_plot.savefig(fig, os.path.join(out_dir, out_name))
        plt.close()
