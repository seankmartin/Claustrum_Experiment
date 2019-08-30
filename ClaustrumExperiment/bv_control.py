"""Control script for MEDPC behaviour analysis."""
import os
import math
import numpy as np
from bvmpc.bv_parse_sessions import SessionExtractor, Session
import bvmpc.bv_analyse as bv_an
from bvmpc.bv_utils import make_dir_if_not_exists, print_h5, mycolors, daterange
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from adjustText import adjust_text
from scipy import interpolate
from datetime import date


def plot_batch_sessions():
    # start_date = date(2019, 7, 15)  # date(year, mth, day)
    start_date = date(2019, 8, 10)  # date(year, mth, day)
    # start_date = date.today()
    # end_date = date.today()
    end_date = date(2019, 8, 14)

    for single_date in daterange(start_date, end_date):
        d = [single_date.isoformat()[-5:]]
        print(d)
        plot_sessions(d)


def plot_sessions(d_list, summary=True, single=False, timeline=False,
                  recent=False, show_date=False,
                  int_only=False, corr_only=True):
    ''' Plots session summaries
    summary = True: Plots all sessions in a single plot, up to 6
    single = True: Plots single session summaries with breakdown of single blocks
    Timeline = True: Plots total rewards from beginining of first session
    int_only = True: Plots only interval trials in zoomed schedule plot
    corr_only = True: Plots seperate summary plot with correct only trials
    '''
    # Parameters for specifying session
    sub_list = ['1', '2', '3', '4', '5', '6']
    # sub_list = ['6']
    # sub_list = ['1', '2', '3', '4']
    # sub_list = ['5', '6']
    s_list = ['4', '5a', '5b', '6', '7']
    #  d_list = ['08-14']

    start_dir = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1"
    # start_dir = r"G:\!Operant Data\Ham"
    in_dir = start_dir + "\hdf5"
    out_dir = start_dir + "\Plots"
    make_dir_if_not_exists(out_dir)

    if summary:
        #  extracts hdf5 session based on specification
        max_plot = 4  # Set max plots per figure
        s_grp = extract_hdf5s(in_dir, out_dir, sub_list, s_list, d_list)
        if s_grp == []:
            return print("***No Files Extracted***")
        idx = 0
        if len(s_grp) > max_plot:
            j = 0
            s_grp_split = []
            s_grp_idx = np.arange(len(s_grp))
            for i in s_grp_idx[max_plot-1::max_plot]:
                s_grp_split.append(s_grp[j:i+1])
                j = i+1

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
        s_grp = extract_hdf5s(in_dir, out_dir, sub_list, s_list, d_list)
        if s_grp == []:
            return print("***No Files Extracted***")

        idx = 0
        if len(s_grp) > max_plot:
            j = 0
            s_grp_split = []
            s_grp_idx = np.arange(len(s_grp))
            for i in s_grp_idx[max_plot-1::max_plot]:
                s_grp_split.append(s_grp[j:i+1])
                j = i+1

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
            s_grp = extract_hdf5s(in_dir, out_dir, sub, s_list, d_list)
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
                size_multiplier = 5
                rows, cols = [len(s_grp), 4]
                fig = plt.figure(
                    figsize=(cols * size_multiplier,
                             rows * size_multiplier), tight_layout=True)
                gs = gridspec.GridSpec(rows, cols, wspace=0.3, hspace=0.3)
                if len(s_grp) == 1:
                    fig.tight_layout(rect=[0, 0.03, 0.8, 0.95])
                for i, s in enumerate(s_grp):
                    ax1 = fig.add_subplot(gs[i, 0])
                    bv_an.cumplot(s, out_dir, ax1, int_only,
                                  zoom=False, zoom_sch=False)
                    ax2 = fig.add_subplot(gs[i, 1])
                    bv_an.cumplot(s, out_dir, ax2, int_only, zoom=False, zoom_sch=True,
                                  plot_error=False, plot_all=True)
                    ax3 = fig.add_subplot(gs[i, 2])
                    bv_an.cumplot(s, out_dir, ax3, int_only, zoom=False, zoom_sch=True,
                                  plot_error=False, plot_all=False)
                    ax4 = fig.add_subplot(gs[i, 3])
                    bv_an.cumplot(s, out_dir, ax4, int_only, zoom=False, zoom_sch=True,
                                  plot_error=True, plot_all=False)
                plt.subplots_adjust(top=0.85)
                fig.suptitle(('Subject ' + subject + ' Performance'),
                             color=mycolors(subject), fontsize=30)

                # # Seperate plots w line
                # ax1.hlines(1.13, -0, 4.9, clip_on=False,
                #             transform=ax1.transAxes, linewidth=0.7)
                s_print = np.array_str(np.unique(np.array(s_passed)))
                d_print = np.array_str(np.unique(np.array(d_passed)))
                out_name = "Sum_" + subject + "_" + d_print + "_" + s_print + ".png"
                print("Saved figure to {}".format(
                    os.path.join(out_dir, out_name)))
                fig.savefig(os.path.join(out_dir, out_name), dpi=400)
                plt.close()
            else:
                sum_plot(s_grp, idx, out_dir, single=single)

    if timeline:
        if not single:
            sub_list = [['1', '2', '3', '4'], ['5', '6']]
            for l in sub_list:
                timeline_plot(l, in_dir, out_dir, single_plot=single,
                              recent=recent, show_date=show_date)
        else:
            # Plots timeline for specified subjects
            timeline_plot(sub_list, in_dir, out_dir, single_plot=single,
                          recent=recent, show_date=show_date)


def sum_plot(s_grp, idx, out_dir, zoom=True, single=False,
             int_only=False, corr_only=False):
    # Plots summary of day
    if zoom:
        if len(s_grp) > 2:
            cols = 2*math.ceil(len(s_grp)/2)
            rows = 2
        else:
            rows = len(s_grp)
            cols = 2
    else:
        if len(s_grp) > 4:
            rows = math.ceil(len(s_grp)/4)
            cols = 4
        else:
            cols = len(s_grp)
            rows = 1
    size_multiplier = 5
    fig = plt.figure(
        figsize=(cols * size_multiplier, rows * size_multiplier),
        tight_layout=False)
    gs = gridspec.GridSpec(rows, cols, wspace=0.2, hspace=0.3)
    s_passed = []
    d_passed = []

    for i, s in enumerate(s_grp):
        subject = s.get_metadata('subject')
        stage = s.get_metadata('name')[:2].replace('_', '')
        date = s.get_metadata("start_date").replace("/", "-")
        s_passed.append(stage)
        d_passed.append(date[:5])

        if zoom:
            ax1 = fig.add_subplot(gs[(i+2) % 2, int(i/2)*2])
        else:
            ax1 = fig.add_subplot(gs[0, i])

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
            ax2 = fig.add_subplot(gs[(i+2) % 2, int(i/2)*2+1])
            bv_an.IRT(s, out_dir, ax2)
        elif zoom:
            ax2 = fig.add_subplot(gs[(i+2) % 2, int(i/2)*2+1])
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
    fig.savefig(os.path.join(out_dir, out_name), dpi=400)
    plt.close()


def timeline_plot(sub_list, in_dir, out_dir, single_plot=False,
                  recent=False, show_date=False):
    # Plot size
    rows, cols = [len(sub_list), 4]
    size_multiplier = 5
    fig = plt.figure(
        figsize=(cols * size_multiplier, rows * size_multiplier),
        tight_layout=False)
    gs = gridspec.GridSpec(rows, cols, wspace=0.4, hspace=0.5)
    for c, sub in enumerate(sub_list):
        # Plot total pellets across sessions
        s_grp = extract_hdf5s(in_dir, out_dir, sub)
        s_list = []
        r_list = []
        changes = []
        stage_change = []
        change_idx = []
        prev_ratio = []
        prev_interval = []
        c_ratio = []
        c_interval = []
        type_list = []
        dpell_change = []
        dpell_old = []
        prev_name = '2'
        d_list = []
        if recent:
            s_grp = s_grp[-20:]
        else:
            pass

        for i, s in enumerate(s_grp):
            s_type = s.get_metadata('name')[:2]
            timestamps = s.get_arrays()
            date = s.get_metadata('start_date')[3:5]
            subject = s.get_metadata('subject')
            pell_ts = timestamps["Reward"]
            pell_double = np.nonzero(np.diff(pell_ts) < 0.5)[0]
            d_list.append(date)
            if len(pell_double):
                dpell_change = 1
            if s_type == '5a':
                s_name = 'R' + str(int(timestamps["Experiment Variables"][3]))
                c_ratio = s_name
            elif s_type == '5b':
                s_name = 'I' + str(int(timestamps[
                    "Experiment Variables"][3]/100))
                c_interval = s_name
            else:
                s_name = s_type.replace('_', '').replace('2', 'M').replace(
                    '3', 'Lh').replace('4', 'Lt').replace(
                    '6', 'B1').replace('7', 'B2')
            if 'B' in s_name:
                c_ratio = 'R' + str(int(timestamps["Experiment Variables"][3]))
                c_interval = 'I' + \
                    str(int(timestamps["Experiment Variables"][5] / 100))
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
            rewards_t = len(timestamps["Reward"]) + len(pell_double)
            s_list.append(s_name)
            r_list.append(rewards_t)
            type_list.append('S-'+s_type[0])
            dpell_old = dpell_change
            prev_ratio = c_ratio
            prev_interval = c_interval
            prev_name = s_type
        s_idx = np.arange(0, len(s_list))
        if single_plot:
            rows, cols = [1, 4]
            size_multiplier = 5
            fig = plt.figure(
                figsize=(cols * size_multiplier, rows * size_multiplier),
                tight_layout=False)
            gs = gridspec.GridSpec(rows, cols, wspace=0.2, hspace=0.3)
            ax = fig.add_subplot(gs[0, :])
            if recent:
                out_name = "Timeline_" + subject + "_recent" + ".png"
            else:
                out_name = "Timeline_" + subject + ".png"
        else:
            ax = fig.add_subplot(gs[int(c), :])

        h1, = plt.plot(s_idx, r_list, label='Animal'+subject, linewidth='4',
                       color=mycolors(subject))

        # # Testing alterative annotation
        # texts = []
        # for stage, x, y, s in zip(stage_change, s_idx, r_list, type_list):
        #     if stage == 1:
        #         texts.append(plt.text(x, y, s))
        # f = interpolate.interp1d(s_idx, r_list)
        # x = np.arange(min(s_idx), max(s_idx), 0.0005)
        # y = f(x)
        # adjust_text(texts, x=x, y=y, only_move={'points':'xy', 'text':'xy'},
        #             autoalign='xy', force_points=0.5, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

        # Annotated changes in protocol
        h2 = None
        h3 = None
        for i, c in enumerate(changes):
            if stage_change[i] == 1:
                #                h2 = ax.annotate(type_list[i], xy=(s_idx[i], r_list[i]),
                #                xytext=(s_idx[i], r_list[i]+(0.1*max(r_list))),
                #                            arrowprops=dict(facecolor='blue', shrink=0.05))
                h2 = ax.annotate(type_list[i], xy=(s_idx[i], r_list[i]),
                                 ha='center', xytext=(0, (.2*max(r_list))),
                                 textcoords='offset points',
                                 arrowprops=dict(facecolor='blue', shrink=0.05))
            elif change_idx[i] == 1:
                #                h3 = ax.annotate(str(c), xy=(s_idx[i], r_list[i]),
                #                xytext=(s_idx[i], r_list[i]+(0.1*max(r_list))),
                #                            arrowprops=dict(facecolor='Red', shrink=0.05))
                h3 = ax.annotate(str(c), xy=(s_idx[i], r_list[i]),
                                 ha='center', xytext=(0, (.2*max(r_list))),
                                 textcoords='offset points',
                                 arrowprops=dict(facecolor='Red', shrink=0.05))
        ax.set_xlim(0, len(s_idx))
        if show_date:
            # plots x-axis ticks as dates
            plt.xticks(s_idx, d_list, fontsize=10)
        else:
            # plots x-axis ticks as stages
            plt.xticks(s_idx, s_list, fontsize=13)
        ax.tick_params(axis='y', labelsize=15)
        plt.axhline(45, color='g', linestyle='-.', linewidth='.5')
        plt.axhline(90, color='r', linestyle='-.', linewidth='.5')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Sessions', fontsize=20)
        ax.set_ylabel('Total Rewards', fontsize=20)
        if h2 is not None and h3 is not None:
            plt.legend([h1, h2.arrow_patch, h3.arrow_patch], (h1.get_label(),
                                                              'Stage Changes', 'Protocol Modification'))
        elif h2 is not None:
            plt.legend([h1, h2.arrow_patch], (h1.get_label(),
                                              'Stage Changes'))
        elif h3 is not None:
            plt.legend([h1, h3.arrow_patch], (h1.get_label(),
                                              'Protocol Modification'))
        ax.set_title('\nSubject {} Timeline'.format(subject), fontsize=25)

        if single_plot:
            print("Saved figure to {}".format(
                os.path.join(out_dir, out_name)))
            fig.savefig(os.path.join(out_dir, out_name), dpi=400)
            plt.close()

    if not single_plot:
        if recent:
            fig.suptitle('Timelines for IR ' +
                         "-".join(sub_list) + "_recent", fontsize=30)
            out_name = "Timeline_Sum_" + \
                "-".join(sub_list) + "_recent" + ".png"
        else:
            fig.suptitle('Timelines for IR ' + "-".join(sub_list), fontsize=30)
            out_name = "Timeline_Sum_" + "-".join(sub_list) + ".png"
        print("Saved figure to {}".format(
            os.path.join(out_dir, out_name)))
        fig.savefig(os.path.join(out_dir, out_name), dpi=400)
        plt.close()


def extract_hdf5s(in_dir, out_dir, sub_list=None, s_list=None, d_list=None):
    '''Extracts specified sessions from hdf5 files '''

    def should_use(val, vlist):
        if vlist is None:
            return True
        if val in vlist:
            return True
        return False

    in_files = os.listdir(in_dir)
    s_grp = []
    # name_dict = {}
    # if sub_list is not None:
    #     name_dict["sub_list"] = sub_list
    # if s_list is not None:
    #     name_dict["s_list"] = s_list
    # if d_list is not None:
    #     name_dict["d_list"] = d_list
    # out_name = ""
    # names = ["sub_list", "d_list", "s_list"]
    # for name in names:
    #     out_name = out_name + "_" + str(name_dict.get(name, "sub_list"))
    # out_name.replace("__", "_")
    # out_name.replace("__", "_")
    # out_name = out_name
    for file in in_files:
        splits = file.split('_')
        subject = splits[0]
        date = splits[1][:5]
        s_type = splits[3]
        subject_ok = should_use(subject, sub_list)
        type_ok = should_use(s_type, s_list)
        date_ok = should_use(date, d_list)
        if subject_ok and type_ok and date_ok:
            filename = os.path.join(in_dir, file)
            if os.path.isfile(filename):
                session = load_hdf5(filename, out_dir)
                s_grp.append(session)
    print('Total Files extracted: {}'.format(len(s_grp)))
    return s_grp


def convert_to_hdf5(filename, out_dir):
    """Convert all sessions in filename to hdf5 and store in out_dir."""
    make_dir_if_not_exists(out_dir)

    s_extractor = SessionExtractor(filename, verbose=True)

    for s in s_extractor:  # Batch run for file
        stage = s.get_metadata('name')
        if stage == 'DNMTS':
            continue
        else:
            s.save_to_h5(out_dir)


def load_hdf5(filename, out_dir):
    print_h5(filename)
    session = Session(h5_file=filename)
    print(session)

    # bv_an.IRT(session, out_dir, False)
    # bv_an.cumplot(session, out_dir, False)
    return session


def run_mpc_file(filename, out_dir):
    """Take in a filename and out_dir then run the main control logic."""
    make_dir_if_not_exists(out_dir)

    s_extractor = SessionExtractor(filename, verbose=True)

    for s in s_extractor:  # Batch run for file

        time_taken = s.time_taken()
        timestamps = s.get_arrays()

        print("Session duration {} mins".format(time_taken))
        if len(timestamps.keys()) == 0:
            print('Not ready for analysis!')
            continue

        bv_an.cumplot(s, out_dir, False)
        # bv_an.IRT(s, out_dir, False)  # Doesnt work with stage 6


if __name__ == "__main__":
    """Main control."""
    start_dir = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1"  # from Ham Personal HD
    # start_dir = r"G:\!Operant Data\Ham"  # from Ham Personal Thumbdrive

    # # Batch processing of sessions in folder
    # in_dir = start_dir
    # out_dir = os.path.join(start_dir, "hdf5")
    # in_files = os.listdir(in_dir)
    # for file in in_files:
    #     filename = os.path.join(in_dir, file)
    #     if os.path.isfile(filename):
    #         convert_to_hdf5(filename, out_dir)  # Uncomment to convert to hdf5

    # # Processing of single sessions
    # filename = os.path.join(start_dir, "!2019-08-11")
    # out_dir = os.path.join(start_dir, "hdf5")
    # convert_to_hdf5(filename, out_dir)  # Uncomment to convert to hdf5

    # Processing specific sessions from hdf5

    # plot_sessions([date.today().isoformat()[-5:]])
    # plot_sessions(['07-17'])
    plot_batch_sessions()

    # # Running single session files
    # filename = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\!2019-08-04"
    # filename = r"G:\test"
    # filename = r"/home/sean/Documents/Data/!2019-07-22"

    # out_dir = r"G:\out_plots"
    # out_dir = r"/home/sean/Documents/Data/results"

    # run_mpc_file(filename, out_dir)

    # load_hdf5(filename, out_dir)
