"""Control script for MEDPC behaviour analysis."""
import os
import math
import numpy as np
from bv_parse_sessions import SessionExtractor, Session
import bv_analyse as bv_an
from bv_utils import make_dir_if_not_exists, print_h5, mycolors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from adjustText import adjust_text
from scipy import interpolate


def plot_sessions(summary=False, single=True, timeline=True):
    ''' Plots session summaries
    summary = True: Plots all sessions in a single plot, up to 6
    single = True: Plots single session summaries with breakdown of single blocks
    Timeline = True: Plots total rewards from beginining of first session
    '''
    # Parameters for specifying session
    sub_list = ['1', '2', '3', '4', '5', '6']
#    sub_list = ['6']
#    sub_list = ['1', '2', '3', '4']
#    sub_list = ['5', '6']
    s_list = ['7']
    d_list = ['08-10']
    in_dir = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\hdf5"
    out_dir = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\Plots"

    if summary:
        #  extracts hdf5 session based on specification
        s_grp, out_name = extract_hdf5s(in_dir, out_dir, sub_list, s_list, d_list)
        if len(s_grp) > 6:
            s_grp1 = s_grp[::2]
            sum_plot(s_grp1, out_name+'_1', out_dir, IRT=False)
            s_grp2 = s_grp[1::2]
            sum_plot(s_grp2, out_name+'_2', out_dir, IRT=False)        
        else:
            sum_plot(s_grp, out_name, out_dir, IRT=False)
    if single and summary:
        # Single Subject Plots
        for c, sub in enumerate(sub_list):
            s_grp, out_name = extract_hdf5s(in_dir, out_dir, sub, s_list, d_list)
            s_passed = []
            for s in s_grp:
                stage = s.get_metadata('name')[:2].replace('_', '')
                s_passed.append(stage)
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
                    bv_an.cumplot(s, out_dir, ax1, zoom=False, zoom_sch=False)
                    ax2 = fig.add_subplot(gs[i, 1])
                    bv_an.cumplot(s, out_dir, ax2, zoom=False, zoom_sch=True,
                              plot_error=False, plot_all=True)
                    ax3 = fig.add_subplot(gs[i, 2])
                    bv_an.cumplot(s, out_dir, ax3, zoom=False, zoom_sch=True,
                              plot_error=False, plot_all=False)
                    ax4 = fig.add_subplot(gs[i, 3])
                    bv_an.cumplot(s, out_dir, ax4, zoom=False, zoom_sch=True,
                              plot_error=True, plot_all=False)
#                if len(s_grp) == 1:
#                    fig.suptitle(('Subject ' + subject + ' Performance'), y = 1.05, fontsize=30)
#                else:
                plt.subplots_adjust(top=0.85)
                fig.suptitle(('Subject ' + subject + ' Performance'), fontsize=30)

#                # Seperate plots w line
#                ax1.hlines(1.13, -0, 4.9, clip_on=False,
#                           transform=ax1.transAxes, linewidth=0.7)
                out_name = "Sum" + out_name + ".png"
                print("Saved figure to {}".format(
                    os.path.join(out_dir, out_name)))
                fig.savefig(os.path.join(out_dir, out_name), dpi=400)
                plt.close()
            else:
                sum_plot(s_grp, out_name, out_dir, IRT=False, single=single)
            
    if timeline:
        if not single:
            sub_list = [['1', '2', '3', '4'], ['5', '6']]
            for l in sub_list:
                timeline_plot(l, in_dir, out_dir, single_plot=single)
        else:
            # Plots timeline for specified subjects
            timeline_plot(sub_list, in_dir, out_dir, single_plot=single)


def sum_plot(s_grp, out_name, out_dir, zoom=True, IRT=True, single=False):
    # Plots summary of day
    cols = 2
    if IRT or zoom:
        if len(s_grp) > 2:
            cols = 2*math.ceil(len(s_grp)/2)
            rows = 2
        else:
            rows = len(s_grp)
    else:
        rows = 1
    size_multiplier = 5
    fig = plt.figure(
            figsize=(cols * size_multiplier, rows * size_multiplier),
            tight_layout=False)
    gs = gridspec.GridSpec(rows, cols, wspace=0.2, hspace=0.3)

    for i, s in enumerate(s_grp):
        subject = s.get_metadata('subject')
        if IRT or zoom:
            ax1 = fig.add_subplot(gs[(i+2) % 2, int(i/2)*2])
        else:
            ax1 = fig.add_subplot(gs[0, i])
        bv_an.cumplot(s, out_dir, ax1, zoom=False, zoom_sch=False)

        if IRT:
            ax2 = fig.add_subplot(gs[i, 1])
            bv_an.IRT(s, out_dir, ax2,)
        elif zoom:
            ax2 = fig.add_subplot(gs[(i+2) % 2, int(i/2)*2+1])
            bv_an.cumplot(s, out_dir, ax2, zoom=False, zoom_sch=True,
                          plot_error=False, plot_all=True)
        plt.tight_layout()
    if single:
        fig.suptitle(('Subject ' + subject + ' Performance'), fontsize=30)
    out_name = "Sum" + out_name + ".png"
    print("Saved figure to {}".format(
        os.path.join(out_dir, out_name)))
    fig.savefig(os.path.join(out_dir, out_name), dpi=400)
    plt.close()


def timeline_plot(sub_list, in_dir, out_dir, single_plot=False):
    # Plot size
    rows, cols = [len(sub_list), 4]
    size_multiplier = 5
    fig = plt.figure(
            figsize=(cols * size_multiplier, rows * size_multiplier), 
            tight_layout=False) 
    gs = gridspec.GridSpec(rows, cols, wspace=0.4, hspace=0.5)
    for c, sub in enumerate(sub_list):
        # Plot total pellets across sessions
        s_grp, out_name = extract_hdf5s(in_dir, out_dir, sub)
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
        prev_name = '2'
        for i, s in enumerate(s_grp):
            s_type = s.get_metadata('name')[:2]
            subject = s.get_metadata('subject')
            timestamps = s.get_arrays()
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
                c_interval = 'I' + str(int(timestamps["Experiment Variables"][5] / 100))
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
                else:
                    changes.append(0)
                    change_idx.append(0)
            rewards_t = len(timestamps["Reward"])
            s_list.append(s_name)
            r_list.append(rewards_t)
            type_list.append('Stage '+s_type[0])
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
            out_name = "Timeline" + out_name + ".png"
        else:
            ax = fig.add_subplot(gs[int(c), :])

        h1, = plt.plot(s_idx, r_list, label='Animal'+subject, linewidth='4',
                 color=mycolors(subject))
        
#        # Testing alterative annotation
#        texts = []
#        for stage, x, y, s in zip(stage_change, s_idx, r_list, type_list):
#            if stage == 1:
#                texts.append(plt.text(x, y, s))
#        f = interpolate.interp1d(s_idx, r_list)
#        x = np.arange(min(s_idx), max(s_idx), 0.0005)
#        y = f(x)  
#        adjust_text(texts, x=x, y=y, only_move={'points':'xy', 'text':'xy'}, 
#                    autoalign='xy', force_points=0.5, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

        # Annotated changes in protocol
        for i, c in enumerate(changes):
            if stage_change[i] == 1:
                h2 = ax.annotate(type_list[i], xy=(s_idx[i], r_list[i]), xytext=(s_idx[i], r_list[i]+(0.1*max(r_list))),
                            arrowprops=dict(facecolor='blue', shrink=0.05))
            elif change_idx[i] == 1:
                h3 = ax.annotate(str(c), xy=(s_idx[i], r_list[i]), xytext=(s_idx[i], r_list[i]+(0.1*max(r_list))),
                            arrowprops=dict(facecolor='Red', shrink=0.05))
        ax.set_xlim(0, len(s_idx))
        plt.xticks(s_idx, s_list, fontsize=13)
        ax.tick_params(axis='y', labelsize=15)
        plt.axhline(45, color='g', linestyle='-.', linewidth='.5')
        plt.axhline(90, color='r', linestyle='-.', linewidth='.5')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Sessions', fontsize=20)
        ax.set_ylabel('Total Rewards', fontsize=20)
        plt.legend([h1, h2.arrow_patch, h3.arrow_patch], (h1.get_label(), 'Stage Changes', 'Protocol Modification'))
        ax.set_title('\nSubject {} Timeline'.format(subject), fontsize=25)
        
        if single_plot:
            print("Saved figure to {}".format(
                    os.path.join(out_dir, out_name)))
            fig.savefig(os.path.join(out_dir, out_name), dpi=400)
            plt.close()
            
    if not single_plot:
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
    name_dict = {}
    if sub_list is not None:
        name_dict["sub_list"] = sub_list
    if s_list is not None:
        name_dict["s_list"] = s_list
    if d_list is not None:
        name_dict["d_list"] = d_list
    out_name = ""
    names = ["sub_list", "d_list", "s_list"]
    for name in names:
        out_name = out_name + "_" + str(name_dict.get(name, ""))
    out_name.replace("__", "_")
    out_name.replace("__", "_")
    out_name = out_name
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
    return s_grp, out_name


def convert_to_hdf5(filename, out_dir):
    """Convert all sessions in filename to hdf5 and store in out_dir."""
    make_dir_if_not_exists(out_dir)

    s_extractor = SessionExtractor(filename, verbose=True)

    for s in s_extractor:  # Batch run for file
        s.save_to_h5(out_dir)


def load_hdf5(filename, out_dir):
    print_h5(filename)
    session = Session(h5_file=filename)
    print(session)

#    bv_an.IRT(session, out_dir, False)
#    bv_an.cumplot(session, out_dir, False)
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
#        bv_an.IRT(s, out_dir, False)  # Doesnt work with stage 6


if __name__ == "__main__":
    """Main control."""
#    # Batch processing of sessions in folder
#    in_dir = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\\"
#    out_dir = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\hdf5"
#    in_files = os.listdir(in_dir)
#    for file in in_files:
#        filename = os.path.join(in_dir, file)
#        if os.path.isfile(filename):
##            main(filename)  # Uncomment to run from mpc file
#            convert_to_hdf5(filename, out_dir)  # Uncomment to convert to hdf5
    
    # Processing specific sessions from hdf5
    plot_sessions()

    # Running single session files
#    filename = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\!2019-08-04"
#     filename = r"G:\test"
#     filename = r"/home/sean/Documents/Data/!2019-07-22"

#    out_dir = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\Plots"
#    out_dir = r"G:\out_plots"
#    out_dir = r"/home/sean/Documents/Data/results"
    
#    filename = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\!2019-08-12"
#    out_dir = r"F:\PhD (Shane O'Mara)\Operant Data\IR Discrimination Pilot 1\hdf5"
#    convert_to_hdf5(filename, out_dir)  # Uncomment to convert to hdf5
#    run_mpc_file(filename, out_dir)

#    filename = r"/home/sean/Documents/Data/h5_files/1_07-22-19_11-30_5a_FixedRatio_p.h5"

#    load_hdf5(filename, out_dir)
