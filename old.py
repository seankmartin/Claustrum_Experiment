# Originally in bv_control
def compare_variables(start_dir):
    """ Temporary Function to plot difference between errors"""
    # Only works for stage 7
    # start_dir = r"G:\!Operant Data\Ham"
    in_dir = os.path.join(start_dir, "hdf5")
    sub_list = ['1', '2', '3', '4']
    s_list = ['7']
    grpA_d_list = ['08-10', '08-11', '08-12']
    grpB_d_list = ['08-17', '08-18', '08-19']
    grpC_d_list = ['09-15', '09-16', '09-17']

    s_grpA = extract_sessions(in_dir, sub_list, s_list, grpA_d_list)
    s_grpB = extract_sessions(in_dir, sub_list, s_list, grpB_d_list)
    s_grpC = extract_sessions(in_dir, sub_list, s_list, grpC_d_list)
    s_grpA.pop()
    s_grpA.pop(-4)
    s_grpA.pop(-7)
    s_grpA.pop(-10)
    s_grps = [s_grpA, s_grpB, s_grpC]
    FR_means = []
    FI_means = []
    FR_stds = []
    FI_stds = []

    for s in s_grps:
        grp_FRerr, grp_FIerr = grp_errors(s)
        FRerr_arr = np.array(grp_FRerr)
        FIerr_arr = np.array(grp_FIerr)
        FR_mean = np.mean(FRerr_arr, axis=0)
        FI_mean = np.mean(FIerr_arr, axis=0)
        FR_means.append(FR_mean)
        FI_means.append(FI_mean)
        FR_std = np.std(FRerr_arr, axis=0)
        FI_std = np.std(FIerr_arr, axis=0)
        FR_stds.append(FR_std)
        FI_stds.append(FI_std)

    # x_label = ['FR6_noDP-Ratio', 'FR6_NoDP-Int', 'FR8-Ratio',
    #            'FR8-Int', 'FR18-Ratio', 'FR18-Int']
    ratio_c = plt.cm.get_cmap('Wistia')
    interval_c = plt.cm.get_cmap('winter')

    _, ax = plt.subplots()
    ind = np.arange(len(FR_means))  # the x locations for the groups
    width = 0.35  # the width of the bars
    ax.bar(ind - width / 2, FR_means, width,
           yerr=FR_stds, label='FR', color=ratio_c(10 * 45), align='center')
    ax.bar(ind + width / 2, FI_means, width,
           yerr=FI_stds, label='FI', color=interval_c(4 * 45), align='center')
    # ax.bar(ind - width/2, np.mean(err_arr, axis=1), tick_label=x_label,
    #        yerr=np.std(err_arr, axis=1), align='center',
    #        alpha=0.5, ecolor='black', capsize=10)
    ax.set_xticks(ind)
    ax.set_xticklabels(('FR6_noDP', 'FR8', 'FR18'))
    ax.set_ylabel('Error Presses')
    ax.set_xlabel('Sessions-Type')
    ax.set_title('Errors Comparison')
    ax.legend()
    plt.show()

# Originally in bv_control
def grp_errors(s_grp):
    grp_FRerr = []
    grp_FIerr = []
    for i, s in enumerate(s_grp):
        err_FI = 0
        err_FR = 0
        _, _, norm_err_ts, _, _ = s.split_sess(
            plot_all=True)
        sch_type = s.get_arrays('Trial Type')
        for i, _ in enumerate(norm_err_ts):
            if sch_type[i] == 1:
                err_FR = err_FR + len(norm_err_ts[i])
            elif sch_type[i] == 0:
                err_FI = err_FI + len(norm_err_ts[i])
        grp_FRerr.append(err_FR)
        grp_FIerr.append(err_FI)
    return grp_FRerr, grp_FIerr
