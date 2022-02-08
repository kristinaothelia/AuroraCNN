from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from pylab import *
import math
import operator
# -----------------------------------------------------------------------------

def Get_hours():
    # Make times
    times = []
    for i in range(24):

        if i < 10:
            str = '0%s' %i
        else:
            str = '%s' %i
        times.append(str)
    return times

def autolabel(n):
    """
    Attach a text label above each bar displaying its height
    autolabel() from: https://stackoverflow.com/a/42498711
    """
    for i in n:
        height = i.get_height()
        plt.text(i.get_x() + i.get_width()/2., 1.01*height,
                '%d' % int(height), ha='center', va='bottom')

def subcategorybar(X, vals, leg, width=0.8):
    """
    From: https://stackoverflow.com/a/48158449
    """
    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        #plt.bar(_X - width/2. + i/float(n)*width, vals[i],
        #        width=width/float(n), align="edge")
        autolabel(plt.bar(_X - width/2. + i/float(n)*width, vals[i],
                width=width/float(n), align="edge"))
    plt.legend(leg)
    plt.xticks(_X, X)

def data(entry, Hours, TH, error_hour, speed):

    Hours.append(entry.timepoint[-8:-6])
    TH.append(entry.timepoint[5:7]+entry.timepoint[-8:-6]) # Month, day and hour

    old_score = error_hour[entry.timepoint[-8:-6]]
    new_score = speed

    edit = [old_score[0]+new_score, old_score[1]+1]
    error_hour.update({entry.timepoint[-8:-6]: edit})

def aurora_speed_stats(container, label, year=False):

    input = 'Speed, km/s'

    Hours = []
    TH = []

    error_hour = {"00": [0, 0], "01": [0, 0], "02": [0, 0], "03": [0, 0], \
                  "04": [0, 0], "05": [0, 0], "06": [0, 0], "07": [0, 0], \
                  "08": [0, 0], "09": [0, 0], "10": [0, 0], "11": [0, 0], \
                  "12": [0, 0], "13": [0, 0], "14": [0, 0], "15": [0, 0], \
                  "16": [0, 0], "17": [0, 0], "18": [0, 0], "19": [0, 0], \
                  "20": [0, 0], "21": [0, 0], "22": [0, 0], "23": [0, 0]}

    if year:

        for entry in container:

            if entry.timepoint[:4] == year:
                if float(entry.solarwind[input]) != 99999.9:

                    if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                        if entry.label == label:
                            speed = float(entry.solarwind[input])
                            print(speed)
                            data(entry, Hours, TH, error_hour, speed)

                    if label == "aurora": # arc, diffuse and discrete aurora is counted as one
                        if entry.label != LABELS[0]:
                            speed = float(entry.solarwind[input])
                            data(entry, Hours, TH, error_hour, speed)

    else:
        print("not coded yet")

    #return a_less_POS, a_less_NEG, arc_POS, arc_NEG, diff_POS, diff_NEG, disc_POS, disc_NEG, a_less_POS_err, a_less_NEG_err, test_p, test_n
    #return a_less, arc, diff, disc
    return Hours, TH, error_hour

#aurora_speed_stats(container, label, year=False)

def sub_plots(year, wl, hours, T_c_N, T_arc_N, T_diff_N, T_disc_N, error, T_Aurora_N=None, month_name=None,  N=4):

    #f = plt.figure(figsize=(10,3))

    if year[:4] == '2020':
        shape = '*-'
    elif year[:4] == '2014':
        shape = '.-'
    elif year[:4] == '2016':
        shape = 'o-'
    else:
        shape = 'x-'

    subplot(N,1,1)
    if month_name != None:
        plt.title('Statistics ({}) for all classes [{}]'.format(month_name, wl), fontsize=18)
    else:
        if len(year) > 4:
            plt.title('Yearly statistics for all classes (weighted) [{}]'.format(wl), fontsize=18)
        else:
            plt.title('Yearly statistics for all classes [{}]'.format(wl), fontsize=18)
    plt.plot(hours, T_arc_N, shape, label='arc - '+year)
    e = error[1]/2  # For 1.5 lim
    plt.errorbar(hours, T_arc_N, yerr=e, fmt='none', ecolor='k', elinewidth=0.7, capsize=2)
    plt.ylabel("%", fontsize=15)
    #plt.ylim(-0.2, 3)
    plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True) #, ncol=2
    #plot(hours, T_arc_N, 'arc', year, month=None, monthly=False)

    subplot(N,1,2)
    #plot(hours, T_diff_N, 'diffuse', year, month=None, monthly=False)
    plt.plot(hours, T_diff_N, shape, label='diffuse - '+year)
    plt.errorbar(hours, T_diff_N, yerr=error[2], fmt='none', ecolor='k', elinewidth=0.7, capsize=2)
    plt.ylabel("%", fontsize=15)
    #plt.ylim(0, 4)
    plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True)

    subplot(N,1,3)
    #plot(hours, T_disc_N, 'discrete', year, month=None, monthly=False)
    plt.plot(hours, T_disc_N, shape, label='discrete - '+year)
    plt.errorbar(hours, T_disc_N, yerr=error[3], fmt='none', ecolor='k', elinewidth=0.7, capsize=2)    # elinewidth=0.1,
    plt.ylabel("%", fontsize=15)
    #plt.ylim(0, 4)
    plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True)

    subplot(N,1,4)
    #plot(hours, T_c_N, 'no aurora', year, month=None, monthly=False, axis=True)
    plt.plot(hours, T_c_N, shape, label='no aurora - '+year)
    plt.errorbar(hours, T_c_N, yerr=error[0], fmt='none', ecolor='k', elinewidth=0.7, capsize=2)
    #plt.xlabel("Hour of the day", fontsize=13)
    plt.ylabel("%", fontsize=15)
    #plt.ylim(0, 4)
    plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True)

    if N == 5:
        subplot(N,1,5)
        #plot(hours, T_c_N, 'no aurora', year, month=None, monthly=False, axis=True)
        plt.plot(hours, T_Aurora_N, shape, label='aurora - '+year)
        plt.errorbar(hours, T_Aurora_N, yerr=error[4], fmt='none', ecolor='k', elinewidth=0.7, capsize=2)
        plt.xlabel("Hour of the day", fontsize=15)
        plt.ylabel("%", fontsize=15)
        #plt.ylim(0, 4.5)
        plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True)
    else:
        plt.xlabel("Hour of the day", fontsize=15)

    #plt.tight_layout(rect=[0,0,0.75,1])

def Hour_subplot(container, year, wl, month_name='Jan', N=4, month=False, weight=False):

    hours = Get_hours()

    Hours, TH, max_score, err_h = aurora_speed_stats(container=container, label="aurora-less", year=year[:4])
    Hours_arc, TH_arc, err_h_arc = aurora_speed_stats(container=container, label="arc", year=year[:4])
    Hours_diff, TH_diff, err_h_diff = aurora_speed_stats(container=container, label="diffuse", year=year[:4])
    Hours_disc, TH_disc, err_h_disc = aurora_speed_stats(container=container, label="discrete", year=year[:4])
    Hours_A, TH_A, err_h_A = aurora_speed_stats(container=container, label="aurora", year=year[:4])

    list = [err_h, err_h_arc, err_h_diff, err_h_disc, err_h_A]
    list_arrays = []

    print(err_h)

    exit()

    for i in range(len(list)):
        error_std = []
        for key, value in list[i].items():
            #print('key/hour', 'value/error', 'error std')
            #print(key, value, value[0]/value[1])
            error_std.append(value[0]/value[1])
            #error_std[value] = value[0]/value[1]

        error_hist = np.array(error_std)
        list_arrays.append(error_hist)

    if month:

        T_c = []; T_arc = []; T_diff = []; T_disc = []
        T_c_N = []; T_arc_N = []; T_diff_N = []; T_disc_N = []
        T_A = []; T_A_N = []

        # Note, removed Mars, because of no data
        # Aurora-less
        TH_01, TH_11, TH_12, TH_01_N, TH_11_N, TH_12_N = get_hour_count_per_month(TH, hours)
        #TH_01, TH_02, TH_03, TH_10, TH_11, TH_12, \
        #TH_01_N, TH_02_N, TH_10_N, TH_11_N, TH_12_N \
        #= get_hour_count_per_month(TH, hours)
        # Arc
        TH_01_arc, TH_11_arc, TH_12_arc, TH_01_arc_N, TH_11_arc_N, TH_12_arc_N \
        = get_hour_count_per_month(TH_arc, hours)
        # Diff
        TH_01_diff, TH_11_diff, TH_12_diff, TH_01_diff_N, TH_11_diff_N, TH_12_diff_N \
        = get_hour_count_per_month(TH_diff, hours)
        # Disc
        TH_01_disc, TH_11_disc, TH_12_disc, TH_01_disc_N, TH_11_disc_N, TH_12_disc_N \
        = get_hour_count_per_month(TH_disc, hours)
        # Aurora
        TH_01_A, TH_11_A, TH_12_A, TH_01_A_N, TH_11_A_N, TH_12_A_N \
        = get_hour_count_per_month(TH_A, hours)

        T_c.extend([TH_01, TH_11, TH_12])
        T_arc.extend([TH_01_arc, TH_11_arc, TH_12_arc])
        T_diff.extend([TH_01_diff, TH_11_diff, TH_12_diff])
        T_disc.extend([TH_01_disc, TH_11_disc, TH_12_disc])
        T_A.extend([TH_01_A, TH_11_A, TH_12_A])

        # Normalized
        T_c_N.extend([TH_01_N, TH_11_N, TH_12_N])
        T_arc_N.extend([TH_01_arc_N, TH_11_arc_N, TH_12_arc_N])
        T_diff_N.extend([TH_01_diff_N, TH_11_diff_N, TH_12_diff_N])
        T_disc_N.extend([TH_01_disc_N, TH_11_disc_N, TH_12_disc_N])
        T_A_N.extend([TH_01_A_N, TH_11_A_N, TH_12_A_N])

        M_ = ['01', '11', '12']
        M_label_N = ['Jan', 'Nov', 'Dec']

        index = M_label_N.index(month_name)
        print(M_label_N[index])

        sub_plots(year, wl, hours, T_c_N[index], T_arc_N[index], T_diff_N[index], T_disc_N[index], T_A_N[index], month_name=month_name, N=N)


        '''
        for i in range(len(M_label_N)):
            #plot_hourly_nor(hours, T_c_N[i], T_arc_N[i], T_diff_N[i], T_disc_N[i], year, M_label_N[i], monthly=True)
            sub_plots(year, hours, T_c_N[i], T_arc_N[i], T_diff_N[i], T_disc_N[i], T_Aurora_N=None,  N=4)
        '''

        M_plots = False

        if M_plots:
            for i in range(len(M_)):

                plt.figure()
                subcategorybar(hours, [T_arc[i], T_diff[i], T_disc[i]], ["arc. tot: %s" %sum(T_arc[i]), "diff. tot: %s"%sum(T_diff[i]), "disc. tot: %s"%sum(T_disc[i])])
                plt.title("Stats %s, %s" %(M_label[i], year))
                #plt.xticks(rotation='vertical')
                plt.xlabel("Hour of the day"); plt.ylabel("Count")
                #plt.savefig("stats/Green/b3/hour_plot_%s_%s.png" %(year, M_label[i]))


    else:
        # Year
        T_c = []; T_arc = []; T_diff = []; T_disc = []
        T_c_N = []; T_arc_N = []; T_diff_N = []; T_disc_N = []

        for i in range(len(hours)):
            T_c.append(Hours.count(hours[i]))
            T_arc.append(Hours_arc.count(hours[i]))
            T_diff.append(Hours_diff.count(hours[i]))
            T_disc.append(Hours_disc.count(hours[i]))

        T_Aurora = []
        T_Aurora_N  = []
        T_Aurora = [a + b + c for a, b, c in zip(T_arc, T_diff, T_disc)]

        tot_sum = sum(T_c+T_arc+T_diff+T_disc)
        tot_sum_a = sum(T_arc+T_diff+T_disc)

        #print(tot_sum)
        #print("aurora: ", tot_sum_a, "aurora-less: ", sum(T_c))

        use_tot_sum = True
        if use_tot_sum:

            for i in range(len(hours)):
                T_c_N.append((T_c[i]/tot_sum)*100)
                T_arc_N.append((T_arc[i]/tot_sum)*100)
                T_diff_N.append((T_diff[i]/tot_sum)*100)
                T_disc_N.append((T_disc[i]/tot_sum)*100)
                T_Aurora_N.append((T_Aurora[i]/tot_sum)*100)

        else:
            for i in range(len(hours)):
                T_c_N.append((T_c[i]/sum(T_c))*100)
                T_arc_N.append((T_arc[i]/sum(T_arc))*100)
                T_diff_N.append((T_diff[i]/sum(T_diff))*100)
                T_disc_N.append((T_disc[i]/sum(T_disc))*100)
                T_Aurora_N.append((T_Aurora[i]/tot_sum_a)*100)


        sub_plots(year, wl, hours, T_c_N, T_arc_N, T_diff_N, T_disc_N, list_arrays, T_Aurora_N, N=N)

        """
        #plot(hours, T_arc_N, 'arc', year, month=None, monthly=False)
        #plot(hours, T_diff_N, 'diffuse', year, month=None, monthly=False)
        #plot(hours, T_disc_N, 'discrete', year, month=None, monthly=False)
        #plot_hourly_nor(hours, T_c_N, T_arc_N, T_diff_N, T_disc_N, year)
        if year == '2014':
            n = 1
            subplot(2,1,n)
            plt.title('Yearly statistics for aurora/no aurora', fontsize=16)
        else:
            n = 2
            subplot(2,1,n)
            plt.xlabel("Hour of the day", fontsize=13)

        plt.plot(hours, T_c_N, '-', label='no aurora - '+year)
        #plt.ylabel("%", fontsize=13)
        plt.plot(hours, T_Aurora_N, '--', label='aurora - '+year)
        plt.ylabel("%", fontsize=13)
        plt.legend(fontsize=11)

        """


#----------------------------------------------------------------------------

def error(entry, error_dict, error):

    keys = []
    for key, value in error_dict.items():
        keys.append(key)

    if str(math.trunc(float(entry.solarwind['Speed, km/s']))) not in keys:
        pass
    else:
        old_score = error_dict[str(math.trunc(float(entry.solarwind['Speed, km/s'])))]  # key
        new_score = error
        edit = [old_score[0]+new_score, old_score[1]+1]
        error_dict.update({str(math.trunc(float(entry.solarwind['Speed, km/s']))): edit})


def aurora_Speed_stats(container, bins, year_='2014', year=False):

    input = 'Speed, km/s'
    input_err = 'Speed, km/s, SD'

    # Lists to add Speed values
    a_less = []
    arc = []
    diff = []
    disc = []
    all = []
    aurora = []

    # Dictionaries to add Speed error values
    a_less_err = {}
    arc_err = {}
    diff_err = {}
    disc_err = {}
    all_err = {}
    aurora_err = {}
    init_vals = [0,0]

    for i in range(len(bins)):
        a_less_err[str(int(bins[i]))] = init_vals
        arc_err[str(int(bins[i]))] = init_vals
        diff_err[str(int(bins[i]))] = init_vals
        disc_err[str(int(bins[i]))] = init_vals
        all_err[str(int(bins[i]))] = init_vals
        aurora_err[str(int(bins[i]))] = init_vals

    count99 = 0
    count99_aless = 0

    if year:
        for entry in container:
            if entry.timepoint[:4] == year_:
                if entry.label == LABELS[0]:
                    if float(entry.solarwind[input]) != 99999.9:
                        a_less.append(float(entry.solarwind[input]))
                        all.append(float(entry.solarwind[input]))
                        err = float(entry.solarwind[input_err])
                        error(entry, a_less_err, err)
                        error(entry, all_err, err)
                        #a_less_err.append(float(entry.solarwind[input_err]))
                    else:
                        count99_aless += 1
                elif entry.label == LABELS[1]:
                    if float(entry.solarwind[input]) != 99999.9:
                        arc.append(float(entry.solarwind[input]))
                        all.append(float(entry.solarwind[input]))
                        aurora.append(float(entry.solarwind[input]))
                        err = float(entry.solarwind[input_err])
                        error(entry, arc_err, err)
                        error(entry, all_err, err)
                        error(entry, aurora_err, err)
                    else:
                        count99 += 1
                elif entry.label == LABELS[2]:
                    if float(entry.solarwind[input]) != 99999.9:
                        diff.append(float(entry.solarwind[input]))
                        all.append(float(entry.solarwind[input]))
                        aurora.append(float(entry.solarwind[input]))
                        err = float(entry.solarwind[input_err])
                        error(entry, diff_err, err)
                        error(entry, all_err, err)
                        error(entry, aurora_err, err)
                    else:
                        count99 += 1
                elif entry.label == LABELS[3]:
                    if float(entry.solarwind[input]) != 99999.9:
                        disc.append(float(entry.solarwind[input]))
                        all.append(float(entry.solarwind[input]))
                        aurora.append(float(entry.solarwind[input]))
                        err = float(entry.solarwind[input_err])
                        error(entry, disc_err, err)
                        error(entry, all_err, err)
                        error(entry, aurora_err, err)
                    else:
                        count99 += 1

    else:
        for entry in container:
            if entry.label == LABELS[0]:
                if float(entry.solarwind[input]) != 99999.9:
                    a_less.append(float(entry.solarwind[input]))
                    all.append(float(entry.solarwind[input]))
                    err = float(entry.solarwind[input_err])
                    error(entry, a_less_err, err)
                    error(entry, all_err, err)
                    #a_less_err.append(float(entry.solarwind[input_err]))
                else:
                    count99_aless += 1
            elif entry.label == LABELS[1]:
                if float(entry.solarwind[input]) != 99999.9:
                    arc.append(float(entry.solarwind[input]))
                    all.append(float(entry.solarwind[input]))
                    aurora.append(float(entry.solarwind[input]))
                    err = float(entry.solarwind[input_err])
                    error(entry, arc_err, err)
                    error(entry, all_err, err)
                    error(entry, aurora_err, err)
                else:
                    count99 += 1
            elif entry.label == LABELS[2]:
                if float(entry.solarwind[input]) != 99999.9:
                    diff.append(float(entry.solarwind[input]))
                    all.append(float(entry.solarwind[input]))
                    aurora.append(float(entry.solarwind[input]))
                    err = float(entry.solarwind[input_err])
                    error(entry, diff_err, err)
                    error(entry, all_err, err)
                    error(entry, aurora_err, err)
                else:
                    count99 += 1
            elif entry.label == LABELS[3]:
                if float(entry.solarwind[input]) != 99999.9:
                    disc.append(float(entry.solarwind[input]))
                    all.append(float(entry.solarwind[input]))
                    aurora.append(float(entry.solarwind[input]))
                    err = float(entry.solarwind[input_err])
                    error(entry, disc_err, err)
                    error(entry, all_err, err)
                    error(entry, aurora_err, err)
                else:
                    count99 += 1

    print("Nr of entries (aurora) with 99999.9 value:    ", count99)
    print("Nr of entries (no aurora) with 99999.9 value: ", count99_aless)

    return a_less, arc, diff, disc, all, aurora, a_less_err, arc_err, diff_err, disc_err, all_err, aurora_err

def sub_plots_Speed(year, wl, a_less, arc, diff, disc, bins, bins_width, error_list, T_Aurora_N=None, month_name=None,  N=4):

    if year[:4] == '2020':
        shape = 'C3*-'
    elif year[:4] == '2014':
        shape = '.-'
    elif year[:4] == '2016':
        shape = 'o-'
    else:
        shape = 'x-'

    plt.suptitle(r'SW speed distribution for all classes [{}]'.format(wl) +'\n'+ r'Bins = {} km/s'.format(bins_width), fontsize=26)

    #subplot(N,1,1)
    subplot(N/2,N/2,1)
    plt.title(r'arc', fontsize = 22)
    heights, bins = np.histogram(arc, bins=bins, density=True)
    heights = heights / heights.sum()
    centers = 0.5*(bins[1:] + bins[:-1])
    err = error_list[1]
    err = err[1:]
    #plt.plot(bins[:-1], heights, shape, label=year)
    plt.plot(centers, heights, shape, label=year)
    plt.errorbar(centers, heights, xerr=err, fmt='none', ecolor='k', elinewidth=0.7, capsize=2)
    plt.ylabel(r"Occurence (norm.)", fontsize=22)
    plt.ylim(-0.01, 0.265)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.legend(fontsize=20, loc='upper left', bbox_to_anchor=(0.345, 1.25),
          fancybox=True, shadow=True, ncol=4)   # 13, bbox_to_anchor=(0.675, 1.2)
    #plt.legend(fontsize=13, shadow=True) #bbox_to_anchor = (1.05, 0.95),
    #plot(hours, T_arc_N, 'arc', year, month=None, monthly=False)

    #subplot(N,1,2)
    subplot(N/2,N/2,2)
    plt.title(r'diffuse', fontsize = 22)
    heights, bins = np.histogram(diff, bins=bins, density=True)
    heights = heights / heights.sum()
    centers = 0.5*(bins[1:] + bins[:-1])
    err = error_list[2]
    err = err[1:]
    #plt.plot(bins[:-1], heights, shape, label=year)
    plt.plot(centers, heights, shape, label=year)
    plt.errorbar(centers, heights, xerr=err, fmt='none', ecolor='k', elinewidth=0.7, capsize=2)
    #plt.plot(bins[:-1], heights, shape, label="Diffuse")
    #plt.ylabel(r"Occurence (norm.)", fontsize=22)    # 15
    plt.ylim(-0.01, 0.265)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    #plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True)
    #plt.legend(fontsize=13, shadow=True) #bbox_to_anchor = (1.05, 0.95),

    #subplot(N,1,3)
    subplot(N/2,N/2,3)
    plt.title(r'discrete', fontsize = 22)
    heights, bins = np.histogram(disc, bins=bins, density=True)
    heights = heights / heights.sum()
    centers = 0.5*(bins[1:] + bins[:-1])
    err = error_list[3]
    err = err[1:]
    #plt.plot(bins[:-1], heights, shape, label=year)
    plt.plot(centers, heights, shape, label=year)
    plt.errorbar(centers, heights, xerr=err, fmt='none', ecolor='k', elinewidth=0.7, capsize=2)
    #plt.plot(bins[:-1], heights, shape, label="Diffuse")
    plt.ylabel(r"Occurrence (norm.)", fontsize=22)    # 15
    plt.ylim(-0.01, 0.265)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19) # 11
    #plt.legend(fontsize=13, shadow=True) #bbox_to_anchor = (1.05, 0.95),
    plt.xlabel(r"Solar wind speed [km/s]", fontsize=22)    # 15
    # r'W1 disk and central $\pm2^\circ$ subtracted'

    #subplot(N,1,4)
    subplot(N/2,N/2,4)
    plt.title(r'no aurora', fontsize = 22)   # 15
    heights, bins = np.histogram(a_less, bins=bins, density=True)
    heights = heights / heights.sum()
    centers = 0.5*(bins[1:] + bins[:-1])
    err = error_list[0]
    err = err[1:]
    #plt.plot(bins[:-1], heights, shape, label=year)
    plt.plot(centers, heights, shape, label=year)
    plt.errorbar(centers, heights, xerr=err, fmt='none', ecolor='k', elinewidth=0.7, capsize=2)
    #plt.plot(bins[:-1], heights, shape, label="Diffuse")
    #plt.ylabel(r"Occurence (norm.)", fontsize=22)    # 15
    plt.ylim(-0.01, 0.265)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19) # 11
    #plt.legend(fontsize=13, shadow=True) #bbox_to_anchor = (1.05, 0.95),

    plt.xlabel(r"Solar wind speed [km/s]", fontsize=22)    # 15

    plt.subplots_adjust(top=0.83)


def Speed_stats(container, year, wl, year_plot=False, subplot=False):
    print('Speed stats')
    print(year)

    #bins = np.linspace(200, 800, 21)   # 30 km/s intervals
    #bins = np.linspace(200, 800, 13)    # 50 km/s intervals
    bins = np.linspace(200, 800, 31)    # 20 km/s intervals
    bins_width = bins[1]-bins[0]

    #plt.figure(figsize=(18, 9)) # bredde, hoyde. 11, 8

    if year == '2014-2020':
        a_less, arc, diff, disc, all, aurora, a_less_err, arc_err, diff_err, disc_err, all_err, aurora_err = aurora_Speed_stats(container, bins)
        #a_less_Day, arc_Day, diff_Day, disc_Day = aurora_Bz_stats(container_D)
        #a_less_Night, arc_Night, diff_Night, disc_Night = aurora_Bz_stats(container_N)
    else:
        a_less, arc, diff, disc, all, aurora, a_less_err, arc_err, diff_err, disc_err, all_err, aurora_err = aurora_Speed_stats(container, bins, year, True)
        #a_less_Day, arc_Day, diff_Day, disc_Day = aurora_Bz_stats(container_D, year, True)
        #a_less_Night, arc_Night, diff_Night, disc_Night = aurora_Bz_stats(container_N, year, True)

    #a_less = [a_less_Day, a_less_Night]
    #arc = [arc_Day, arc_Night]
    #diff = [diff_Day, diff_Night]
    #disc = [disc_Day, disc_Night]

    list = [a_less_err, arc_err, diff_err, disc_err, all_err, aurora_err]
    list_arrays = []

    for i in range(len(list)):
        error_std = []
        for key, value in list[i].items():

            if value[1] == 0:
                error_std.append(0)
            else:
                error_std.append(value[0]/value[1])

        error_hist = np.array(error_std)
        list_arrays.append(error_hist)
    #print(list_arrays)

    if year == '2020':
        shape = '*-'
    elif year == '2014':
        shape = '.-'
    elif year == '2016':
        shape = 'o-'
    elif year == '2018':
        shape = 'x-'
    else:
        shape = 'k--'

    if year_plot:

        plt.title(r'Solar wind speed distribution [{}]'.format(wl) +'\n'+ r'Bins = {} km/s'.format(bins_width), fontsize=16)
        heights, bins = np.histogram(all, bins=bins, density=True) #, density=True
        unity_density = heights / heights.sum()
        #unity_density = heights
        centers = 0.5*(bins[1:] + bins[:-1])
        #plt.plot(bins[:-1], unity_density, shape, label=year)
        err = list_arrays[4]
        err = err[1:]
        plt.plot(centers, unity_density, shape, label=year)
        plt.errorbar(centers, unity_density, xerr=err, fmt='none', ecolor='k', elinewidth=0.7, capsize=2)
        plt.legend(fontsize=13)
        plt.xlabel(r"Solar wind speed [km/s]", fontsize=14)
        plt.ylabel(r"Occurrence (norm.)", fontsize=14)    # 15
        plt.tight_layout()
        print(unity_density.sum())

    elif subplot:
        sub_plots_Speed(year, wl, a_less, arc, diff, disc, bins, bins_width, list_arrays)

    else:

        plt.figure(figsize=(7.4, 4.0))

        plt.title(r'SW speed distribution for {} [{}]'.format(year, wl) +'\n'+ r'Bins = {} km/s'.format(bins_width), fontsize=16)

        heights, bins = np.histogram(arc, bins=bins, density=True)
        heights = heights / heights.sum()
        centers = 0.5*(bins[1:] + bins[:-1])
        err = list_arrays[1]
        err = err[1:]
        plt.plot(centers, heights, 'o-', label="arc")
        plt.errorbar(centers, heights, xerr=err, fmt='none', ecolor='k', elinewidth=0.7, capsize=2)
        #plt.plot(bins[:-1], heights, 'o-', label="Arc")

        heights, bins = np.histogram(diff, bins=bins, density=True)
        heights = heights / heights.sum()
        centers = 0.5*(bins[1:] + bins[:-1])
        err = list_arrays[2]
        err = err[1:]
        plt.plot(centers, heights, '*-', label="diffuse")
        plt.errorbar(centers, heights, xerr=err, fmt='none', ecolor='k', elinewidth=0.7, capsize=2)
        #plt.plot(bins[:-1], heights, '*-', label="Diffuse")

        heights, bins = np.histogram(disc, bins=bins, density=True)
        heights = heights / heights.sum()
        centers = 0.5*(bins[1:] + bins[:-1])
        err = list_arrays[3]
        err = err[1:]
        plt.plot(centers, heights, '.-', label="discrete")
        plt.errorbar(centers, heights, xerr=err, fmt='none', ecolor='k', elinewidth=0.7, capsize=2)
        #plt.plot(bins[:-1], heights, '.-', label="Discrete")

        heights, bins = np.histogram(a_less, bins=bins, density=True) #, density=True
        heights = heights / heights.sum()
        centers = 0.5*(bins[1:] + bins[:-1])
        err = list_arrays[0]
        err = err[1:]
        plt.plot(centers, heights, 'x-', label="no aurora")
        plt.errorbar(centers, heights, xerr=err, fmt='none', ecolor='k', elinewidth=0.7, capsize=2)
        #plt.plot(bins[:-1], heights, 'x-', label="No aurora")

        plt.legend(fontsize=13)
        plt.xlabel(r"Solar wind speed [km/s]", fontsize=14)
        plt.ylabel(r"Occurrence (norm.)", fontsize=14)    # 15
        #plt.ylim(-0.01, 0.30)

        plt.tight_layout()



if __name__ == "__main__":

    LABELS = ['aurora-less', 'arc', 'diffuse', 'discrete']

    Green = True
    if Green:

        # All 4 years, jan+nov+dec
        predicted_G_Full = r'C:\Users\Krist\Documents\ASI_json_files\AuroraFull_G_omni_mean_predicted_efficientnet-b3.json'
        container_Full = DatasetContainer.from_json(predicted_G_Full)
        print("len container Full: ", len(container_Full))
        wl = [r'5577 Å', r'5577 Å', r'5577 Å', r'5577 Å']

    else:
        # All 4 years, jan+nov+dec
        predicted_R_Full = r'C:\Users\Krist\Documents\ASI_json_files\AuroraFull_R_omni_mean_predicted_efficientnet-b3.json'
        container_Full = DatasetContainer.from_json(predicted_R_Full)
        print("len container Full: ", len(container_Full))
        wl = [r'6300 Å', r'6300 Å', r'6300 Å', r'6300 Å']

    # Average spped pr hour, for each aurora class.
    # Plot for each year, month?

    # Make hour plots (line plots)
    # Yearly
    N = 4
    #plt.figure(figsize=(8, 11)) # bredde, hoyde
    #Hour_subplot(container=container_Full, year="2014", wl=wl[0], N=N, month=False)
    #Hour_subplot(container=container_Full, year="2016", wl=wl[0], N=N, month=False)
    #Hour_subplot(container=container_Full, year="2018", wl=wl[0], N=N, month=False)
    #Hour_subplot(container=container_Full, year="2020", wl=wl[0], N=N, month=False)
    '''
    if Green:
        plt.savefig("stats/Green/b3/yearly_hour_plot_{}.png".format(N), bbox_inches="tight")
    else:
        plt.savefig("stats/Red/b3/yearly_hour_plot_R_{}.png".format(N), bbox_inches="tight")
    '''
    #plt.show()





    def Speed_distribution_plots(path, wl, Green=True, years_plot=False, subplot=False):

        years = [r'2014', r'2016', r'2018', r'2020', r'2014-2020']
        if years:
            years = [r'2014', r'2016', r'2018', r'2020']
            #years = [r'2014', r'2016', r'2020']
        path_ASI = r'C:\Users\Krist\Documents\ASI_json_files'

        if years_plot:

            plt.figure(figsize=(7.4, 4.0)) # 6.4, 4.8
            for i in range(len(years)):
                Speed_stats(container_Full, years[i], wl, year_plot=True)
                #Speed_stats(container_D, container_N, years[i], wl)
                #plt.savefig(path+r'Speed_plot_{}.png'.format(years[i]), bbox_inches="tight")
            if Green:
                plt.savefig(path+r'SW_speed_year_plot.png', bbox_inches="tight")
            else:
                plt.savefig(path+r'SW_speed_year_plot_R.png', bbox_inches="tight")

        elif subplot:
            plt.figure(figsize=(16, 13)) # bredde, hoyde. 11, 8
            #plt.figure(figsize=(18, 11)) # bredde, hoyde. 11, 8
            for i in range(len(years)):
                Speed_stats(container_Full, years[i], wl, subplot=True)
            if Green:
                plt.savefig(path+r'SW_speed_subplot.png', bbox_inches="tight")
                #plt.savefig(path+r'Density_speed_subplot_141620.png', bbox_inches="tight")
            else:
                plt.savefig(path+r'SW_speed_subplot_R.png', bbox_inches="tight")
                #plt.savefig(path+r'Density_speed_subplot_141620_R.png', bbox_inches="tight")

        else:
            for i in range(len(years)):
                Speed_stats(container_Full, years[i], wl)
                #Speed_stats(container_D, container_N, years[i], wl)
                #plt.savefig(path+r'Speed_plot_{}.png'.format(years[i]), bbox_inches="tight")
                if Green:
                    plt.savefig(path+r'SW_speed_plot_{}.png'.format(years[i]), bbox_inches="tight")
                else:
                    plt.savefig(path+r'SW_speed_plot_{}_R.png'.format(years[i]), bbox_inches="tight")
                #plt.show()



    if Green:
        print('Green')
        Speed_distribution_plots(path=r'stats/Green/b3/Speed/', wl=wl[0], years_plot=True)
        Speed_distribution_plots(path=r'stats/Green/b3/Speed/', wl=wl[0])
        Speed_distribution_plots(path=r'stats/Green/b3/Speed/', wl=wl[0], subplot=True)
    else:
        print('Red')
        Speed_distribution_plots(path=r'stats/Red/b3/Speed/', wl=wl[0], Green=False, years_plot=True)
        Speed_distribution_plots(path=r'stats/Red/b3/Speed/', wl=wl[0], Green=False)
        Speed_distribution_plots(path=r'stats/Red/b3/Speed/', wl=wl[0], Green=False, subplot=True)
