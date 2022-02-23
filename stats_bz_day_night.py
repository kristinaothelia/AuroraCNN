# Plot percentage vs hour of the day, for all classes
# For pos and negative Bz separate
# For night and day?

from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from pylab import *
import math
import operator
# -----------------------------------------------------------------------------

LABELS = ['aurora-less', 'arc', 'diffuse', 'discrete']

Green = False
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


def aurora_Bz_stats(container, label, year_='2014', day_=None, year=False, day=False):

    input = 'Bz, nT (GSM)'
    Hours_POS = []
    Hours_NEG = []

    if year:
        for entry in container:
            if entry.timepoint[:4] == year_:

                if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                    if entry.label == label:
                        if float(entry.solarwind[input]) != 9999.99 and float(entry.solarwind[input]) >= 0.0:
                            Hours_POS.append(entry.timepoint[-8:-6])
                        else:
                            Hours_NEG.append(entry.timepoint[-8:-6])

                if label == "aurora": # arc, diffuse and discrete aurora is counted as one
                    if entry.label != LABELS[0]:
                        if float(entry.solarwind[input]) != 9999.99 and float(entry.solarwind[input]) >= 0.0:
                            Hours_POS.append(entry.timepoint[-8:-6])
                        else:
                            Hours_NEG.append(entry.timepoint[-8:-6])

    elif day:
        for entry in container:
            if entry.timepoint[:10] == day_:
                print(entry.timepoint, entry.label)
                if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                    if entry.label == label:
                        if float(entry.solarwind[input]) != 9999.99 and float(entry.solarwind[input]) >= 0.0:
                            Hours_POS.append(entry.timepoint[-8:-6])
                        else:
                            Hours_NEG.append(entry.timepoint[-8:-6])

                if label == "aurora": # arc, diffuse and discrete aurora is counted as one
                    if entry.label != LABELS[0]:
                        if float(entry.solarwind[input]) != 9999.99 and float(entry.solarwind[input]) >= 0.0:
                            Hours_POS.append(entry.timepoint[-8:-6])
                        else:
                            Hours_NEG.append(entry.timepoint[-8:-6])
    else:
        for entry in container:

            if label == LABELS[0] or LABELS[1] or LABELS[2] or LABELS[3]:
                if entry.label == label:
                    if float(entry.solarwind[input]) != 9999.99 and float(entry.solarwind[input]) >= 0.0:
                        Hours_POS.append(entry.timepoint[-8:-6])
                    else:
                        Hours_NEG.append(entry.timepoint[-8:-6])

            if label == "aurora": # arc, diffuse and discrete aurora is counted as one
                if entry.label != LABELS[0]:
                    if float(entry.solarwind[input]) != 9999.99 and float(entry.solarwind[input]) >= 0.0:
                        Hours_POS.append(entry.timepoint[-8:-6])
                    else:
                        Hours_NEG.append(entry.timepoint[-8:-6])

    return Hours_POS, Hours_NEG


def omni_ting(container, year_='2014', year=False):

    a_less_NEG = []; a_less_POS = []
    arc_NEG = []; arc_POS = []
    diff_NEG = []; diff_POS = []
    disc_NEG = []; disc_POS = []

    input = 'Bz, nT (GSM)'
    input_err = 'Bz, nT (GSM), SD'
    a_less_NEG_err = []; a_less_POS_err = []

    test_p = {}
    test_n = {}

    index = 0

    if year:
        for entry in container:
            index += 1
            if float(entry.solarwind[input]) != 9999.99:
                if entry.timepoint[:4] == year_:
                    if entry.label == LABELS[0]:
                        if float(entry.solarwind[input]) != 9999.99 and float(entry.solarwind[input]) >= 0.0:
                            a_less_POS.append(float(entry.solarwind[input]))
                            a_less_POS_err.append(float(entry.solarwind[input_err]))
                            test_p[index] = list()
                            test_p[index].extend([float(entry.solarwind[input]), float(entry.solarwind[input_err])])
                            #print(test_p)
                            #exit()
                        else:
                            a_less_NEG.append(float(entry.solarwind[input]))
                            a_less_NEG_err.append(float(entry.solarwind[input_err]))
                            test_n[index] = list()
                            test_n[index].extend([float(entry.solarwind[input]), float(entry.solarwind[input_err])])
                            #print(test_p)
                            #exit()

                    elif entry.label == LABELS[1]:
                        if float(entry.solarwind[input]) != 9999.99 and float(entry.solarwind[input]) >= 0.0:
                            arc_POS.append(float(entry.solarwind[input]))
                        else:
                            arc_NEG.append(float(entry.solarwind[input]))

                    elif entry.label == LABELS[2]:
                        if float(entry.solarwind[input]) != 9999.99 and float(entry.solarwind[input]) >= 0.0:
                            diff_POS.append(float(entry.solarwind[input]))
                        else:
                            diff_NEG.append(float(entry.solarwind[input]))

                    elif entry.label == LABELS[3]:
                        if float(entry.solarwind[input]) != 9999.99 and float(entry.solarwind[input]) >= 0.0:
                            disc_POS.append(float(entry.solarwind[input]))
                        else:
                            disc_NEG.append(float(entry.solarwind[input]))

    else:
        for entry in container:
            if entry.label == LABELS[0]:
                if float(entry.solarwind[input]) != 9999.99 and float(entry.solarwind[input]) >= 0.0:
                    a_less_POS.append(float(entry.solarwind[input]))
                    a_less_POS_err.append(float(entry.solarwind[input_err]))
                else:
                    a_less_NEG.append(float(entry.solarwind[input]))
                    a_less_NEG_err.append(float(entry.solarwind[input_err]))

            elif entry.label == LABELS[1]:
                if float(entry.solarwind[input]) != 9999.99 and float(entry.solarwind[input]) >= 0.0:
                    arc_POS.append(float(entry.solarwind[input]))
                else:
                    arc_NEG.append(float(entry.solarwind[input]))

            elif entry.label == LABELS[2]:
                if float(entry.solarwind[input]) != 9999.99 and float(entry.solarwind[input]) >= 0.0:
                    diff_POS.append(float(entry.solarwind[input]))
                else:
                    diff_NEG.append(float(entry.solarwind[input]))

            elif entry.label == LABELS[3]:
                if float(entry.solarwind[input]) != 9999.99 and float(entry.solarwind[input]) >= 0.0:
                    disc_POS.append(float(entry.solarwind[input]))
                else:
                    disc_NEG.append(float(entry.solarwind[input]))

    return a_less_POS, a_less_NEG, arc_POS, arc_NEG, diff_POS, diff_NEG, disc_POS, disc_NEG, a_less_POS_err, a_less_NEG_err, test_p, test_n

def sub_plots(year, wl, hours, name, T_c_N, T_arc_N, T_diff_N, T_disc_N, T_Aurora_N=None, month_name=None,  N=4, shape='*-'):

    '''
    if year == 'All years':
        shape = '.-'
    if year == '2020':
        shape = '*-'
    elif year == '2014':
        shape = '.-'
    elif year == '2016':
        shape = 'o-'
    else:
        shape = 'x-'
    '''
    #shape = '*-'

    N = 5

    subplot(N,1,1)
    if month_name != None:
        plt.suptitle('Classification for positive and negative Bz, {} {} [{}]'.format(month_name, year, wl), fontsize=18)
        #plt.title('Statistics ({}) for all classes, {}'.format(month_name, year), fontsize=18)
    else:
        plt.suptitle('Classification for positive and negative Bz, {} [{}]'.format(year, wl), fontsize=18)
        #plt.title('Statistics for all classes, {}'.format(year), fontsize=18)

    plt.plot(hours, T_arc_N, shape, label='{}'.format(name))
    #plt.axvspan(6, 18, alpha=0.45, color='lightyellow') # 'lightskyblue'
    plt.axvspan(0, 6, alpha=0.09, color='lightskyblue') # 'lightskyblue'
    plt.axvspan(18, 23, alpha=0.09, color='lightskyblue') # 'lightskyblue'
    #plt.text(0.5, 3.5, 'nightside', fontsize = 11, color='deepskyblue')
    #plt.text(6.5, 3.5, 'dayside', fontsize = 11, color='yellow')
    #plt.text(18.5, 3.5, 'nightside', fontsize = 11, color='deepskyblue')
    plt.title('arc', fontsize=15)
    if N < 4:
        plt.ylabel(r"Occurrence", fontsize=15)
    plt.ylim(-0.1, 2.1)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    #plt.legend(fontsize=13)
    plt.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, 1.75),
          fancybox=True, shadow=True, ncol=2)
    #plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True) #, ncol=2
    #plot(hours, T_arc_N, 'arc', year, month=None, monthly=False)

    subplot(N,1,2)
    #plot(hours, T_diff_N, 'diffuse', year, month=None, monthly=False)
    plt.plot(hours, T_diff_N, shape, label='{}'.format(name))
    #plt.axvspan(6, 18, alpha=0.45, color='lightyellow') # 'lightskyblue'
    plt.axvspan(0, 6, alpha=0.095, color='lightskyblue') # 'lightskyblue'
    plt.axvspan(18, 23, alpha=0.095, color='lightskyblue') # 'lightskyblue'
    #plt.text(0.5, 3.5, 'nightside', fontsize = 11, color='deepskyblue')
    #plt.text(6.5, 3.5, 'dayside', fontsize = 11, color='yellow')
    #plt.text(18.5, 3.5, 'nightside', fontsize = 11, color='deepskyblue')
    plt.title('diffuse', fontsize=15)
    if N < 4:
        plt.ylabel(r"Occurrence", fontsize=15)
    plt.ylim(-0.1, 4.0)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    #plt.legend(fontsize=13)
    #plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True)

    subplot(N,1,3)
    #plot(hours, T_disc_N, 'discrete', year, month=None, monthly=False)
    plt.plot(hours, T_disc_N, shape, label='{}'.format(name))
    #plt.axvspan(6, 18, alpha=0.45, color='lightyellow') # 'lightskyblue'
    plt.axvspan(0, 6, alpha=0.095, color='lightskyblue') # 'lightskyblue'
    plt.axvspan(18, 23, alpha=0.095, color='lightskyblue') # 'lightskyblue'
    #plt.text(0.5, 3.5, 'nightside', fontsize = 11, color='deepskyblue')
    #plt.text(6.5, 3.5, 'dayside', fontsize = 11, color='y')
    #plt.text(18.5, 3.5, 'nightside', fontsize = 11, color='deepskyblue')
    plt.title('discrete', fontsize=15)
    if N > 4:
        plt.ylabel(r"Occurrence (norm.)", fontsize=15)
    else:
        plt.ylabel(r"Occurrence", fontsize=15)
    plt.ylim(-0.1, 4.3)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    #plt.legend(fontsize=13)
    #plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True)

    subplot(N,1,4)
    #plot(hours, T_c_N, 'no aurora', year, month=None, monthly=False, axis=True)
    plt.plot(hours, T_c_N, shape, label='{}'.format(name))
    #plt.axvspan(6, 18, alpha=0.45, color='lightyellow') # 'lightskyblue'
    plt.axvspan(0, 6, alpha=0.095, color='lightskyblue') # 'lightskyblue'
    plt.axvspan(18, 23, alpha=0.095, color='lightskyblue') # 'lightskyblue'
    #plt.text(0.5, 3.5, 'nightside', fontsize = 11, color='deepskyblue')
    #plt.text(6.5, 3.5, 'dayside', fontsize = 11, color='limegreen')
    #plt.text(18.5, 3.5, 'nightside', fontsize = 11, color='deepskyblue')
    plt.title('no aurora', fontsize=15)
    #plt.xlabel("Hour of the day", fontsize=13)
    if N < 4:
        plt.ylabel(r"Occurrence", fontsize=15)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylim(-0.1, 4.6)
    #plt.legend(fontsize=13)
    #plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True)

    if N == 5:
        subplot(N,1,5)
        #plot(hours, T_c_N, 'no aurora', year, month=None, monthly=False, axis=True)
        plt.plot(hours, T_Aurora_N, shape, label='{}'.format(name))
        #plt.axvspan(6, 18, alpha=0.45, color='lightyellow') # 'lightskyblue'
        plt.axvspan(0, 6, alpha=0.095, color='lightskyblue') # 'lightskyblue'
        plt.axvspan(18, 23, alpha=0.095, color='lightskyblue') # 'lightskyblue'
        #plt.text(0.5, 0.5, 'nightside', fontsize = 10, color='deepskyblue')
        #plt.text(6.5, 0.5, 'dayside', fontsize = 10, color='limegreen')
        #plt.text(18.5, 0.5, 'nightside', fontsize = 10, color='deepskyblue')
        plt.title('combined aurora classes', fontsize=15)
        plt.xlabel("Hour of the day", fontsize=15)
        #plt.ylabel(r"Occurrence (norm.)", fontsize=15)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        #plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True)
    else:
        plt.xlabel("Hour of the day", fontsize=15, labelpad=25)

    plt.tight_layout() #rect=[0,0,0.75,1]
    plt.subplots_adjust(top=0.85)


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

def Hour_subplot(container, wl, year_="2014", day_=None, month_name='Jan', N=4, month=False, weight=False, year=False, day=False):

    hours = Get_hours()
    Hours_POS, Hours_NEG            = aurora_Bz_stats(container=container, label="aurora-less", year_=year_, day_=day_, year=year, day=day)
    #Hours_POS_arc, Hours_NEG_arc    = aurora_Bz_stats(container=container, label="arc", year_=year_, year=year)
    #Hours_POS_diff, Hours_NEG_diff  = aurora_Bz_stats(container=container, label="diffuse", year_=year_, year=year)
    #Hours_POS_disc, Hours_NEG_disc  = aurora_Bz_stats(container=container, label="discrete", year_=year_, year=year)
    #Hours_POS_A, Hours_NEG_A        = aurora_Bz_stats(container=container, label="aurora", year_=year_, year=year)
    Hours_POS_arc, Hours_NEG_arc    = aurora_Bz_stats(container=container, label="arc", year_=year_, day_=day_, year=year, day=day)
    Hours_POS_diff, Hours_NEG_diff  = aurora_Bz_stats(container=container, label="diffuse", year_=year_, day_=day_, year=year, day=day)
    Hours_POS_disc, Hours_NEG_disc  = aurora_Bz_stats(container=container, label="discrete", year_=year_, day_=day_, year=year, day=day)
    Hours_POS_A, Hours_NEG_A        = aurora_Bz_stats(container=container, label="aurora", year_=year_, day_=day_, year=year, day=day)


    if month:
        T_c = []; T_arc = []; T_diff = []; T_disc = []
        T_c_N = []; T_arc_N = []; T_diff_N = []; T_disc_N = []
        T_A = []; T_A_N = []

        # Note, removed Mars, because of no data
        # Aurora-less
        TH_01, TH_02, TH_03, TH_10, TH_11, TH_12, \
        TH_01_N, TH_02_N, TH_10_N, TH_11_N, TH_12_N \
        = get_hour_count_per_month(TH, hours)
        # Arc
        TH_01_arc, TH_02_arc, TH_03_arc, TH_10_arc, TH_11_arc, TH_12_arc, \
        TH_01_arc_N, TH_02_arc_N, TH_10_arc_N, TH_11_arc_N, TH_12_arc_N \
        = get_hour_count_per_month(TH_arc, hours)
        # Diff
        TH_01_diff, TH_02_diff, TH_03_diff, TH_10_diff, TH_11_diff, TH_12_diff, \
        TH_01_diff_N, TH_02_diff_N, TH_10_diff_N, TH_11_diff_N, TH_12_diff_N \
        = get_hour_count_per_month(TH_diff, hours)
        # Disc
        TH_01_disc, TH_02_disc, TH_03_disc, TH_10_disc, TH_11_disc, TH_12_disc, \
        TH_01_disc_N, TH_02_disc_N, TH_10_disc_N, TH_11_disc_N, TH_12_disc_N \
        = get_hour_count_per_month(TH_disc, hours)
        # Aurora
        TH_01_A, TH_02_A, TH_03_A, TH_10_A, TH_11_A, TH_12_A, \
        TH_01_A_N, TH_02_A_N, TH_10_A_N, TH_11_A_N, TH_12_A_N \
        = get_hour_count_per_month(TH_A, hours)

        T_c.extend([TH_01, TH_02, TH_03, TH_10, TH_11, TH_12])
        T_arc.extend([TH_01_arc, TH_02_arc, TH_03_arc, TH_10_arc, TH_11_arc, TH_12_arc])
        T_diff.extend([TH_01_diff, TH_02_diff, TH_03_diff, TH_10_diff, TH_11_diff, TH_12_diff])
        T_disc.extend([TH_01_disc, TH_02_disc, TH_03_disc, TH_10_disc, TH_11_disc, TH_12_disc])
        T_A.extend([TH_01_A, TH_02_A, TH_03_A, TH_10_A, TH_11_A, TH_12_A])

        # Normalized
        T_c_N.extend([TH_01_N, TH_02_N, TH_10_N, TH_11_N, TH_12_N])
        T_arc_N.extend([TH_01_arc_N, TH_02_arc_N, TH_10_arc_N, TH_11_arc_N, TH_12_arc_N])
        T_diff_N.extend([TH_01_diff_N, TH_02_diff_N, TH_10_diff_N, TH_11_diff_N, TH_12_diff_N])
        T_disc_N.extend([TH_01_disc_N, TH_02_disc_N, TH_10_disc_N, TH_11_disc_N, TH_12_disc_N])
        T_A_N.extend([TH_01_A_N, TH_02_A_N, TH_10_A_N, TH_11_A_N, TH_12_A_N])

        M_ = ['01', '02', '03', '10', '11', '12']
        M_label = ['Jan', 'Feb', 'Mar', 'Oct', 'Nov', 'Dec']
        M_label_N = ['Jan', 'Feb', 'Oct', 'Nov', 'Dec']

        index = M_label_N.index(month_name)
        print(M_label_N[index])

        sub_plots(year, hours, T_c_N[index], T_arc_N[index], T_diff_N[index], T_disc_N[index], T_A_N[index], month_name=month_name, N=N)


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
                #plt.savefig("stats/Green/b2/hour_plot_%s_%s.png" %(year, M_label[i]))



    else:
        # Year
        T_c_POS = []; T_arc_POS = []; T_diff_POS = []; T_disc_POS = []
        T_c_NEG = []; T_arc_NEG = []; T_diff_NEG = []; T_disc_NEG = []

        T_c_N_POS = []; T_arc_N_POS = []; T_diff_N_POS = []; T_disc_N_POS = []
        T_c_N_NEG = []; T_arc_N_NEG = []; T_diff_N_NEG = []; T_disc_N_NEG = []

        for i in range(len(hours)):
            T_c_POS.append(Hours_POS.count(hours[i]))
            T_arc_POS.append(Hours_POS_arc.count(hours[i]))
            T_diff_POS.append(Hours_POS_diff.count(hours[i]))
            T_disc_POS.append(Hours_POS_disc.count(hours[i]))

            T_c_NEG.append(Hours_NEG.count(hours[i]))
            T_arc_NEG.append(Hours_NEG_arc.count(hours[i]))
            T_diff_NEG.append(Hours_NEG_diff.count(hours[i]))
            T_disc_NEG.append(Hours_NEG_disc.count(hours[i]))

        T_Aurora_POS = []
        T_Aurora_N_POS  = []
        T_Aurora_POS = [a + b + c for a, b, c in zip(T_arc_POS, T_diff_POS, T_disc_POS)]

        tot_sum_POS = sum(T_c_POS+T_arc_POS+T_diff_POS+T_disc_POS)
        tot_sum_a_POS = sum(T_arc_POS+T_diff_POS+T_disc_POS)

        T_Aurora_NEG = []
        T_Aurora_N_NEG  = []
        T_Aurora_NEG = [a + b + c for a, b, c in zip(T_arc_NEG, T_diff_NEG, T_disc_NEG)]

        tot_sum_NEG = sum(T_c_NEG+T_arc_NEG+T_diff_NEG+T_disc_NEG)
        tot_sum_a_NEG = sum(T_arc_NEG+T_diff_NEG+T_disc_NEG)

        use_tot_sum = True
        if use_tot_sum:

            for i in range(len(hours)):
                T_c_N_POS.append((T_c_POS[i]/tot_sum_POS)*100)
                T_arc_N_POS.append((T_arc_POS[i]/tot_sum_POS)*100)
                T_diff_N_POS.append((T_diff_POS[i]/tot_sum_POS)*100)
                T_disc_N_POS.append((T_disc_POS[i]/tot_sum_POS)*100)
                T_Aurora_N_POS.append((T_Aurora_POS[i]/tot_sum_POS)*100)

                T_c_N_NEG.append((T_c_NEG[i]/tot_sum_NEG)*100)
                T_arc_N_NEG.append((T_arc_NEG[i]/tot_sum_NEG)*100)
                T_diff_N_NEG.append((T_diff_NEG[i]/tot_sum_NEG)*100)
                T_disc_N_NEG.append((T_disc_NEG[i]/tot_sum_NEG)*100)
                T_Aurora_N_NEG.append((T_Aurora_NEG[i]/tot_sum_NEG)*100)

        else:
            for i in range(len(hours)):
                T_c_N_POS.append((T_c_POS[i]/sum(T_c_POS))*100)
                T_arc_N_POS.append((T_arc_POS[i]/sum(T_arc_POS))*100)
                T_diff_N_POS.append((T_diff_POS[i]/sum(T_diff_POS))*100)
                T_disc_N_POS.append((T_disc_POS[i]/sum(T_disc_POS))*100)
                T_Aurora_N_POS.append((T_Aurora_POS[i]/tot_sum_a_POS)*100)

                T_c_N_NEG.append((T_c_NEG[i]/sum(T_c_NEG))*100)
                T_arc_N_NEG.append((T_arc_NEG[i]/sum(T_arc_NEG))*100)
                T_diff_N_NEG.append((T_diff_NEG[i]/sum(T_diff_NEG))*100)
                T_disc_N_NEG.append((T_disc_NEG[i]/sum(T_disc_NEG))*100)
                T_Aurora_N_NEG.append((T_Aurora_NEG[i]/tot_sum_a_NEG)*100)

        #sub_plots(year, hours, T_c, T_arc, T_diff, T_disc, T_Aurora, N=5)
        print(T_Aurora_N_POS)
        print(T_Aurora_N_NEG)

        names = ["Bz >= 0", "Bz < 0"]

        sub_plots(year_, wl, hours, names[0], T_c_N_POS, T_arc_N_POS, T_diff_N_POS, T_disc_N_POS, T_Aurora_N_POS, shape='o-')
        sub_plots(year_, wl, hours, names[1], T_c_N_NEG, T_arc_N_NEG, T_diff_N_NEG, T_disc_N_NEG, T_Aurora_N_NEG, shape='*-')



def sub_plots_Bz(year, a_less, arc, diff, disc, neg=None, pos=None, T_Aurora_N=None, month_name=None,  N=4):

    bins = np.linspace(-25, 25, 51)

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
        plt.title('Statistics ({}) for all classes'.format(month_name), fontsize=18)
    else:
        if len(year) > 4:
            plt.title('Yearly statistics for all classes (weighted)', fontsize=18)
        else:
            plt.title('Yearly statistics for all classes. {}'.format(year[:4]), fontsize=18)


    a_heights, a_bins = np.histogram(arc[0], bins=bins, density=True)
    b_heights, b_bins = np.histogram(arc[1], bins=bins, density=True)
    plt.plot(a_bins[:-1], a_heights, '.-', label='arc - day')
    plt.plot(b_bins[:-1], b_heights, '*-', label='arc - night')

    #plt.errorbar(a_bins[:-1], a_heights, yerr=arc[2])


    #adding text inside the plot
    #plt.text(-24, 0.20, 'Bz < 0: {:.2f}%'.format(neg[0][1]), fontsize = 13, color='C0')
    #plt.text(-24, 0.17, 'Bz > 0: {:.2f}%'.format(pos[0][1]), fontsize = 13, color='C0')
    #plt.text(-24, 0.14, 'Bz < 0: {:.2f}%'.format(neg[1][1]), fontsize = 13, color='C1')
    #plt.text(-24, 0.11, 'Bz > 0: {:.2f}%'.format(pos[1][1]), fontsize = 13, color='C1')
    plt.axvline(x=0, ls='--', color='lightgrey')
    #plt.plot(hours, T_arc_N, shape, label='arc - '+year)
    plt.ylabel("%", fontsize=15)
    plt.ylim(0, 0.30)
    plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True) #, ncol=2
    #plot(hours, T_arc_N, 'arc', year, month=None, monthly=False)

    subplot(N,1,2)
    a_heights, a_bins = np.histogram(diff[0], bins=bins, density=True)
    b_heights, b_bins = np.histogram(diff[1], bins=bins, density=True)
    #plt.text(-24, 0.20, 'Bz < 0: {:.2f}%'.format(neg[0][2]), fontsize = 13, color='C0')
    #plt.text(-24, 0.17, 'Bz > 0: {:.2f}%'.format(pos[0][2]), fontsize = 13, color='C0')
    #plt.text(-24, 0.14, 'Bz < 0: {:.2f}%'.format(neg[1][2]), fontsize = 13, color='C1')
    #plt.text(-24, 0.11, 'Bz > 0: {:.2f}%'.format(pos[1][2]), fontsize = 13, color='C1')
    plt.plot(a_bins[:-1], a_heights, '.-', label='diff - day')
    plt.plot(b_bins[:-1], b_heights, '*-', label='diff - night')
    plt.axvline(x=0, ls='--', color='lightgrey')
    #plot(hours, T_diff_N, 'diffuse', year, month=None, monthly=False)
    #plt.plot(hours, T_diff_N, shape, label='diffuse - '+year)
    plt.ylabel("%", fontsize=15)
    plt.ylim(0, 0.30)
    plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True)

    subplot(N,1,3)
    a_heights, a_bins = np.histogram(disc[0], bins=bins, density=True)
    b_heights, b_bins = np.histogram(disc[1], bins=bins, density=True)
    #plt.text(-24, 0.20, 'Bz < 0: {:.2f}%'.format(neg[0][3]), fontsize = 13, color='C0')
    #plt.text(-24, 0.17, 'Bz > 0: {:.2f}%'.format(pos[0][3]), fontsize = 13, color='C0')
    #plt.text(-24, 0.14, 'Bz < 0: {:.2f}%'.format(neg[1][3]), fontsize = 13, color='C1')
    #plt.text(-24, 0.11, 'Bz > 0: {:.2f}%'.format(pos[1][3]), fontsize = 13, color='C1')
    plt.plot(a_bins[:-1], a_heights, '.-', label='disc - day')
    plt.plot(b_bins[:-1], b_heights, '*-', label='disc - night')
    plt.axvline(x=0, ls='--', color='lightgrey')
    #plot(hours, T_disc_N, 'discrete', year, month=None, monthly=False)
    #plt.plot(hours, T_disc_N, shape, label='discrete - '+year)
    plt.ylabel("%", fontsize=15)
    plt.ylim(0, 0.30)
    plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True)

    subplot(N,1,4)
    a_heights, a_bins = np.histogram(a_less[0], bins=bins, density=True)
    b_heights, b_bins = np.histogram(a_less[1], bins=bins, density=True)
    #plt.text(-24, 0.20, 'Bz < 0: {:.2f}%'.format(neg[0][0]), fontsize = 13, color='C0')
    #plt.text(-24, 0.17, 'Bz > 0: {:.2f}%'.format(pos[0][0]), fontsize = 13, color='C0')
    #plt.text(-24, 0.14, 'Bz < 0: {:.2f}%'.format(neg[1][0]), fontsize = 13, color='C1')
    #plt.text(-24, 0.11, 'Bz > 0: {:.2f}%'.format(pos[1][0]), fontsize = 13, color='C1')
    plt.plot(a_bins[:-1], a_heights, '.-', label='no aurora - day')
    plt.plot(b_bins[:-1], b_heights, '*-', label='no aurora - night')
    plt.axvline(x=0, ls='--', color='lightgrey')
    #plot(hours, T_c_N, 'no aurora', year, month=None, monthly=False, axis=True)
    #plt.plot(hours, T_c_N, shape, label='no aurora - '+year)
    #plt.xlabel("Hour of the day", fontsize=13)
    plt.ylabel("%", fontsize=15)
    plt.ylim(0, 0.30)
    plt.legend(fontsize=13, bbox_to_anchor = (1.05, 0.95), shadow=True)

    plt.xlabel("Bz value [nT]", fontsize=15)

def Bz_stats(year):
    print('Bz stats')
    plt.figure(figsize=(8, 11)) # bredde, hoyde

    split_day = False
    if split_day:
        container_D = DatasetContainer.from_json(r'C:\Users\Krist\Documents\ASI_json_files\AuroraFull_G_omni_mean_predicted_efficientnet-b3_daytime.json')
        container_N = DatasetContainer.from_json(r'C:\Users\Krist\Documents\ASI_json_files\AuroraFull_G_omni_mean_predicted_efficientnet-b3_nighttime.json')

        print('len container day:   ', len(container_D))
        print('len container night: ', len(container_N))

    if year == 'all years':
        a_less_POS, a_less_NEG, arc_POS, arc_NEG, diff_POS, diff_NEG, disc_POS, disc_NEG, a_less_POS_err, a_less_NEG_err, test_p, test_n = omni_ting(container_Full)
        #a_less_POS_D, a_less_NEG_D, arc_POS_D, arc_NEG_D, diff_POS_D, diff_NEG_D, disc_POS_D, disc_NEG_D, a_less_POS_err, a_less_NEG_err = omni_ting(container_D)
        #a_less_POS_N, a_less_NEG_N, arc_POS_N, arc_NEG_N, diff_POS_N, diff_NEG_N, disc_POS_N, disc_NEG_N, a_less_POS_err, a_less_NEG_err = omni_ting(container_N)
    else:
        a_less_POS, a_less_NEG, arc_POS, arc_NEG, diff_POS, diff_NEG, disc_POS, disc_NEG, a_less_POS_err, a_less_NEG_err, test_p, test_n = omni_ting(container_Full, year, True)
        #a_less_POS_D, a_less_NEG_D, arc_POS_D, arc_NEG_D, diff_POS_D, diff_NEG_D, disc_POS_D, disc_NEG_D, a_less_POS_err, a_less_NEG_err = omni_ting(container_D, year, True)
        #a_less_POS_N, a_less_NEG_N, arc_POS_N, arc_NEG_N, diff_POS_N, diff_NEG_N, disc_POS_N, disc_NEG_N, a_less_POS_err, a_less_NEG_err = omni_ting(container_N, year, True)

    a_less = [a_less_POS, a_less_NEG, a_less_POS_err, a_less_NEG_err]
    arc = [arc_POS, arc_NEG]
    diff = [diff_POS, diff_NEG]
    disc = [disc_POS, disc_NEG]

    #dict_items = test_p.items()
    #first = list(dict_items)[:10]

    #d = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
    #print('Original dictionary : ',test_p)
    sorted_p = dict(sorted(test_p.items(), key=operator.itemgetter(1)))
    #print('Dictionary in ascending order by value : ',sorted_p)
    #sorted_d = dict( sorted(d.items(), key=operator.itemgetter(1),reverse=True))
    #print('Dictionary in descending order by value : ',sorted_d)

    sorted_n = dict(sorted(test_n.items(), key=operator.itemgetter(1)))

    values_val_p = []
    values_err_p = []
    values_val_n = []
    values_err_n = []

    for key, value in sorted_p.items():
        values_val_p.append(math.trunc(value[0]))
        values_err_p.append(value[1])
    for key, value in sorted_n.items():
        values_val_n.append(math.trunc(value[0]))
        values_err_n.append(value[1])

    #bins = np.linspace(0, 25, 26)   # -25, 51
    bins = np.linspace(-25, 25, 51)
    print(bins)
    bins_p = np.linspace(0, 25, 26)
    bins_n = np.linspace(-25, 0, 26)

    error_list_p = []
    count_list_p = []
    for j in range(len(bins_p)):
        error = []
        count = 0
        for i in range(len(values_val_p)):

            if math.trunc(values_val_p[i]) == bins_p[j]:
                count += 1
                error.append(values_err_p[i])
        error_list_p.append(np.std(error))
        count_list_p.append(count)

    #print(error_list_p)
    #print(count_list_p)

    error_list_n = []
    count_list_n = []
    for j in range(len(bins_n)):
        error = []
        count = 0
        for i in range(len(values_val_n)):

            if math.trunc(values_val_n[i]) == bins_n[j]:
                count += 1
                error.append(values_err_n[i])
        error_list_n.append(np.std(error))
        count_list_n.append(count)

    #print(error_list_n)
    #print(count_list_n)

        #print(math.trunc(values_val[i]), values_err[i])

    #for i in range(len(bins)):
    #    print(values_val.count(bins[i]))
    figure()
    # FEEEEEEEEEEEEIL
    counts, bin_edges = np.histogram(a_less[0]+a_less[1], bins=np.linspace(-25, 25, 51), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    err = error_list_p[:-1] + error_list_n[:-1]
    plt.errorbar(bin_centers, counts, xerr=err, fmt='-o', ecolor='k')

    figure()
    counts_p,bin_edges_p = np.histogram(a_less[0], bins_p, density=True)
    counts_n,bin_edges_n = np.histogram(a_less[1], bins_n, density=True)
    bin_centers_p = (bin_edges_p[:-1] + bin_edges_p[1:])/2.
    bin_centers_n = (bin_edges_n[:-1] + bin_edges_n[1:])/2.
    err_p = error_list_p #np.random.rand(bin_centres.size)*100
    err_n = error_list_n
    plt.errorbar(bin_centers_p, counts_p, xerr=err_p[:-1], fmt='-o', ecolor='k', capsize=0.8)    #, fmt='o'
    plt.errorbar(bin_centers_n, counts_n, xerr=err_n[:-1], fmt='-o', ecolor='k')

    plt.show()
    exit()


    exit()

    '''
    a_less = [a_less_Day, a_less_Night]
    arc = [arc_Day, arc_Night]
    diff = [diff_Day, diff_Night]
    disc = [disc_Day, disc_Night]
    neg = [neg_Day, neg_Night]
    pos = [pos_Day, pos_Night]
    '''

    #a_less = []
    '''
    x = np.random.rand(1000)
    y, bin_edges = np.histogram(x, bins=10)
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])

    plt.errorbar(
        bin_centers,
        y,
        yerr = y**0.5,
        marker = '.',
        #drawstyle = 'steps-mid'
    )
    plt.show()


    a_heights, a_bins = np.histogram(a_less[0], bins=bins, density=True)
    bin_centers = 0.5*(a_bins[1:] + a_bins[:-1])
    #plt.plot()
    plt.errorbar(
        bin_centers,
        a_heights,
        yerr = y**0.5,
        marker = '.',
    )
    plt.show()
    exit()
    '''

    sub_plots_Bz(year, a_less, arc, diff, disc)

    #plt.savefig("stats/Green/b3/yearly_Bz_plot_{}.png".format(year), bbox_inches="tight")
    plt.show()

# Make
#Bz_stats(year='2014')
#Bz_stats(year='2016')
#Bz_stats(year='2018')
#Bz_stats(year='2020')
#Bz_stats(year='All years')

#exit()

def Bz_split(container, year_="all years", day_=None, month=False, year=False, day=False):

    #plt.figure(figsize=(8, 8)) # bredde, hoyde
    plt.figure(figsize=(8, 11))
    #Hour_subplot(container=container, wl='5577 Å', year_=year_, month=month, year=year)
    if Green:
        Hour_subplot(container=container, wl='5577 Å', year_=year_, day_=day_, month=month, year=year, day=day)
    else:
        Hour_subplot(container=container, wl='6300 Å', year_=year_, day_=day_, month=month, year=year, day=day)


    #Hour_subplot(container=container, wl='6300 Å', year_=year_, month=month, year=year)
    plt.text(1.5, -1.8, 'nightside', fontsize = 13, color='deepskyblue')
    plt.text(10.9, -1.8, 'dayside', fontsize = 13, color='limegreen')
    plt.text(19.1, -1.8, 'nightside', fontsize = 13, color='deepskyblue')
    if Green:
        plt.savefig('stats/Green/b3/TEST_{}.png'.format(year_), bbox_inches="tight")
    else:
        plt.savefig('stats/Red/b3/TEST_{}_R.png'.format(year_), bbox_inches="tight")
    #plt.show()

Bz_split(container=container_Full)
#Bz_split(container=container_Full, year_='2014', year=True)
#Bz_split(container=container_Full, year_='2016', year=True)
#Bz_split(container=container_Full, year_='2018', year=True)
#Bz_split(container=container_Full, year_='2020', year=True)

# Test for a single day: (2016-01-11 has data for all 24 hours) (Plot title etc are not edited)
#Bz_split(container=container_Full, day_='2016-01-11', day=True)
exit()


plt.figure(figsize=(8, 11)) # bredde, hoyde
#plt.figure(figsize=(8, 14))
Hour_subplot(container=container_Full, month=False, year=False)
#Hour_subplot(container=container_Full, year_='2016', month=False, year=True)
#Hour_subplot(container=container_Full, year_='2020', month=False, year=True)
#Hour_subplot(container=container_Full, year_="2014", month=False, year=True)

plt.savefig("stats/Green/b2/TEST.png", bbox_inches="tight")
plt.show()
