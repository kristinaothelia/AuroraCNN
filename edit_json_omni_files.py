from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from pylab import *
# -----------------------------------------------------------------------------

def remove(container):
    # Remove data for Feb, Mar, Oct in a container
    counter = 0
    for i in range(len(container)):
        i -= counter
        if container[i].timepoint[5:7] == '02' \
        or container[i].timepoint[5:7] == '03' \
        or container[i].timepoint[5:7] == '10':
            del container[i]
            counter += 1
    print('removed images from container: ', counter)
    print('new container len: ', len(container))

    container.to_json(r'C:\Users\Krist\Documents\ASI_json_files\Aurora_G_omni_mean_predicted_efficientnet-b3_cut.json')

def split(container, to_json, nightside=False):

    day_start = 6
    day_end = 17

    if nightside:
        print('nightside')
        counter = 0

        for i in range(len(container)):

            i -= counter

            if int(container[i].timepoint[-8:-6]) >= day_start and int(container[i].timepoint[-8:-6]) <= day_end:

                del container[i]
                counter += 1
        print('removed images from container: ', counter)
        print('new container len: ', len(container))

        container.to_json(to_json)
        return container

    else:
        print('dayside')
        counter = 0

        for i in range(len(container)):

            i -= counter

            if int(container[i].timepoint[-8:-6]) >= day_start and int(container[i].timepoint[-8:-6]) <= day_end:
                #counter += 1
                continue

            else:
                #print(container[i].timepoint[-8:-6])
                del container[i]
                counter += 1

        #print('removed images from container: ', counter)
        print('new container len: ', len(container))

        container.to_json(to_json)
        return container

def split_container(container, to_json, nightside=False):

    container = split(container, to_json, nightside)

    times = []
    for entry in container:
        if entry.timepoint[-8:-6] not in times:
            times.append(entry.timepoint[-8:-6])
    print(times)

    return container

def Make_files(json_file, json_day, json_night):
    container_Full = DatasetContainer.from_json(json_file)
    print(len(container_Full))
    container_D = split_container(container_Full, json_day)

    container_Full = DatasetContainer.from_json(predicted_R_Full)
    container_N = split_container(container_Full, json_night, True)


if __name__ == "__main__":

    # All 4 years, jan+nov+dec
    predicted_G_Full = r'C:\Users\Krist\Documents\ASI_json_files\AuroraFull_G_omni_mean_predicted_efficientnet-b3.json'
    predicted_R_Full = r'C:\Users\Krist\Documents\ASI_json_files\AuroraFull_R_omni_mean_predicted_efficientnet-b3.json'

    to_json_R_night = r'C:\Users\Krist\Documents\ASI_json_files\AuroraFull_R_omni_mean_predicted_efficientnet-b3_nighttime.json'
    to_json_R_day = r'C:\Users\Krist\Documents\ASI_json_files\AuroraFull_R_omni_mean_predicted_efficientnet-b3_daytime.json'

    to_json_G_night = r'C:\Users\Krist\Documents\ASI_json_files\AuroraFull_G_omni_mean_predicted_efficientnet-b3_nighttime.json'
    to_json_G_day = r'C:\Users\Krist\Documents\ASI_json_files\AuroraFull_G_omni_mean_predicted_efficientnet-b3_daytime.json'

    # If container include images from months Feb, Mar, Oct
    #remove(predicted_G_Full)

    # Split original data set into sets containing nightside and dayside data
    # Red
    Make_files(predicted_R_Full, to_json_R_day, to_json_R_night)

    # Green
    Make_files(predicted_G_Full, to_json_G_day, to_json_G_night)
