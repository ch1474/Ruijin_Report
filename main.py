import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
import pandas as pd
import json
import glob
import os


def get_epochs(path):
    '''
    Returns the Pupil Epoch and System Epoch timestamps

            Parameters:
                    path (str): Path to recording directory
            Returns:
                    timestamps (tuple str): Timestamps read from the file
    '''

    f = open(path + "info.player.json")

    data = json.load(f)  # returns JSON object as a dictionary

    f.close()

    return (data['start_time_synced_s'], data['start_time_system_s'])

def convert_pupil_timestamp(input_timestamp, start_time_system, start_time_synched):
    '''
    Returns the converted Pupil Epoch timestamp to Unix Epoch

            Parameters:
                    input_timestamp (int): Time delta measured from Pupil Epoch
                    start_time_system (int): Unix Epoch at start time of the recording
                    start_time_synched (int): Pupil Epoch at start time of the recording

            Returns:
                    correlated_timestamp (datetime obj): A format that allows it to be easily converted to wall time
    '''
    offset = start_time_system - start_time_synched

    return dt.datetime.fromtimestamp(input_timestamp + offset)

def get_leap_timestamps(path):
    '''
    Returns the system timestamp and leap timestamp as a dictionary by recording id.

            Parameters:
                    path (str): The path to the patient folder
            Returns:
                    leap_timestamps_dict (datetime obj): A format that allows it to be easily converted to wall time
    '''
    leap_timestamps_path = glob.glob(path + r"\*_leap_timestamps.csv")[0]


    with open(leap_timestamps_path) as f:
        data = [x.split(",") for x in f.readlines()]


    leap_timestamps = {}

    for dat in data:
        # Leap timestamp is measured in microseconds
        # System timestamp is the seconds since epoch
        leap_timestamps[dat[0]] = (float(dat[1]) * 10 ** -6, float(dat[2]))

        print((float(dat[1]) * 10 ** -6) - (float(dat[2])))

    return leap_timestamps

def get_tone_timestamps(path):
    '''
    Returns the system timestamp and leap timestamp as a dictionary by recording id.

            Parameters:
                    path (str): The path to the patient folder.
            Returns:
                    leap_timestamps_dict (datetime obj): A format that allows it to be easily converted to wall time
    '''

    tone_timestamps_path = glob.glob(path + r"\*_tone_timestamps.csv")[0]


    with open(tone_timestamps_path) as f:
        data = [x.split(",") for x in f.readlines()]

    tone_timestamps = {}

    for dat in data:
        # Leap timestamp is measured in microseconds
        # System timestamp is the seconds since epoch

        datetime_object  = dt.datetime.fromisoformat(dat[-1].replace("\n" ,""))
        epoch = dt.datetime.utcfromtimestamp(0)
        tone_timestamps[dat[0]] = (datetime_object - epoch).total_seconds()

    return tone_timestamps


def plot_leap_hand_visible_time(leap_df):
    leap_copy = leap_df.copy(deep=True)
    leap_copy['timestamp'] = (leap_copy['timestamp'] - leap_copy['timestamp'].min()) * (10 ** -6)

    grouped_leap_df = leap_copy.groupby(by=['hand_type', 'hand_id'])

    left = []
    right = []

    for name, group in grouped_leap_df:

        start_dur = (group['timestamp'].min(), group['timestamp'].max() - group['timestamp'].min())

        if 'left' in name:
            left.append(start_dur)
        else:
            right.append(start_dur)

    fig, ax = plt.subplots()

    ax.broken_barh(left, (10, 9), facecolors='tab:blue')
    ax.broken_barh(right, (20, 9), facecolors='tab:red')

    ax.set_xlabel('seconds since start of recording')
    ax.set_ylabel('Hand type')
    ax.set_yticks([15, 25])
    ax.set_yticklabels(['left', 'right'])
    ax.grid(False)

    plt.show()


def parse_path(path):
    name = path.replace("_leap.csv", "").split("\\")[-1]
    ic(name)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ic.disable()

    example_path = r"E:\Ruijin\Recording\20210621-132108_songshanying"

    recording_id = example_path.split("\\")[-1]

    leap_list = glob.glob(example_path + r"\*_leap.csv")

    leap_recordings = {}
    for x in leap_list:
        rm_path = x.replace(example_path + "\\", "")
        rm_id = rm_path.replace(recording_id + "_", "")
        leap_recordings[rm_id.replace("_leap.csv", "")] = pd.read_csv(x)

    #ic.enable()
    leap_timestamps = ic(get_leap_timestamps(example_path))

    for key in leap_recordings.keys():
        leap_system, system = leap_timestamps[key]
        ic(system, leap_system)
        leap_df = leap_recordings[key]
        leap_df['timestamp'] *= 10 ** -6
        leap_df['timestamp'] -= leap_system
        leap_df['timestamp'] += system
        ic(leap_df['timestamp'][0])
        ic(leap_recordings[key]['timestamp'][0])

    #ic.enable()

    world_timestamps = {}

    for key in leap_recordings.keys():
        pupil_recording_path = example_path + "\\" + recording_id + "_" + key + "\\000\\"
        start_time_synched, start_time_system = ic(get_epochs(pupil_recording_path))

        timestamps = np.load(pupil_recording_path + "world_timestamps.npy" )

        adjusted_timestamps = []

        for x in timestamps:
            pupil_timestamp = convert_pupil_timestamp(x, start_time_system, start_time_synched)
            epoch = dt.datetime.utcfromtimestamp(0)
            adjusted_timestamps.append((pupil_timestamp - epoch).total_seconds())

        world_timestamps[key] = np.array(adjusted_timestamps)

    ic.enable()


    for key in leap_recordings.keys():
        leap_video_df = leap_recordings[key].copy(deep=True)
        grouped_leap_df = leap_video_df.groupby(by=['hand_type'])

        for name, group in grouped_leap_df:

            group = leap_video_df.set_index('timestamp')

        world_timestamp = world_timestamps[key]
        world_timestamp_index = pd.Float64Index(world_timestamp)

        leap_timestamps = leap_video_df.index

        joint_series = leap_timestamps.append(pd.Float64Index(world_timestamp)).sort_values()

        ic(joint_series)


        ic(leap_video_df[leap_video_df.index.duplicated()])
        joint_series = joint_series.unique()
        #ic(leap_video_df['timestamp'])




        #leap_video_df = leap_video_df.reindex(joint_series)

        leap_video_df.to_csv(key + ".csv")

        break
    #file_list = ic(leap_recordings.keys()) # Files in the folder

    # 1. have all of the relevant files available
    # 2. Apply the pre-processing steps so that the data can be used together
