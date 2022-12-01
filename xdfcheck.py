import pyxdf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def get_stream_channel(data, stream_name, channel_name='', channel_number=-1, channel_numbers=[]):
    # Get stream time series
    streams = {data[i]["info"]["name"][0]: i for i in range(len(data))}
    if not stream_name in streams:
        raise NameError(f"Channel {stream_name} not found!")
    stream = data[streams[stream_name]]
    print(f"Getting {stream_name} data...")
    t = stream['time_stamps']

    # Get channel data
    if channel_number > -1:
        channel_index = channel_number
        print(f"...and using channel {channel_number}")

    elif len(channel_numbers) > 0:
        channel_index = channel_numbers
        print(f"...and using channels {channel_numbers}")

    elif channel_name != '':
        for index, (key, value) in enumerate(stream["info"]["desc"][0].items()):
            if key == channel_name:
                channel_index = index
        if not 'channel_index' in locals():
            raise NameError(f"Channel {channel_name} not found!")
        else:
            print(f"...and using channel called '{channel_name}'")
    else:
        print(f"...and using default channel 0")
        channel_index = 0
    d = np.transpose(stream['time_series'])[channel_index]

    return t, d


def create_csv_by_millisecond(time_data_streams, sampling_rate=0.1, supplied_start_time=-1):
    # Create columns of pandas output data frame
    df_columns = ['time']
    start_time = 9999999999999
    end_time = -9999999999999
    for stream in time_data_streams:
        start_time = min(stream[1].min(), start_time)
        end_time = max(stream[1].max(), end_time)
        df_columns.append(stream[0])

    # If start time is given use this one
    if supplied_start_time > -1:
        start_time = supplied_start_time

    # Create output dataframe
    streams_dataframe = pd.DataFrame(columns=df_columns)
    for i in range(int(round(start_time / sampling_rate, 0)), int(round(end_time / sampling_rate, 0))):
        data = [float(i) * sampling_rate]
        for stream in time_data_streams:
            data.append(np.interp(i * sampling_rate, stream[1], stream[2]))
        streams_dataframe = pd.concat([streams_dataframe, pd.DataFrame([data], columns=streams_dataframe.columns)], ignore_index=True)

    return streams_dataframe


# Set parameters
filename = '/Users/johnmadrid/Local/data/nbp2022/wd_room/recorded/01_room1_010121.xdf'
start_time_offset = 0 # start at defined point or later?
time_window = 10 # in seconds
use_photo_diode = False
use_head_eye_tracking = not use_photo_diode
csv_create = False
csv_sampling_rate = 0.01 # sampling rate is in seconds (0.1 = 100ms sampling rate)

# Load xdf file
data, _ = pyxdf.load_xdf(filename, synchronize_clocks=True, handle_clock_resets=True, dejitter_timestamps=True)

# Extract streams w/ channels
time_unity, data_unity = get_stream_channel(data, 'Visual', channel_name='displayStatus')
time_image, data_image = get_stream_channel(data, 'ImageInfo', channel_name='imageName')
if use_photo_diode:
    time_photo, data_photo = get_stream_channel(data, 'openvibeSignal', channel_number=64)
    data_photo = -data_photo
if use_head_eye_tracking:
    time_brainsignal, data_brainsignal = get_stream_channel(data, 'openvibeSignal', channel_numbers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    time_headtracking, data_headtracking = get_stream_channel(data, 'HeadTracking', channel_numbers=[0, 1, 2, 3, 4, 5])
    time_eyetracking_world, data_eyetracking_world = get_stream_channel(data, 'EyeTrackingWorld', channel_numbers=[0, 1, 2, 3, 4, 5, 6])
    # time_eyetracking_local, data_eyetracking_local = get_stream_channel(data, 'EyeTrackingLocal', channel_numbers=[0, 1, 2, 3, 4, 5])

# Calculate start and end time -- use latest start time to make sure all streams have started already
shift_time = max(time_unity.min(),
                 time_image.min(),
                 time_photo.min() if use_photo_diode else 0,
                 time_brainsignal.min() if use_head_eye_tracking else 0,
                 time_headtracking.min() if use_head_eye_tracking else 0,
                 # time_eyetracking_local.min() if use_head_eye_tracking else 0,
                 time_eyetracking_world.min() if use_head_eye_tracking else 0) # starting point (s)
start_time = 0 + start_time_offset
end_time = start_time + time_window # ending point (s)
print(f"Displaying between time {round(shift_time, 4)} and {round(shift_time + (end_time-start_time), 4)}")
print(f"Total time displayed: {round(time_window, 2)} seconds")

# Set to event time with start_time = 0
time_unity = time_unity - shift_time
time_image = time_image - shift_time
if use_photo_diode:
    time_photo = time_photo - shift_time
if use_head_eye_tracking:
    time_brainsignal = time_brainsignal - shift_time
    time_headtracking = time_headtracking - shift_time
    time_eyetracking_world = time_eyetracking_world - shift_time
    # time_eyetracking_local = time_eyetracking_local - shift_time

# Create a sampled data set and save as CSV
if csv_create and use_photo_diode:
    extracted_streams = [["photodiode", time_photo, pd.Series(data_photo).rolling(window=12).max()],
                         ["unity_state", time_unity, data_unity]]
    streams_df = create_csv_by_millisecond(extracted_streams, csv_sampling_rate, start_time)
    streams_df.to_csv(f"{filename[:-4]}-sampling_rate_{csv_sampling_rate}.csv", index=False)

# Now plot the selected range to compare eeg vs unity values
nplots = 2 if use_photo_diode else (4 if use_head_eye_tracking else 1)  # number of subplots
fig, axes = plt.subplots(nrows=nplots, ncols=1, figsize=(15, nplots * 5))

# Unity stream: lsl
sns.scatterplot(x=time_unity, y=data_unity, ax=axes[0], hue=data_unity)
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Unity (markers)")
axes[0].set_title("Unity")
axes[0].set_xlim(start_time, end_time)
axes[0].set_ylim(data_unity.min()-1, data_unity.max() + 1)

# photodiode stream
if use_photo_diode:
    sns.lineplot(x=time_photo, y=data_photo, ax=axes[1])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("EEG (sensor data)")
    axes[1].set_xlim(start_time, end_time)
    axes[1].set_ylim(0, 35000)
    axes[1].set_title("Photodiode")
    axes[1].legend(["Photodiode"], loc="upper right")

if use_head_eye_tracking:
    data_headtracking_subset = data_headtracking.transpose()[:, [3, 4]].transpose()
    data_headtracking_merged = np.concatenate((time_headtracking.reshape(1, len(time_headtracking)), data_headtracking_subset)).transpose()
    data_headtracking_merged = data_headtracking_merged[data_headtracking_merged[:, 0] >= start_time]
    data_headtracking_merged = data_headtracking_merged[data_headtracking_merged[:, 0] <= end_time]
    data_headtracking_pd = pd.DataFrame(data_headtracking_merged, columns=['Time', 'Direction X', 'Direction Y']).set_index('Time')
    sns.lineplot(data=data_headtracking_pd, ax=axes[1])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Head (sensor data)")
    axes[1].set_xlim(start_time, end_time)
    axes[1].set_ylim(data_headtracking_subset.min(), data_headtracking_subset.max())
    axes[1].set_title("Head Tracking")
    axes[1].legend(loc="upper right")

    data_eyetracking_world_subset = data_eyetracking_world.transpose()[:, [3, 4, 5]].transpose()
    data_eyetracking_world_merged = np.concatenate((time_eyetracking_world.reshape(1, len(time_eyetracking_world)), data_eyetracking_world_subset)).transpose()
    data_eyetracking_world_merged = data_eyetracking_world_merged[data_eyetracking_world_merged[:, 0] >= start_time]
    data_eyetracking_world_merged = data_eyetracking_world_merged[data_eyetracking_world_merged[:, 0] <= end_time]
    data_eyetracking_world_pd = pd.DataFrame(data_eyetracking_world_merged, columns=['Time', 'Direction X', 'Direction Y', 'Direction Z']).set_index('Time')
    sns.lineplot(data=data_eyetracking_world_pd, ax=axes[2])
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Eye World (sensor data)")
    axes[2].set_xlim(start_time, end_time)
    axes[2].set_ylim(data_eyetracking_world_subset.min(), data_eyetracking_world_subset.max())
    axes[2].set_title("Eye Tracking World")
    axes[2].legend(loc="upper right")

    data_brainsignal_merged = np.concatenate((time_brainsignal.reshape(1, len(time_brainsignal)), data_brainsignal)).transpose()
    data_brainsignal_merged = data_brainsignal_merged[data_brainsignal_merged[:, 0] >= start_time]
    data_brainsignal_merged = data_brainsignal_merged[data_brainsignal_merged[:, 0] <= end_time]
    data_brainsignal_pd = pd.DataFrame(data_brainsignal_merged, columns=['Time', 'Ch0', 'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8', 'Ch9']).set_index('Time')
    sns.lineplot(data=data_brainsignal_pd, ax=axes[3])
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Brain Signal (sensor data)")
    axes[3].set_xlim(start_time, end_time)
    axes[3].set_ylim(data_brainsignal.min(), data_brainsignal.max())
    axes[3].set_title("Brain Signal (channels 0-9)")
    axes[3].legend(loc="upper right")

    # data_eyetracking_local_subset = data_eyetracking_local.transpose()[:, [3, 4, 5]].transpose()
    # data_eyetracking_local_merged = np.concatenate((time_eyetracking_local.reshape(1, len(time_eyetracking_local)), data_eyetracking_local_subset)).transpose()
    # data_eyetracking_local_merged = data_eyetracking_local_merged[data_eyetracking_local_merged[:, 0] >= start_time]
    # data_eyetracking_local_merged = data_eyetracking_local_merged[data_eyetracking_local_merged[:, 0] <= end_time]
    # data_eyetracking_local_pd = pd.DataFrame(data_eyetracking_local_merged, columns=['Time', 'Direction X', 'Direction Y', 'Direction Z']).set_index('Time')
    # sns.lineplot(data=data_eyetracking_local_pd, ax=axes[3])
    # axes[3].set_xlabel("Time (s)")
    # axes[3].set_ylabel("Eye Local (sensor data)")
    # axes[2].set_ylim(data_eyetracking_local_subset.min(), data_eyetracking_local_subset.max())
    # axes[3].set_ylim(0, 1)
    # axes[3].set_title("Eye Tracking Local ")
    # axes[3].legend(loc="upper right")

fig.tight_layout()
plt.show()
