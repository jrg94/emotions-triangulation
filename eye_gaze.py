import csv
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

META_DATA = "table_0"
GAZE_CALIBRATION_POINTS_DETAILS = "table_1"
GAZE_CALIBRATION_SUMMARY_DETAILS = "table_2"
DATA = "table_3"
STIMULUS_NAME = "StimulusName"
AVERAGE_FIX_DUR = "Average Fixation Duration"
FIXATION_DURATION = "FixationDuration"
FIXATION_SEQUENCE = "FixationSeq"
FIXATION_X = "FixationX"
FIXATION_Y = "FixationY"
FIXATION_COUNTS = "Fixation Counts"
SPATIAL_DENSITY = "Spatial Density"
TIMESTAMP = "Timestamp"
TIME_FORMAT = "%Y%m%d_%H%M%S%f"
WINDOW = "30S"
PUPIL_LEFT = "PupilLeft"
PUPIL_RIGHT = "PupilRight"
VISUAL_SCALE = 100  # Scales the dilation dot visually


def main():
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
        participants = read_tsv_files(*paths)
        for participant in paths:
            generate_statistics(participants[participant])


def read_tsv_files(*paths) -> dict:
    """
    Reads a series of TSV files from iMotions.

    :param paths: a list of TSV paths
    :return: a dictionary of parsed files
    """
    output = dict()
    for path in paths:
        output[path] = read_tsv_file(path)
    return output


def read_tsv_file(path: str) -> dict:
    """
    Reads a single TSV file from iMotions

    :param path: a path to a TSV file
    :return: a dictionary of the TSV data in a series of tables by index
    """
    tables = dict()
    index = 0
    with open(path) as data:
        csv_reader = csv.reader(data, delimiter="\t")
        tables[f'table_{index}'] = []
        for row in csv_reader:
            if not "".join(row):  # identifies empty rows
                index += 1
                tables[f'table_{index}'] = []
            else:
                tables[f'table_{index}'].append(row)
    return tables


def clean_data(tables: dict) -> pd.DataFrame:
    """
    Assigns appropriate types to columns. For example, this method
    converts the timestamp column to the appropriate Python data type
    datetime.

    :param tables: a raw dictionary of iMotions data for a participant
    :return: a pandas DataFrame of the iMotions data
    """
    data_table = tables[DATA]
    header = data_table[0]
    data = pd.DataFrame(data_table[1:], columns=header)
    data[TIMESTAMP] = pd.to_datetime(data[TIMESTAMP], format=TIME_FORMAT)
    data[FIXATION_SEQUENCE] = pd.to_numeric(data[FIXATION_SEQUENCE])
    data[FIXATION_DURATION] = pd.to_numeric(data[FIXATION_DURATION])
    data[FIXATION_X] = pd.to_numeric(data[FIXATION_X])
    data[FIXATION_Y] = pd.to_numeric(data[FIXATION_Y])
    data[PUPIL_LEFT] = pd.to_numeric(data[PUPIL_LEFT])
    data[PUPIL_RIGHT] = pd.to_numeric(data[PUPIL_RIGHT])
    return data


def summary_report(stimulus: str, stimulus_data: pd.DataFrame) -> dict:
    """
    Generates a summary report of the data for a given stimuli.

    :param stimulus: a stimulus like MATLAB
    :param stimulus_data: the section of the data only relevant to the stimulus
    :return: a summary report as a dictionary
    """
    fixation_sequence_sans_dupes = stimulus_data.drop_duplicates(FIXATION_SEQUENCE)
    fixation_sequence: pd.Series = fixation_sequence_sans_dupes[FIXATION_SEQUENCE]
    fixation_duration: pd.Series = fixation_sequence_sans_dupes[FIXATION_DURATION]
    fixation_sequence_length = fixation_sequence.max()
    fixation_sequence_duration = fixation_duration.sum()
    seconds = fixation_sequence_duration / 1000
    fixation_duration_mean = fixation_duration.mean()
    fixation_duration_median = fixation_duration.median()
    fixation_duration_min = fixation_duration.min()
    fixation_duration_max = fixation_duration.max()
    fixation_duration_std = fixation_duration.std()
    start_date_time = stimulus_data.iloc[0][TIMESTAMP]
    end_date_time = stimulus_data.iloc[-1][TIMESTAMP]
    pupil_dilation = stimulus_data[[TIMESTAMP, PUPIL_LEFT, PUPIL_RIGHT]]
    pupil_dilation = pupil_dilation[(pupil_dilation[PUPIL_LEFT] != -1) & (pupil_dilation[PUPIL_RIGHT] != -1)]  # removes rows which have no data
    pupil_dilation_mean = pupil_dilation[PUPIL_LEFT].mean(), pupil_dilation[PUPIL_RIGHT].mean()
    pupil_dilation_median = pupil_dilation[PUPIL_LEFT].median(), pupil_dilation[PUPIL_RIGHT].median()
    pupil_dilation_min = pupil_dilation[PUPIL_LEFT].min(), pupil_dilation[PUPIL_RIGHT].min()
    pupil_dilation_max = pupil_dilation[PUPIL_LEFT].max(), pupil_dilation[PUPIL_RIGHT].max()
    return {
        f"{stimulus}": {
            "Stimulus Metrics": {
                "Start time": start_date_time,
                "End time": end_date_time,
                "Duration": end_date_time - start_date_time
            },
            "Fixation Sequence Metrics": {
                "Number of Points": fixation_sequence_length,
                "Duration (seconds)": seconds
            },
            "Fixation Duration Metrics": {
                "Mean (ms)": fixation_duration_mean,
                "Median (ms)": fixation_duration_median,
                "Standard Deviation (ms)": fixation_duration_std,
                "Minimum (ms)": fixation_duration_min,
                "Maximum (ms)": fixation_duration_max
            },
            "Pupil Metrics": {
                "Mean Left/Right (mm)": pupil_dilation_mean,
                "Median Left/Right (mm)": pupil_dilation_median,
                "Min Left/Right (mm)": pupil_dilation_min,
                "Max Left/Right (mm)": pupil_dilation_max
            }
        }
    }


def grid_index(x: int, y: int) -> int:
    """
    Given the x and y coordinates of the screen, this function returns the index of the cell
    that the point occupies. Currently, this function is hardcoded for a 10x10 grid and
    a 2560x1440 screen. If x and y coordinates are not valid, this function returns -1.
    Indices are numbered between 0 and 99.

    :param x: the x position of a pixel
    :param y: the y position of a pixel
    :return: the index of that pixel with a 10x10 grid
    """
    if not np.isnan(x) and not np.isnan(y):
        row = int(x / 2560 * 10)
        col = int(y / 1440 * 10)
        return row + col * 10
    return -1


def compute_spatial_density(df: pd.DataFrame) -> float:
    """
    Given a set of fixation points, this function returns the spatial density. In other words,
    the ratio of grid points that are occupied by fixation points.

    :param df: a dataframe containing fixation points
    :return: a ratio
    """
    points = [index for x, y in zip(df[FIXATION_X], df[FIXATION_Y]) if (index := grid_index(x, y)) >= 0]
    count = len(np.unique(points))
    return count / 100


def windowed_metrics(stimulus_data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes fixation counts and average fixation duration within some window of time.

    :param stimulus_data: the section of the data only relevant to the stimulus
    :return: fixation counts, average fixation duration (tuple)
    """
    fixation_sequence_sans_dupes = stimulus_data.drop_duplicates(FIXATION_SEQUENCE)
    windowed_data = fixation_sequence_sans_dupes.resample(WINDOW, on=TIMESTAMP)
    unique_fixation_counts = windowed_data.nunique()[FIXATION_SEQUENCE]
    average_fixation_duration = windowed_data.mean()[FIXATION_DURATION]
    fixation_windows = windowed_data[[FIXATION_SEQUENCE, FIXATION_X, FIXATION_Y]] 
    spatial_density = fixation_windows.apply(compute_spatial_density)
    frame = {
        FIXATION_COUNTS: unique_fixation_counts,
        AVERAGE_FIX_DUR: average_fixation_duration,
        SPATIAL_DENSITY: spatial_density
    }
    return pd.DataFrame(frame)


def output_summary_report(metrics: dict, depth: int = 0):
    """
    Dumps a summary report as a string.

    :param metrics: a dictionary of metrics
    :param depth: the depth of indentation
    :return: None
    """
    for k, v in metrics.items():
        indent = "\t" * depth
        if isinstance(v, dict):
            print(f'{indent}{k}')
            output_summary_report(v, depth + 1)
        else:
            print(f'{indent}{k}: {v}')


def plot_data(participant, stimulus, window_metrics: pd.DataFrame, pupil_dilation: pd.DataFrame):
    """
    Plots the fixation count and average fixation duration data.

    :param window_metrics: a set of windowed metrics for plotting
    :param participant: the name of the participant
    :param stimulus: the current stimulus used as the plot title
    :param pupil_dilation: a dataframe of pupil information
    :return: None
    """
    fixation_time = (window_metrics[FIXATION_COUNTS].index.astype(np.int64) / 10 ** 9) / 60  # Converts datetime to minutes
    fixation_time = fixation_time - fixation_time.min()  # Scales minutes back to 0

    pupil_time = (pupil_dilation[TIMESTAMP].astype(np.int64) / 10 ** 9) / 60
    #pupil_time = pupil_time - pupil_time.min()

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    top_plot = ax[0]
    bot_plot = ax[1]

    generate_fixation_plot(top_plot, fixation_time, window_metrics)
    generate_pupil_circle_plot(bot_plot, fixation_time, pupil_dilation)
    #generate_pupil_dilation_plot(bot_plot, pupil_time, pupil_dilation)

    plt.sca(top_plot)
    plt.title(f'{stimulus}: {participant}')
    fig.tight_layout()
    plt.show()


def generate_pupil_circle_plot(axes, time: np.array, dilation: pd.DataFrame):
    """
    A handy method for generating the pupil dilation plot.

    :param axes: the axes to plot on
    :param time: the numpy array of times
    :param dilation: the dataframe of pupil data
    :return: None
    """
    plt.sca(axes)
    windowed_data = dilation.resample(WINDOW, on=TIMESTAMP)

    # left
    try:  # a patch for now
        left_pupil = windowed_data.mean()[PUPIL_LEFT]
        category_left = ["left"] * len(time)
        normalized_left_pupil = (left_pupil - left_pupil.min()) / (left_pupil.max() - left_pupil.min()) * VISUAL_SCALE
        # abs(left_pupil - left_pupil.max())/abs(left_pupil.max() - left_pupil.min())

        # right
        right_pupil = windowed_data.mean()[PUPIL_RIGHT]
        category_right = ["right"] * len(time)
        normalized_right_pupil = (right_pupil - right_pupil.min()) / (right_pupil.max() - right_pupil.min()) * VISUAL_SCALE

        axes.scatter(time, category_left, s=normalized_left_pupil)
        axes.scatter(time, category_right, s=normalized_right_pupil)
    except AttributeError:
        pass


def generate_pupil_dilation_plot(axes, time: np.array, dilation: pd.DataFrame):
    """
    A handy method for generating the pupil dilation plot.

    :param axes: the axes to plot on
    :param time: the numpy array of times
    :param dilation: the dataframe of pupil data
    :return: None
    """
    plt.sca(axes)

    axes.plot(time, dilation[PUPIL_LEFT], label="Left Pupil")
    axes.plot(time, dilation[PUPIL_RIGHT], label="Right Pupil")
    axes.set_xlabel("Time (minutes)", fontsize="large")
    axes.set_ylabel("Pupil Dilation (mm)", fontsize="large")
    axes.legend()

    if len(time) != 0:
        plt.xticks(np.arange(0, time.max() + 1, step=2))  # Force two-minute labels


def generate_fixation_plot(axes, time: np.array, window_metrics: pd.DataFrame):
    """
    A handy method for generating the fixation plot.

    :param axes: the axes to plot on
    :param time: the numpy array of times
    :return: None
    """
    plt.sca(axes)

    color = 'tab:red'
    axes.plot(time, window_metrics[FIXATION_COUNTS], color=color, linewidth=2)
    axes.set_xlabel("Time (minutes)", fontsize="large")
    axes.set_ylabel("Fixation Count", color=color, fontsize="large")
    axes.tick_params(axis='y', labelcolor=color)

    ax2 = axes.twinx()

    color = 'tab:cyan'
    ax2.plot(time, window_metrics[AVERAGE_FIX_DUR], color=color, linewidth=2)
    ax2.set_ylabel("Mean Fixation Duration (ms)", color=color, fontsize="large")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.xticks(np.arange(0, time.max() + 1, step=2))  # Force two-minute labels


def generate_statistics(tables: dict):
    """
    Generates various statistics for a participant.

    :param tables: a participant table
    :return: None
    """
    df = clean_data(tables)
    participant = df.iloc[0]["Name"]
    stimuli = df[STIMULUS_NAME].unique()
    for stimulus in stimuli:
        stimulus_filter = df[STIMULUS_NAME] == stimulus
        stimulus_data = df[stimulus_filter]
        report = summary_report(stimulus, stimulus_data)
        output_summary_report(report)
        window_metrics = windowed_metrics(stimulus_data)
        pupil_dilation = stimulus_data[[TIMESTAMP, PUPIL_LEFT, PUPIL_RIGHT]]
        pupil_dilation = pupil_dilation[(pupil_dilation[PUPIL_LEFT] != -1) & (pupil_dilation[PUPIL_RIGHT] != -1)]  # removes rows which have no data
        plot_data(participant, stimulus, window_metrics, pupil_dilation)


if __name__ == '__main__':
    main()
