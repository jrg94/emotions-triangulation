import csv
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

META_DATA = "table_0"
GAZE_CALIBRATION_POINTS_DETAILS = "table_1"
GAZE_CALIBRATION_SUMMARY_DETAILS = "table_2"
DATA = "table_1"
STIMULUS_NAME = "StimulusName"
AVERAGE_FIX_DUR = "Average Fixation Duration"
FIXATION_DURATION = "FixationDuration"
FIXATION_SEQUENCE = "FixationSeq"
FIXATION_X = "FixationX"
FIXATION_Y = "FixationY"
FIXATION_COUNTS = "Fixation Counts"
SPATIAL_DENSITY = "Spatial Density"
FIXATION_TIME = "Fixation Time"
QUADRANTS = "Quadrants"
TIMESTAMP = "Timestamp"
TIME_FORMAT = "%Y%m%d_%H%M%S%f"
WINDOW = "30S"
PUPIL_LEFT = "PupilLeft"
PUPIL_RIGHT = "PupilRight"
VISUAL_SCALE = 190  # Scales the dilation dot visually


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
        f"{stimulus} ({stimulus_data['Name'].iloc[0]})": {
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
    fixation_time = windowed_data.sum()[FIXATION_DURATION] / 300  # converts to a percentage assuming 30 second window
    fixation_windows = windowed_data[[FIXATION_SEQUENCE, FIXATION_X, FIXATION_Y]] 
    spatial_density = fixation_windows.apply(compute_spatial_density)
    quadrants = compute_quadrant(average_fixation_duration, unique_fixation_counts)
    frame = {
        FIXATION_COUNTS: unique_fixation_counts,
        AVERAGE_FIX_DUR: average_fixation_duration,
        SPATIAL_DENSITY: spatial_density,
        FIXATION_TIME: fixation_time,
        QUADRANTS: quadrants
    }
    return pd.DataFrame(frame)


def compute_quadrant(average_fixation_duration, fixation_counts):
    """
    Generates a list of quadrants based on correlation.

    :param average_fixation_duration: a list of mean fixation durations
    :param fixation_counts: a list of fixation counts
    :return: a list of quadrants
    """
    mean_fixation_duration_mid = (average_fixation_duration.max() + average_fixation_duration.min()) / 2
    fixation_count_mid = (fixation_counts.max() + fixation_counts.min()) / 2
    quadrants = list()
    for mean, count in zip(average_fixation_duration, fixation_counts):
        if mean > mean_fixation_duration_mid and count > fixation_count_mid:
            quadrants.append("Q1")
        elif mean <= mean_fixation_duration_mid and count > fixation_count_mid:
            quadrants.append("Q2")
        elif mean <= mean_fixation_duration_mid and count <= fixation_count_mid:
            quadrants.append("Q3")
        elif mean > mean_fixation_duration_mid and count <= fixation_count_mid:
            quadrants.append("Q4")
        else:
            quadrants.append(None)
    return quadrants


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

    fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    line_plot = ax[1]
    dilation_plot = ax[0]
    correlation_plot = ax[2]

    generate_fixation_plot(line_plot, fixation_time, window_metrics)
    generate_pupil_circle_plot(dilation_plot, fixation_time, pupil_dilation)
    generate_correlation_plot(correlation_plot, window_metrics)

    plt.sca(line_plot)
    fig.suptitle(f'{stimulus}: {participant}')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def generate_pupil_circle_plot(axes: plt.Axes, time: np.array, dilation: pd.DataFrame):
    """
    A handy method for generating the pupil dilation plot.

    :param axes: the axes to plot on
    :param time: the numpy array of times
    :param dilation: the dataframe of pupil data
    :return: None
    """
    plt.sca(axes)
    windowed_data = dilation.resample(WINDOW, on=TIMESTAMP)

    axes.set_title("Pupil Dilation Over Time")
    axes.set_xlabel("Time (minutes)", fontsize="large")

    try:  # a patch for now
        # left
        left_pupil = windowed_data.mean()[PUPIL_LEFT]
        category_left = ["left"] * len(time)
        normalized_left_pupil = ((left_pupil - left_pupil.min()) / (left_pupil.max() - left_pupil.min()) + .1) * VISUAL_SCALE

        # right
        right_pupil = windowed_data.mean()[PUPIL_RIGHT]
        category_right = ["right"] * len(time)
        normalized_right_pupil = ((right_pupil - right_pupil.min()) / (right_pupil.max() - right_pupil.min()) + .1) * VISUAL_SCALE

        # average
        avg_pupil = windowed_data.mean()[[PUPIL_LEFT, PUPIL_RIGHT]].mean(axis=1)
        category_avg = ["average"] * len(time)
        normalized_avg_pupil = ((avg_pupil - avg_pupil.min()) / (avg_pupil.max() - avg_pupil.min()) + .1) * VISUAL_SCALE

        # Pupil labels
        max_index = np.argmax(normalized_avg_pupil)
        min_index = np.argmin(normalized_avg_pupil)
        edge_avg = ["green"] * len(time)
        edge_avg[max_index] = "black"
        edge_avg[min_index] = "black"
        axes.annotate(f'{right_pupil.max():.2f} mm', (time[max_index], category_avg[max_index]),
                      textcoords="offset points", ha='center', va='center', xytext=(0, 15))
        axes.annotate(f'{right_pupil.min():.2f} mm', (time[min_index], category_avg[min_index]),
                      textcoords="offset points", ha='center', va='center', xytext=(0, -15))

        axes.scatter(time, category_left, s=normalized_left_pupil)
        axes.scatter(time, category_avg, s=normalized_avg_pupil, edgecolors=edge_avg, color="green")
        axes.scatter(time, category_right, s=normalized_right_pupil)

        if len(time) != 0:
            plt.xticks(np.arange(0, time.max() + 1, step=2))  # Force two-minute labels
    except AttributeError:
        pass


def generate_pupil_dilation_plot(axes: plt.Axes, time: np.array, dilation: pd.DataFrame):
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


def generate_correlation_plot(axes: plt.Axes, window_metrics: pd.DataFrame):
    plt.sca(axes)

    min_fix_dur = window_metrics[AVERAGE_FIX_DUR].min()
    max_fix_dur = window_metrics[AVERAGE_FIX_DUR].max()
    min_fix_count = window_metrics[FIXATION_COUNTS].min()
    max_fix_count = window_metrics[FIXATION_COUNTS].max()

    axes.set_title("Overview of Participant Visual Effort")

    x_mid = (max_fix_dur + min_fix_dur) / 2
    y_mid = (window_metrics[FIXATION_COUNTS].max() + min_fix_count) / 2

    # Background quadrant colors
    bars = axes.bar(
        x=(x_mid, min_fix_dur, min_fix_dur, x_mid),
        height=y_mid,
        bottom=(y_mid, y_mid, min_fix_count, min_fix_count),
        width=x_mid - min_fix_dur,
        color=get_quadrant_color_map().values(),
        align='edge',
        alpha=.3,
        zorder=1
    )

    # Vertical line for quadrants
    axes.plot(
        [x_mid, x_mid],
        [min_fix_count, max_fix_count],
        color="black",
        zorder=2
    )

    # Horizontal line for quadrants
    axes.plot(
        [min_fix_dur, max_fix_dur],
        [y_mid, y_mid],
        color="black",
        zorder=3
    )

    legend = plt.legend(
        bars,
        (
            "Slow Comprehension\nComplexity\nImportance\nConfusion",
            "Fast Comprehension\nSimplicity\nImportance\nPossible confusion",
            "Fast Comprehension\nSimplicity\nUnimportance",
            "Slow Comprehension\nComplexity\nUnimportance"
        ),
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )

    for lh in legend.legendHandles:
        lh.set_alpha(1)

    axes.scatter(window_metrics[AVERAGE_FIX_DUR], window_metrics[FIXATION_COUNTS], zorder=4)
    axes.set_xlabel("Mean Fixation Duration (ms)", fontsize="large")
    axes.set_ylabel("Fixation Count", fontsize="large")


def generate_fixation_plot(axes: plt.Axes, time: np.array, window_metrics: pd.DataFrame):
    """
    A handy method for generating the fixation plot.

    :param window_metrics: a set of windowed data points
    :param axes: the axes to plot on
    :param time: the numpy array of times
    :return: None
    """
    plt.sca(axes)

    axes.set_title("Eye Gaze Metrics with Visual Effort Transitions Over Time")

    # Fixation count plot
    color = 'tab:red'
    axes.plot(time, window_metrics[FIXATION_COUNTS], color=color, linewidth=2)
    axes.set_xlabel("Time (minutes)", fontsize="large")
    axes.set_ylabel("Fixation Count", color=color, fontsize="large")
    axes.tick_params(axis='y', labelcolor=color)

    # Mean fixation duration plot
    ax2 = axes.twinx()
    color = 'tab:cyan'
    ax2.plot(time, window_metrics[AVERAGE_FIX_DUR], color=color, linewidth=2)
    ax2.set_ylabel("Mean Fixation Duration (ms)", color=color, fontsize="large")
    ax2.tick_params(axis='y', labelcolor=color)

    # Spatial density plot
    ax3 = axes.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    color = 'tab:green'
    ax3.plot(time, window_metrics[SPATIAL_DENSITY], color=color, linewidth=2)
    ax3.set_ylabel("Spatial Density", color=color, fontsize="large")
    ax3.tick_params(axis="y", labelcolor=color)

    # Fixation time plot
    ax4 = axes.twinx()
    ax4.spines["right"].set_position(("axes", 1.2))
    color = 'tab:purple'
    ax4.plot(time, window_metrics[FIXATION_TIME], color=color, linewidth=2)
    ax4.set_ylabel("Fixation Time (%)", color=color, fontsize="large")
    ax4.tick_params(axis="y", labelcolor=color)

    # Background quadrants
    colors = get_quad_colors(window_metrics[QUADRANTS])
    axes.bar(time, window_metrics[FIXATION_COUNTS].max(), alpha=.3, width=.5, color=colors)

    plt.xticks(np.arange(0, time.max() + 1, step=2))  # Force two-minute labels


def get_quadrant_color_map():
    colors = cm.get_cmap("Pastel1").colors
    quads = {
        "Q1": colors[0],
        "Q2": colors[1],
        "Q3": colors[2],
        "Q4": colors[3]
    }
    return quads


def get_quad_colors(column):
    return [get_quadrant_color_map().get(value, "white") for value in column]


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
        if stimulus == "MATLAB Session":
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
