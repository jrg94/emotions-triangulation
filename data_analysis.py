import csv
import logging
import sys
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
from scipy.signal import find_peaks

META_DATA = "table_0"
GAZE_CALIBRATION_POINTS_DETAILS = "table_1"
GAZE_CALIBRATION_SUMMARY_DETAILS = "table_2"
DATA = "table_1"

# Data column names
STIMULUS_NAME = "StimulusName"
FIXATION_DURATION = "FixationDuration"
FIXATION_SEQUENCE = "FixationSeq"
FIXATION_X = "FixationX"
FIXATION_Y = "FixationY"
TIMESTAMP = "Timestamp"
MOUSE_EVENT = "MouseEvent"
GSR_RAW = "GSR RAW (no units) (Shimmer)"
GSR_KILOHMS = "GSR CAL (kOhms) (Shimmer)"
GSR_MICROSIEMENS = "GSR CAL (ÂµSiemens) (Shimmer)"
KEY_CODE = "KeyCode"

# Analysis column names
AVERAGE_FIX_DUR = "Average Fixation Duration"
FIXATION_COUNTS = "Fixation Counts"
SPATIAL_DENSITY = "Spatial Density"
FIXATION_TIME = "Fixation Time"
QUADRANTS = "Quadrants"
CLICK_STREAM = "Click Stream"
RANGE_CORRECT_EDA = "range_corrected_eda"

TIME_FORMAT = "%Y%m%d_%H%M%S%f"
WINDOW = "20S"
PUPIL_LEFT = "PupilLeft"
PUPIL_RIGHT = "PupilRight"
VISUAL_SCALE = 100  # Scales the dilation dot visually


# DATA LOADING -----------------------------------------------------------------------


def read_data_files(*paths) -> dict:
    """
    Reads a series of TSV files from iMotions.

    :param paths: a list of TSV paths
    :return: a dictionary of parsed files
    """
    output = dict()
    for path in paths:
        logging.info("-" * 50)
        logging.info(f"Loading {path} as a dictionary")
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
    logging.info("Converting dictionary to DataFrame")
    data_table = tables[DATA]
    header = data_table[0]
    data = pd.DataFrame(data_table[1:], columns=header)
    data[TIMESTAMP] = pd.to_datetime(data[TIMESTAMP], format=TIME_FORMAT)
    data = data.set_index(TIMESTAMP)
    data[FIXATION_SEQUENCE] = pd.to_numeric(data[FIXATION_SEQUENCE])
    data[FIXATION_DURATION] = pd.to_numeric(data[FIXATION_DURATION])
    data[FIXATION_X] = pd.to_numeric(data[FIXATION_X])
    data[FIXATION_Y] = pd.to_numeric(data[FIXATION_Y])
    data[PUPIL_LEFT] = pd.to_numeric(data[PUPIL_LEFT])
    data[PUPIL_RIGHT] = pd.to_numeric(data[PUPIL_RIGHT])
    data[GSR_RAW] = pd.to_numeric(data[GSR_RAW])
    data[GSR_KILOHMS] = pd.to_numeric(data[GSR_KILOHMS])
    data[GSR_MICROSIEMENS] = pd.to_numeric(data[GSR_MICROSIEMENS])
    data[[MOUSE_EVENT, KEY_CODE]] = data[
        [MOUSE_EVENT, KEY_CODE]
    ].replace(r'^\s*$', np.NAN, regex=True)
    data[MOUSE_EVENT] = pd.Categorical(data[MOUSE_EVENT])
    data[KEY_CODE] = pd.Categorical(data[KEY_CODE])
    return data


# DATA ANALYSIS ---------------------------------------------------------------


def analyze_data(tables: dict):
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
            logging.info("Dumping Summary Report")
            output_summary_report(report)
            plot_data(participant, stimulus, stimulus_data, "Overview")
    del df


def plot_data(participant, stimulus, stimulus_data: pd.DataFrame, segment: str):
    """
    Given a set of data, a participant, and their stimulus, this function will plot various forms analyses as figures.

    :param participant: the name of the participant
    :param stimulus: the current stimulus used as the plot title
    :param stimulus_data: all raw data
    :param segment: the unit of analysis
    :return: None
    """

    figures: List[plt.Figure] = [
        plot_eye_gaze_data(stimulus, participant, stimulus_data),
        plot_pupil_data(stimulus, participant, stimulus_data),
        plot_eda_data(stimulus, participant, stimulus_data),
        plot_click_stream_data(stimulus, participant, stimulus_data)
    ]

    for fig in figures:
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_file(fig, participant, segment)
        plt.close(fig)


def plot_click_stream_data(stimulus: str, participant: str, stimulus_data: pd.DataFrame) -> plt.Figure:
    """
    Plots the click stream data on a figure.

    :param stimulus: the raw stimulus name
    :param participant: the participant name
    :param stimulus_data: the raw stimulus data
    :return: the resulting figure
    """

    # Setup figure
    fig_click, ax_click = plt.subplots(1, 1, figsize=(12, 6))
    fig_click.suptitle(f'{stimulus}: {participant}')
    fig_click.canvas.set_window_title("Click Stream Analysis")
    click_stream_plot = ax_click

    # Plot
    generate_click_stream_plot(click_stream_plot, stimulus_data)

    return fig_click


def plot_eda_data(stimulus: str, participant: str, stimulus_data: pd.DataFrame) -> plt.Figure:
    """
    Plots all the EDA related data.

    :param stimulus: the name of the stimulus
    :param participant: the name of the participant
    :param stimulus_data: the raw stimulus data
    :return: the resulting figure
    """

    # Setup figure
    fig_eda, ax_eda = plt.subplots(2, 2, figsize=(14, 8))
    fig_eda.suptitle(f'{stimulus}: {participant}')
    fig_eda.canvas.set_window_title("EDA Analysis")
    gsr_inverse_plot = ax_eda[0][0]
    gsr_range_corrected_plot = ax_eda[0][1]
    gsr_range_corrected_mean_plot = ax_eda[1][0]
    gsr_peaks_plot = ax_eda[1][1]

    # Quick analysis
    gsr_us = stimulus_data[GSR_MICROSIEMENS]
    range_corrected_eda = (gsr_us - gsr_us.max()).abs() / abs(gsr_us.max() - gsr_us.min())
    indices = find_peaks(range_corrected_eda)[0]
    peaks = [0] * len(range_corrected_eda)
    for index in indices:
        peaks[index] = 1
    stimulus_data = stimulus_data.assign(
        range_corrected_eda=range_corrected_eda,
        peaks=peaks
    )

    # Plot
    generate_gsr_inverse_plot(gsr_inverse_plot, stimulus_data)
    generate_gsr_range_corrected_plot(gsr_range_corrected_plot, stimulus_data)
    generate_gsr_range_correct_means_plot(gsr_range_corrected_mean_plot, stimulus_data)
    generate_gsr_peaks_plot(gsr_peaks_plot, stimulus_data)

    return fig_eda


def plot_pupil_data(stimulus: str, participant: str, stimulus_data: pd.DataFrame) -> plt.Figure:
    """
    Plots all pupil related data.

    :param stimulus: the name of the stimulus
    :param participant: the name of the participant
    :param stimulus_data: the raw stimulus data
    :return: the resulting figure
    """

    # Setup figure
    fig_dilation, ax_dilation = plt.subplots(2, 1, figsize=(12, 8))
    fig_dilation.suptitle(f'{stimulus}: {participant}')
    fig_dilation.canvas.set_window_title("Pupil Analysis")
    dilation_plot = ax_dilation[0]
    raw_dilation_plot = ax_dilation[1]

    # Plot
    generate_pupil_circle_plot(dilation_plot, stimulus_data)
    generate_pupil_dilation_plot(raw_dilation_plot, stimulus_data)

    return fig_dilation


def plot_eye_gaze_data(stimulus: str, participant: str, stimulus_data: pd.DataFrame) -> plt.Figure:
    """
    Plots all eye gaze related data.

    :param stimulus: the name of the stimulus
    :param participant: the name of the participant
    :param stimulus_data: the raw stimulus data
    :return: the resulting figure
    """

    # Setup data
    window_metrics = windowed_metrics(stimulus_data)
    fixation_time = convert_date_to_time(window_metrics.index)

    # Setup figure
    fig_fixation, ax_fixation = plt.subplots(3, 1, figsize=(14, 10))
    fig_fixation.suptitle(f'{stimulus}: {participant}')
    fig_fixation.canvas.set_window_title("Eye Gaze Analysis")
    line_plot = ax_fixation[1]
    correlation_plot = ax_fixation[0]
    aux_plot = ax_fixation[2]

    # Plot
    generate_fixation_plot(line_plot, fixation_time, window_metrics)
    generate_correlation_plot(correlation_plot, window_metrics)
    generate_auxiliary_eye_gaze_plot(aux_plot, fixation_time, window_metrics)

    return fig_fixation


def generate_gsr_range_correct_means_plot(axes: plt.Axes, stimulus_data: pd.DataFrame):
    """
    Plots range corrected means over two minute windows.

    :param axes: the axes to plot on
    :param stimulus_data: the raw stimulus data
    :return: None
    """
    plt.sca(axes)

    windowed_data = stimulus_data.resample("2min").mean()[:15]
    time = convert_date_to_time(windowed_data.index)

    axes.set_title("Range-Corrected GSR Means Over Two-Minute Windows")
    axes.set_xlabel("Time (minutes)", fontsize="large")
    axes.set_ylabel("Range-Corrected GSR (dimensionless)")
    axes.set_ylim(0, 1)
    set_windowed_x_axis(axes)
    axes.bar(time, windowed_data[RANGE_CORRECT_EDA], width=2, align="edge", edgecolor="black")


def generate_gsr_peaks_plot(axes: plt.Axes, stimulus_data: pd.DataFrame):
    """
    Plots range-corrected peaks over two-minute windows.

    :param axes: the axes to plot on
    :param stimulus_data: the raw stimulus data
    :return: None
    """
    plt.sca(axes)

    windowed_data = stimulus_data.resample("2min").sum()[:15]
    time = convert_date_to_time(windowed_data.index)

    axes.set_title("Range-Corrected GSR Peaks Over Two-Minute Windows")
    axes.set_xlabel("Time (minutes)", fontsize="large")
    axes.set_ylabel("Peak Count")
    set_windowed_x_axis(axes)
    axes.bar(time, windowed_data["peaks"], width=2, align="edge", edgecolor="black")


def generate_gsr_range_corrected_plot(axes: plt.Axes, stimulus_data: pd.DataFrame):
    """
    Plots the range-corrected EDA according to Villanueva (2018)—page 424.

    :param axes: the axes to plot on
    :param stimulus_data: the stimulus data
    :return: None
    """

    plt.sca(axes)

    time = convert_date_to_time(stimulus_data.index)
    range_corrected_gsr = stimulus_data[RANGE_CORRECT_EDA]

    axes.set_title("Range-Corrected GSR Over Time")
    axes.set_xlabel("Time (minutes)", fontsize="large")
    axes.set_ylabel("Range-Corrected GSR (dimensionless)")
    axes.set_ylim(0, 1)
    set_windowed_x_axis(axes)
    axes.plot(time, range_corrected_gsr)


def generate_gsr_inverse_plot(axes: plt.Axes, stimulus_data: pd.DataFrame):
    """
    Plots Galvanic Skin Response (GSR) data (aka EDA) using the inverse units (e.g. kOhms & microSiemens).

    :param axes: the axis to plot on
    :param stimulus_data: the data to use for plotting
    :return: None
    """
    plt.sca(axes)

    # Setup data
    time = convert_date_to_time(stimulus_data.index)

    axes.set_title("GSR Over Time")
    set_windowed_x_axis(axes)
    axes.set_ylabel("GSR (µS)", fontsize="large")
    axes.plot(time, stimulus_data[GSR_MICROSIEMENS], linewidth=2)


def generate_pupil_circle_plot(axes: plt.Axes, stimulus_data: pd.DataFrame):
    """
    A handy method for generating the pupil dilation plot.

    :param stimulus_data: the raw stimulus data for analysis
    :param axes: the axes to plot on
    :return: None
    """
    plt.sca(axes)

    dilation = stimulus_data[[PUPIL_LEFT, PUPIL_RIGHT]]
    dilation = dilation[(dilation[PUPIL_LEFT] != -1) & (dilation[PUPIL_RIGHT] != -1)]
    windowed_data_mean = dilation.resample(WINDOW).mean()
    time = convert_date_to_time(windowed_data_mean.index)

    # left
    left_pupil = windowed_data_mean[PUPIL_LEFT]
    category_left = ["left"] * len(time)
    normalized_left_pupil = normalize_column(left_pupil)

    # right
    right_pupil = windowed_data_mean[PUPIL_RIGHT]
    category_right = ["right"] * len(time)
    normalized_right_pupil = normalize_column(right_pupil)

    # average
    avg_pupil = windowed_data_mean[[PUPIL_LEFT, PUPIL_RIGHT]].mean(axis=1)
    category_avg = ["average"] * len(time)
    normalized_avg_pupil = normalize_column(avg_pupil)

    # Pupil labels
    label_pupils(merge_pupil_data(normalized_left_pupil, left_pupil, time, category_left), axes, "C0", (0, 15))
    label_pupils(merge_pupil_data(normalized_avg_pupil, avg_pupil, time, category_avg), axes, "C1", (0, 15))
    label_pupils(merge_pupil_data(normalized_right_pupil, right_pupil, time, category_right), axes, "C2", (0, -15))

    # Clean up plot
    axes.set_title("Mean Pupil Dilation Over Time")
    axes.set_xlabel("Time (minutes)", fontsize="large")
    set_windowed_x_axis(axes)


def generate_pupil_edge_colors(column: pd.Series, color: str) -> Tuple[int, int, list]:
    """
    A helper function for generating min/max edge colors for scatter plot.

    :param column: a series of numeric data points
    :param color: the color of the edge
    :return:
    """
    column_list = column.tolist()
    max_index: int = column_list.index(max(column_list))
    min_index: int = column_list.index(min(column_list))
    edge_colors = [color] * len(column)
    edge_colors[max_index] = "black"
    edge_colors[min_index] = "black"
    return min_index, max_index, edge_colors


def generate_pupil_dilation_plot(axes: plt.Axes, stimulus_data: pd.DataFrame):
    """
    A handy method for generating the pupil dilation plot.

    :param stimulus_data: the raw stimulus data to analyze
    :param axes: the axes to plot on
    :return: None
    """
    plt.sca(axes)

    # Data analysis
    dilation = stimulus_data[[PUPIL_LEFT, PUPIL_RIGHT]]
    dilation = dilation[(dilation[PUPIL_LEFT] != -1) & (dilation[PUPIL_RIGHT] != -1)]
    time = convert_date_to_time(dilation.index)
    axes.plot(time, dilation[PUPIL_LEFT], label="Left Pupil")
    axes.plot(time, dilation[PUPIL_RIGHT], label="Right Pupil")

    # Clean up plot
    axes.set_title("Raw Pupil Dilation Over Time")
    axes.set_xlabel("Time (minutes)", fontsize="large")
    axes.set_ylabel("Pupil Dilation (mm)", fontsize="large")
    axes.legend()
    set_windowed_x_axis(axes)


def generate_correlation_plot(axes: plt.Axes, window_metrics: pd.DataFrame):
    """
    Creates plot that demonstrates correlation between fixation counts and average fixation duration.

    :param axes: the axes to plot on
    :param window_metrics: the pre-cleaned data
    :return: None
    """
    plt.sca(axes)

    min_fix_dur = window_metrics[AVERAGE_FIX_DUR].min()  # left
    max_fix_dur = window_metrics[AVERAGE_FIX_DUR].max()  # right
    min_fix_count = window_metrics[FIXATION_COUNTS].min()  # bottom
    max_fix_count = window_metrics[FIXATION_COUNTS].max()  # top

    axes.set_title("Overview of Participant Visual Effort")

    x_mid = (max_fix_dur + min_fix_dur) // 2
    y_mid = (max_fix_count + min_fix_count) // 2

    # Background quadrant colors
    bars = axes.bar(
        x=(x_mid, min_fix_dur, min_fix_dur, x_mid),
        height=y_mid - min_fix_count,
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
    axes.autoscale(tight=True)


def generate_auxiliary_eye_gaze_plot(axes: plt.Axes, time: np.array, window_metrics: pd.DataFrame):
    """
    Plots eye gaze metrics that may assist in triangulation.

    :param axes: the axes to plot on
    :param time: the time axes
    :param window_metrics: the windowed dataframe
    :return: None
    """
    plt.sca(axes)

    axes.set_title("Auxiliary Eye Gaze Metrics Over Time")

    # Spatial density plot
    color = 'tab:red'
    axes.plot(time, window_metrics[SPATIAL_DENSITY], color=color, linewidth=2)
    axes.set_xlabel("Time (minutes)", fontsize="large")
    axes.set_ylabel("Spatial Density", color=color, fontsize="large")
    axes.tick_params(axis="y", labelcolor=color)

    # Fixation time plot
    ax2 = axes.twinx()
    color = 'tab:cyan'
    ax2.plot(time, window_metrics[FIXATION_TIME], color=color, linewidth=2)
    ax2.set_ylabel("Fixation Time (%)", color=color, fontsize="large")
    ax2.tick_params(axis="y", labelcolor=color)
    set_windowed_x_axis(axes)


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

    # Background quadrants
    minutes = int(WINDOW[:-1]) / 60
    width = minutes
    colors = get_quad_colors(window_metrics[QUADRANTS])
    axes.bar(time, window_metrics[FIXATION_COUNTS].max(), alpha=.3, width=width, color=colors, align="edge")
    set_windowed_x_axis(axes)


def generate_click_stream_plot(axes: plt.Axes, stimulus_data: pd.DataFrame):
    """
    Generates a line plot of all click stream related data.

    :param stimulus_data: the raw stimulus data
    :param axes: the axes to plot on
    :return: None
    """
    plt.sca(axes)

    data = stimulus_data.groupby([pd.Grouper(freq=WINDOW), MOUSE_EVENT])[MOUSE_EVENT].count().unstack()
    click_stream = stimulus_data.resample(WINDOW)[[MOUSE_EVENT, KEY_CODE]].count()
    time = convert_date_to_time(click_stream.index)
    minutes = int(WINDOW[:-1]) / 60
    width = minutes
    axes.bar(time, click_stream[KEY_CODE].values, width=width, align="edge", label="Keyboard Events", edgecolor="black")
    accumulator = click_stream[KEY_CODE].values
    for column in data:
        axes.bar(time, data[column], width=width, align="edge", bottom=accumulator, label=column, edgecolor="black")
        accumulator += data[column]

    # Clean up plot
    axes.set_title("Click Stream Events Over Time", fontsize="large")
    axes.set_xlabel("Time (minutes)", fontsize="large")
    axes.set_ylabel("Click Stream Event Counts", fontsize="large")
    set_windowed_x_axis(axes)
    axes.legend()


def generate_mouse_event_plot(axes: plt.Axes, stimulus_data: pd.DataFrame):
    """
    Generates a mouse event plot for each of the possible mouse events.

    :param axes: the axes to plot on
    :param stimulus_data: the raw stimulus data to analyze
    :return: None
    """

    plt.sca(axes)

    # Data analysis
    data = stimulus_data.groupby([pd.Grouper(freq=WINDOW), MOUSE_EVENT])[MOUSE_EVENT].count().unstack()
    time = convert_date_to_time(data.index)
    axes.plot(time, data)

    # Clean up plot
    axes.set_title("Mouse Events Over Time", fontsize="large")
    axes.set_xlabel("Time (minutes)", fontsize="large")
    axes.set_ylabel("Mouse Event Counts", fontsize="large")
    set_windowed_x_axis(axes)
    axes.legend(labels=data.columns)


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
    start_date_time = stimulus_data.index[0]
    end_date_time = stimulus_data.index[-1]
    pupil_dilation = stimulus_data[[PUPIL_LEFT, PUPIL_RIGHT]]
    pupil_dilation = pupil_dilation[(pupil_dilation[PUPIL_LEFT] != -1) & (pupil_dilation[PUPIL_RIGHT] != -1)]
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


# HELPER FUNCTIONS --------------------------------------------------------------------------------


def plot_segments(stimulus: str, participant: str, stimulus_data: pd.DataFrame):
    """
    A helper function for generating segment plots.

    :param stimulus: the stimulus to analyze
    :param participant: the participant name
    :param stimulus_data: the raw stimulus data
    :return: None
    """
    start = stimulus_data.index[0]
    for i in range(15):
        end = start + pd.Timedelta("2min")
        chunk = stimulus_data.loc[start: end]
        plot_data(participant, stimulus, chunk, f"Segment {i + 1:02}")
        start = end


def set_windowed_x_axis(axes: plt.Axes):
    """
    A helper function for showing markers on windowed data.
    In particular, we show two-minute segments with the major
    indicator and the proper window size with the minor indicator

    :param axes:
    :return:
    """
    seconds = int(WINDOW[:-1])
    axes.xaxis.set_major_locator(MultipleLocator(2))
    axes.xaxis.set_minor_locator(MultipleLocator(seconds / 60))
    axes.autoscale(tight=True, axis="x")


def convert_date_to_time(date: pd.Series) -> pd.Series:
    """
    A helper function which converts a series of dates to a series of times in minutes.

    :param date: a series of dates
    :return: a series of times in minutes
    """
    time = (date.astype(np.int64) / 10 ** 9) / 60
    time = time - time.min()
    return time


def normalize_column(column: pd.Series) -> pd.Series:
    """
    Normalizes a column of data and applies a visual scale to it.

    :param column: a column of numeric data
    :return: a normalized column of data
    """
    return ((column - column.min()) / (column.max() - column.min()) + .1) * VISUAL_SCALE


def merge_pupil_data(norm_column: pd.Series, raw_column: pd.Series, time: np.array, category: list) -> pd.DataFrame:
    """
    A helper function for merging seemingly unrelated pupil data sources into a single data frame.
    This function exists to assist with pupil labeling.

    :param norm_column: the normalized pupil data
    :param raw_column: the raw pupil data
    :param time: the time stamps in minutes
    :param category: the category data for the y-axis (e.g. left, average, right)
    :return: a data frame containing all of this data
    """
    return pd.DataFrame(
        data={"Normalized": norm_column, "Raw": raw_column, "Category": category, "Time": time}
    )


def label_pupils(pupil_data: pd.DataFrame, axes: plt.Axes, color: str, xy_text: tuple):
    """
    A pupil labeling procedure which leverages pupil data to apply annotations.
    In particular, this function labels the min and max dots in a scatter plot
    with an annotation and a dark outline.

    :param pupil_data: a special dataframe of pupil data
    :param axes: the axes to plot on
    :param color: the color of the scatter plot dots
    :param xy_text: the offset for the label
    :return: None
    """
    edge_data = generate_pupil_edge_colors(pupil_data["Normalized"], color)
    axes.annotate(
        f'{pupil_data["Raw"].max():.2f} mm',
        (pupil_data["Time"][edge_data[1]], pupil_data["Category"][edge_data[1]]),
        textcoords="offset points",
        ha='center',
        va='center',
        xytext=xy_text
    )
    axes.annotate(
        f'{pupil_data["Raw"].min():.2f} mm',
        (pupil_data["Time"][edge_data[0]], pupil_data["Category"][edge_data[0]]),
        textcoords="offset points",
        ha='center',
        va='center',
        xytext=xy_text
    )
    # noinspection PyTypeChecker
    axes.scatter(
        pupil_data["Time"],
        pupil_data["Category"],
        s=pupil_data["Normalized"],
        edgecolors=edge_data[2],
        color=color
    )


def get_quadrant_color_map() -> dict:
    """
    Generates the color map for the correlation plot.

    :return: a dictionary of colors depending on quadrant
    """
    colors = cm.get_cmap("Pastel1").colors
    quads = {
        "Q1": colors[0],
        "Q2": colors[1],
        "Q3": colors[2],
        "Q4": colors[3]
    }
    return quads


def get_quad_colors(column: pd.Series) -> list:
    """
    Given a column of data, this function will generate a list of colors.
    Specifically, quadrant labels are mapped to colors.

    :param column: a column of quadrant labels
    :return: a list of colors
    """
    return [get_quadrant_color_map().get(value, "white") for value in column]


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
    windowed_data = fixation_sequence_sans_dupes.resample(WINDOW)
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


def save_file(fig: plt.Figure, participant: str, segment: str):
    """
    A helper function which generates folders and file names for each figure to be saved.

    :param fig: the figure object to be saved
    :param participant: the name of the participant whose figure is being saved
    :param segment: the name of the segment to be stored
    :return:
    """
    file_name = f"{segment} of {participant}'s {fig.canvas.get_window_title()} Over {WINDOW}"
    file_path = Path("plots", participant, segment)
    file_path.mkdir(parents=True, exist_ok=True)
    to_save = file_path / Path(file_name).with_suffix(".png")
    logging.info(f"Saving {file_name}")
    fig.savefig(to_save)


def expand_paths(paths: List[str]) -> List[str]:
    """
    A helper function for expanding a list of paths that may contain directories.

    :param paths: a list of paths
    :return: an expanded list of paths
    """
    logging.info(f"Expanding provided paths list: {paths}")
    all_paths = list()
    for path in paths:
        if not Path(path).is_dir():
            all_paths.append(path)
        else:
            for sub_path in Path(path).iterdir():
                all_paths.append(sub_path)
    logging.info(f"Expanded paths: {all_paths}")
    return all_paths


# MAIN LOGIC -----------------------------------------------------------------------


def main():
    """
    The main drop in function for the program.

    :return: None
    """
    logging.basicConfig(
        level=logging.INFO,
        format=">>> %(levelname)s: %(asctime)s: %(message)s",
        datefmt='%m/%d/%Y @ %I:%M:%S %p'
    )
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
        paths = expand_paths(paths)
        for path in paths:
            participants = read_data_files(path)
            analyze_data(participants[path])
            del participants


if __name__ == '__main__':
    main()
