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
FIXATION_DURATION = "FixationDuration"
FIXATION_SEQUENCE = "FixationSeq"
TIMESTAMP = "Timestamp"
TIME_FORMAT = "%Y%m%d_%H%M%S%f"
WINDOW = "30S"
PUPIL_LEFT = "PupilLeft"
PUPIL_RIGHT = "PupilRight"


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
            }
        }
    }


def windowed_metrics(stimulus_data: pd.DataFrame) -> tuple:
    """
    Computes fixation counts and average fixation duration within some window of time.

    :param stimulus_data: the section of the data only relevant to the stimulus
    :return: fixation counts, average fixation duration (tuple)
    """
    fixation_sequence_sans_dupes = stimulus_data.drop_duplicates(FIXATION_SEQUENCE)
    windowed_data = fixation_sequence_sans_dupes.resample(WINDOW, on=TIMESTAMP)
    unique_fixation_counts = windowed_data.nunique()[FIXATION_SEQUENCE]
    average_fixation_duration = windowed_data.mean()[FIXATION_DURATION]
    return unique_fixation_counts, average_fixation_duration


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


def plot_data(stimulus, fixation_counts, avg_fixation_duration, pupil_dilation):
    """
    Plots the fixation count and average fixation duration data.

    :param stimulus: the current stimulus used as the plot title
    :param fixation_counts: the unique counts of fixations by time
    :param avg_fixation_duration: the average fixation durations by time
    :return: None
    """
    time = (fixation_counts.index.astype(np.int64) / 10 ** 9) / 60  # Converts datetime to minutes
    time = time - time.min()  # Scales minutes back to 0

    fig, ax = plt.subplots(figsize=(12, 4))

    plt.title(stimulus)

    color = 'tab:red'
    ax.plot(time, fixation_counts, color=color, linewidth=2)
    ax.set_xlabel("Time (minutes)", fontsize="large")
    ax.set_ylabel("Fixation Count", color=color, fontsize="large")
    ax.tick_params(axis='y', labelcolor=color)

    ax2 = ax.twinx()

    color = 'tab:cyan'
    ax2.plot(time, avg_fixation_duration, color=color, linewidth=2)
    ax2.set_ylabel("Mean Fixation Duration (ms)", color=color, fontsize="large")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.xticks(np.arange(0, time.max() + 1, step=2))  # Force two-minute labels

    fig.tight_layout()
    plt.show()


def generate_statistics(tables: dict):
    """
    Generates various statistics for a participant.

    :param tables: a participant table
    :return: None
    """
    df = clean_data(tables)
    stimuli = df[STIMULUS_NAME].unique()
    for stimulus in stimuli:
        stimulus_filter = df[STIMULUS_NAME] == stimulus
        stimulus_data = df[stimulus_filter]
        report = summary_report(stimulus, stimulus_data)
        output_summary_report(report)
        fixation_counts, avg_fixation_duration = windowed_metrics(stimulus_data)
        pupil_dilation = stimulus_data[[TIMESTAMP, PUPIL_LEFT, PUPIL_RIGHT]]
        print(pupil_dilation)
        plot_data(stimulus, fixation_counts, avg_fixation_duration, pupil_dilation)


if __name__ == '__main__':
    main()
