import csv
import sys
from datetime import datetime

import pandas as pd

META_DATA = "table_0"
GAZE_CALIBRATION_POINTS_DETAILS = "table_1"
GAZE_CALIBRATION_SUMMARY_DETAILS = "table_2"
DATA = "table_3"
STIMULUS_NAME = "StimulusName"
FIXATION_DURATION = "FixationDuration"
FIXATION_SEQUENCE = "FixationSeq"
TIMESTAMP = "Timestamp"
TIME_FORMAT = "%Y%m%d_%H%M%S%f"


def main():
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
        participants = read_tsv_files(*paths)
        for participant in paths:
            output_statistics(participants[participant])


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
    return data


def statistics_windows(stimulus_data: pd.DataFrame, duration: int = None) -> pd.DataFrame:
    if duration:
        timestamps = pd.to_datetime(stimulus_data[TIMESTAMP], format=TIME_FORMAT)
        return timestamps
    else:
        return stimulus_data


def output_statistics(tables: dict):
    """
    Outputs summary statistics for a participant.

    :param tables: a participant table
    :return: None
    """
    df = clean_data(tables)
    stimuli = df[STIMULUS_NAME].unique()
    for stimulus in stimuli:
        stimulus_filter = df[STIMULUS_NAME] == stimulus
        stimulus_data = df[stimulus_filter]
        fixation_sequence_sans_dupes = stimulus_data.drop_duplicates(FIXATION_SEQUENCE)
        fixation_sequence: pd.Series = pd.to_numeric(fixation_sequence_sans_dupes[FIXATION_SEQUENCE],
                                                     downcast='unsigned')
        fixation_duration: pd.Series = pd.to_numeric(fixation_sequence_sans_dupes[FIXATION_DURATION],
                                                     downcast='unsigned')
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
        print(
            "\n".join(
                [
                    f'{stimulus}:',
                    f'\t Stimulus Metrics:',
                    f'\t\tStart time: {start_date_time}',
                    f'\t\tEnd time: {end_date_time}',
                    f'\t\tDuration: {end_date_time - start_date_time}',
                    f'\tFixation Sequence Metrics:',
                    f'\t\tNumber of Points: {fixation_sequence_length} points',
                    f'\t\tDuration: {seconds} seconds',
                    f'\tFixation Duration Metrics:',
                    f'\t\tMean: {fixation_duration_mean} milliseconds',
                    f'\t\tMedian: {fixation_duration_median} milliseconds',
                    f'\t\tStandard Deviation: {fixation_duration_std} milliseconds',
                    f'\t\tMinimum: {fixation_duration_min} milliseconds',
                    f'\t\tMaximum: {fixation_duration_max} milliseconds',
                ]
            )
        )


if __name__ == '__main__':
    main()

# TODO: 30 second or one or two-minute minute buckets
# TODO: Mean fixation duration & fixation count
# TODO: slice window by 2 minutes
