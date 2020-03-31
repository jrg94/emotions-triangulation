import csv
import sys

import pandas as pd

META_DATA = "table_0"
GAZE_CALIBRATION_POINTS_DETAILS = "table_1"
GAZE_CALIBRATION_SUMMARY_DETAILS = "table_2"
DATA = "table_3"
STIMULUS_NAME = "StimulusName"
FIXATION_DURATION = "FixationDuration"
FIXATION_SEQUENCE = "FixationSeq"

tables = dict()
index = 0

if len(sys.argv) > 1:
    path = sys.argv[1]
    with open(path) as data:
        csv_reader = csv.reader(data, delimiter="\t")
        tables[f'table_{index}'] = []
        for row in csv_reader:
            if not "".join(row):  # identifies empty rows
                index += 1
                tables[f'table_{index}'] = []
            else:
                tables[f'table_{index}'].append(row)
else:
    print("Please provide the path to a data file.")

data_table = tables[DATA]
header = data_table[0]
df = pd.DataFrame(data_table[1:], columns=header)
stimuli = df[STIMULUS_NAME].unique()
for stimulus in stimuli:
    stimulus_filter = df[STIMULUS_NAME] == stimulus
    stimulus_data = df[stimulus_filter]
    fixation_sequence_sans_dupes = stimulus_data.drop_duplicates(FIXATION_SEQUENCE)
    fixation_sequence: pd.Series = pd.to_numeric(fixation_sequence_sans_dupes[FIXATION_SEQUENCE], downcast='unsigned')
    fixation_duration: pd.Series = pd.to_numeric(fixation_sequence_sans_dupes[FIXATION_DURATION], downcast='unsigned')
    fixation_sequence_length = fixation_sequence.max()
    fixation_sequence_duration = fixation_duration.sum()
    fixation_duration_mean = fixation_duration.mean()
    fixation_duration_median = fixation_duration.median()
    fixation_duration_min = fixation_duration.min()
    fixation_duration_max = fixation_duration.max()
    fixation_duration_std = fixation_duration.std()
    print(
        "\n".join(
            [
                f'{stimulus}:',
                f'\tFixation Sequence Length: {fixation_sequence_length} points',
                f'\tAverage Fixation Duration: {fixation_duration_mean} milliseconds',
                f'\tMedian Fixation Duration: {fixation_duration_median} milliseconds',
                f'\tMinimum Fixation Duration: {fixation_duration_min} milliseconds',
                f'\tMaximum Fixation Duration: {fixation_duration_max} milliseconds',
                f'\tStandard Deviation Fixation Duration: {fixation_duration_std} milliseconds',
                f'\tFixation Sequence Duration: {fixation_sequence_duration} milliseconds'
            ]
        )
    )

    # TODO: add length of time of segment
