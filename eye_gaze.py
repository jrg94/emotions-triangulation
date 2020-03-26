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
    fixation_sequence = stimulus_data.drop_duplicates(FIXATION_SEQUENCE)
    fixation_sequence_length = pd.to_numeric(fixation_sequence[FIXATION_SEQUENCE]).max()
    fixation_duration_mean = pd.to_numeric(fixation_sequence[FIXATION_DURATION]).mean()
    print(
        f'{stimulus}: \n'
        f'\tFixation Sequence Length: {fixation_sequence_length}\n'
        f'\tAverage Fixation_Duration: {fixation_duration_mean}'
    )
