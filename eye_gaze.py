import csv
import sys

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
            tables[f'table_{index}'].append(row)
else:
    print("Please provide the path to a data file.")
