import csv
import sys

if len(sys.argv) > 1:
    path = sys.argv[1]
    with open(path) as data:
        csv_reader = csv.reader(data, delimiter="\t")
        for row in csv_reader:
            if not "".join(row):  # identies empty rows
                print("hi")
else:
    print("Please provide the path to a data file.")
