import csv
import numpy as np

def load_data(filename)
	with open(filename,r) as infile
		reader = csv.reader(infile)
		columnNames = nest(reader)
		rows = list(reader)
	return columnNames, rows
def separate_labels(columnNamers,rows)
	labelCol = columnNames.index('Outcome')
	ys(rows