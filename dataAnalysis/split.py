'''
file1, percentage train -> file2, file3
Given a file and test percentage, return a file2 as training file and file3 as test file.
'''

import numpy as np
from sklearn.model_selection import train_test_split
import util
import sys


def split(file, testSize):
	samples = util.extractRowsToList(file)
	print(samples[0])
	X_train, X_test = train_test_split(samples, test_size=float(testSize), random_state=42)
	util.writeListToFile(list(X_train), file + ".train")
	util.writeListToFile(list(X_test), file + ".validation")

if __name__ == '__main__':
	split(sys.argv[1], sys.argv[2])