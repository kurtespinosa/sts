
'''
MAIN file
file1, file2 -> histogram
Given file1 and file2, return a histogram.
'''

import random
import numpy
import matplotlib.pyplot as plt
import sys
import logging
import util
logging.basicConfig(filename='plot.log', filemode='w', level=logging.INFO)



def plotOneFile(file1, percentWidth, column, smallest, largest):
	
	x = util.extractValuesToList(file1, column)

	histogram=plt.figure()
	logging.info("Elements:" + str(len(x)))

	#0 and 5 are the smallest and largest values possible, 3rd parameter adjusts the width of the bins - the bigger the number the bigger the bin width
	bins = numpy.linspace(float(smallest), float(largest), len(x)*float(percentWidth))
	# bins = numpy.linspace(0,5,10)

	plt.hist(x, bins, alpha=0.5, )
	filename = file1.split('/')[-1].split('.')[0]
	plt.title(filename)
	plt.xlabel("Relatedness Score")
	plt.ylabel("Number of samples")
	plt.savefig(filename+"_"+percentWidth+".png")


def plotTwoFiles(file1, file2, percentWidth, column, smallest, largest):
	
	x = util.extractValuesToList(file1, column)
	y = util.extractValuesToList(file2, column)

	histogram=plt.figure()
	logging.info("Elements:" + str(max(len(x), len(y))))

	#0 and 5 are the smallest and largest values possible, 3rd parameter adjusts the width of the bins - the bigger the number the bigger the bin width
	bins = numpy.linspace(float(smallest), float(largest), max(len(x), len(y))*float(percentWidth))
	# bins = numpy.linspace(0,5,10)

	plt.hist(x, bins, alpha=0.5)
	plt.hist(y, bins, alpha=0.5)
	filename1 = file1.split('/')[-1].split('.')[0]
	filename2 = file1.split('/')[-1].split('.')[0]
	plt.title(filename1+"vs"+filename2)
	plt.xlabel("Relatedness Score")
	plt.ylabel("Number of samples")
	plt.savefig(filename1+"vs"+filename2+"_"+percentWidth+".png")

if __name__ == '__main__':
	# extractValuesToList(sys.argv[1])
	if len(sys.argv) == 7:
		plotTwoFiles(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
	elif len(sys.argv) == 6:
		plotOneFile(sys.argv[1], sys.argv[2], sys.argv[3],sys.argv[4], sys.argv[5])
	else:
		print("Error: usage below")
		print("One file: file1, percentWidth, column, smallest, largest, title")
		print("Two files: file1, file2, percentWidth, column, smallest, largest, title")