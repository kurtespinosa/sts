import sys
import logging
import util
logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)


'''
file1, file2 > file3
Given two files with file1 containing sentence pairs (tab separated) and file2 containing the score, write into file3
sentence pairs and score separated by tabs.
'''

def merge(file1, file2, file3):
	logging.basicConfig(filename=file3+'.log',level=logging.INFO)
	#read file1 into a list
	pairs = open(file1, 'r')
	pairsList = []
	for p in pairs:
		pairsList.append(p)
	logging.info("There are "+str(len(pairsList)))

	#read file2 into a list
	scores = open(file2, 'r')
	scoresList = []
	for s in scores:
		scoresList.append(s)
	logging.info("There are "+str(len(scoresList)))

	#read into a list the final pairs
	finalPairs = []
	for i in range(len(scoresList)):
		if scoresList[i].strip() != '':
			line = scoresList[i].strip() + "\t" + pairsList[i].strip()
			finalPairs.append(line+'\n')
	logging.info("Final list:"+str(len(finalPairs)))

	#write the finalPairs into file3
	util.writeListToFile(finalPairs, file3)
	logging.info("Done writing to file.")

if __name__ == '__main__':
	merge(sys.argv[1], sys.argv[2], sys.argv[3])