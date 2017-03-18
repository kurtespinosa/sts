'''
extract the related scores in the specified column into a list
'''
def extractValuesToList(file, column):
	file = open(file, 'r')
	print(file)
	values = []
	for f in file:
		# print(f.strip().split('\t')[int(column)])
		values.append(float(f.strip().split('\t')[int(column)]))
	file.close()
	'''print values'''
	# for v in values:
	# 	print v
	# print(len(values))
	return values

def extractRowsToList(file):
	file = open(file, 'r')
	values = []
	for f in file:
		values.append(f)
	file.close()
	'''print values'''
	# for v in values:
	# 	print v
	# print(len(values))
	return values

def writeListToFile(inputList, filename):
	# print(len(inputList))
	# print(filename)
	# print(inputList[0])
	file = open(filename, 'w')
	for fp in inputList:
		# print(str(fp))
		file.write(str(fp))
	file.close()
    
def extractRowsToListWithLabels(file, isSick=True):
	file = open(file, 'r')
	values = []
	labels = []
	for f in file:
		values.append(f)
        if isSick:
            labels.append(float(f.strip().split("\t")[3]))
        else:
            labels.append(float(f.strip().split("\t")[0]))
	file.close()
	'''print values'''
	# for v in values:
	# 	print v
	# print(len(values))
	return values, labels