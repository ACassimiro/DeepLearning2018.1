from exercicio1 import Perceptron

def parseInput(inputFile):
	#arquivo de input
	f1=open(inputFile, "r")

	#processamento das linhas
	lines = f1.readlines()

	X = []
	Y = []

	for line in lines:
		auxArray = line.split()

		X.append([float(auxArray[0]), float(auxArray[1]), float(auxArray[2])])
		Y.append([int(x) for x in auxArray[3:]])

	f1.close()

	return X, Y

def parseTest(inputFile, neuron):
	f1 = open(inputFile, "r")
	lines = f1.readlines()
	cnt = i = 0
	for line in lines:
		i += 1
		auxArray = line.split()
		X = [float(auxArray[0]), float(auxArray[1]), float(auxArray[2])]
		Y = (int(auxArray[3+nodeNum]))

		pred = str(neuron.forward(X))
		print ("Prediction: " + pred + " Output Expected " + str(Y))

		if (str(pred) == str(Y)):
			answer = "OK"
			cnt += 1
		else:
			answer = "FAIL!"
		
		print(answer)

	print("Total Correct answers: " + str(cnt) + " = " + str((cnt/i) * 100) + "% of Total")

def parseXORInput(inputFile):
	f1=open(inputFile, "r")
	lines = f1.readlines()
	X =[]
	Y =[[]]

	for line in lines:
		auxArray = line.split()
		X.append([float(auxArray[0]), float(auxArray[1])])
		Y[0].append(float(auxArray[2]))
	
	return X,Y

def parseSinInput(inputFile):
	f1=open(inputFile, "r")
	lines = f1.readlines()
	X =[]
	Y =[[]]

	for line in lines:
		auxArray = line.split()
		X.append([float(auxArray[0])])
		Y[0].append(float(auxArray[1]))
	
	return X,Y


def parsePatternInput(inputFile):
	#arquivo de input
	f1=open(inputFile, "r")

	X = []
	Y = []

	#processamento das linhas
	lines = f1.readlines()

	for line in lines:
		auxArray = line.split()

		X.append([float(auxArray[0]), float(auxArray[1])])
		Y.append([int(x) for x in auxArray[2:]])

	f1.close()

	return X, Y




if __name__ == "__main__":
	parseXORInput("XORSample.txt")

	'''nodeNum = 0

	X = []
	Y = []

	for i in range(1,9):
		inputFile = "Noise " + str(i) + ".txt"
		parseInput(inputFile, nodeNum, X, Y)

	#inputFile = "Noise " + "5" + ".txt"
	#parseInput(inputFile, nodeNum, X, Y)

	neuron = Perceptron(3, 0, "degrau")
	#neuron.perc_training(X, Y, 1000, 0.1)
	singleLayer = Layer(neuron, 8)
	singleLayer.start_layer()
	singleLayer.training_Layer(X, Y, 1000, 0.1)
	parseTest("TestFile.txt", neuron)'''

	#print ("Prediction: " + str(neuron.forward([-0.05, 0.01, 0.03]))) #0 0 0 
	#print ("Prediction: " + str(neuron.forward([-0.05, 0.01, 1.03]))) #0 0 1
	#print ("Prediction: " + str(neuron.forward([-0.05, 1.01, 0.03]))) #0 1 0
	#print ("Prediction: " + str(neuron.forward([1.05, 0.01, 0.03])))  #1 0 0 
	#print ("Prediction: " + str(neuron.forward([0.05, 1.01, 1.03])))  #0 1 1
	#print ("Prediction: " + str(neuron.forward([1.05, 1.01, 0.03])))  #1 1 0
	#print ("Prediction: " + str(neuron.forward([1.05, 0.01, 1.03])))  #1 0 1
	#print ("Prediction: " + str(neuron.forward([1.05, 1.01, 1.03])))  #1 1 1

	#print("") 
	#print(X) 
	#print("") 
	#print(Y) 