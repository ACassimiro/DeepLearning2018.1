from exercicio1 import Perceptron

def parseInput(inputFile, nodeNum, X, Y):
	#arquivo de input
	f1=open(inputFile, "r")

	#processamento das linhas
	lines = f1.readlines()

	for line in lines:
		auxArray = line.split()

		X.append([float(auxArray[0]), float(auxArray[1]), float(auxArray[2])])
		Y.append(int(auxArray[3+nodeNum]))

	f1.close()

	return X, Y

if __name__ == "__main__":

	nodeNum = 0

	X = []
	Y = []

	for i in range(1,9):
		inputFile = "Noise " + str(i) + ".txt"
		parseInput(inputFile, nodeNum, X, Y)

	#inputFile = "Noise " + "5" + ".txt"
	#parseInput(inputFile, nodeNum, X, Y)

	neuron = Perceptron(3, 0, "degrau")

	neuron.perc_training(X, Y, 10000, 0.1)


	print ("Prediction: " + str(neuron.forward([-0.05, 0.01, 0.03]))) #0 0 0 
	print ("Prediction: " + str(neuron.forward([-0.05, 0.01, 1.03]))) #0 0 1
	print ("Prediction: " + str(neuron.forward([-0.05, 1.01, 0.03]))) #0 1 0
	print ("Prediction: " + str(neuron.forward([1.05, 0.01, 0.03])))  #1 0 0 
	print ("Prediction: " + str(neuron.forward([0.05, 1.01, 1.03])))  #0 1 1
	print ("Prediction: " + str(neuron.forward([1.05, 1.01, 0.03])))  #1 1 0
	print ("Prediction: " + str(neuron.forward([1.05, 0.01, 1.03])))  #1 0 1
	print ("Prediction: " + str(neuron.forward([1.05, 1.01, 1.03])))  #1 1 1

	#print("") 
	#print(X) 
	#print("") 
	#print(Y) 