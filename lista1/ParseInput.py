from exercicio1 import Perceptron

def parseInput(inputFile, nodeNum, X, Y):
	#arquivo de input
	f1=open(inputFile, "r")

	#processamento das linhas
	lines = f1.readlines()

	for line in lines:
		auxArray = line.split()

		X.append([auxArray[0], auxArray[1], auxArray[2]])
		Y.append(auxArray[3+nodeNum])

	f1.close()

	return X, Y

if __name__ == "__main__":

	nodeNum = 0

	X = []
	Y = []

	for i in range(1,9):
		inputFile = "Noise " + str(i) + ".txt"
		parseInput(inputFile, nodeNum, X, Y)

	neuron = Perceptron(3, 0, "degrau")

	neuron.perc_training(X, Y, 10000, 0.25)

	#print("") 
	#print(X) 
	#print("") 
	#print(Y) 