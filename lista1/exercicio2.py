#PERCEPTRON

import copy
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import spline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from exercicio1 import Perceptron
from exercicio1 import Layer


#FUNCAO DE PARSING DOS ARQUIVOS NOISE
def parseInput(inputFile, X, Y):
	#arquivo de input
	f1=open(inputFile, "r")

	#processamento das linhas
	lines = f1.readlines()

	for line in lines:
		auxArray = line.split()

		X.append([float(auxArray[0]), float(auxArray[1]), float(auxArray[2])])
		Y.append([int(x) for x in auxArray[3:]])

	f1.close()

	return X, Y


def parseTest(inputFile, layer):
	verbose = True

	f1 = open(inputFile, "r")
	lines = f1.readlines()
	cnt = i = 0
	for line in lines:
		i += 1
		auxArray = line.split()
		X = [float(auxArray[0]), float(auxArray[1]), float(auxArray[2])]
		Y = [int(x) for x in auxArray[3:]]

		out = layer.out_Layer(X)
		pred = layer.predict(out)

		if verbose:
			print ("Prediction: ")
			print(pred)
			print("Output Expected: ") 
			print(Y)

		if (pred == Y):
			answer = "OK"
			cnt += 1
		else:
			answer = "FAIL!"
		
		if verbose:	
			print(answer)

	print("Total Correct answers: " + str(cnt) + " = " + str((cnt/i) * 100) + "% of Total")


#FUNCAO MAIN
if __name__ == "__main__":
	
	X = []
	Y = []

	for i in range(1, 9):
		inputFile = "Noise " + str(i) + ".txt"
		parseInput(inputFile, X, Y)


	#Y = [[1],[0],[0],[0],[0],[0],[0],[0]]

	"""
	for i in range(len(Y)):
		for j in range(len(Y[0])):
			if(Y[i][j] == 1):
				Y[i][j] = 100
	"""

	neuron1 = Perceptron(3, 0, "relu")
	

	Layer1 = Layer(neuron1, 8)
	Layer1.start_layer()
	Layer1.layer_training(X, Y, 1000, 0.25)

	out1 = Layer1.out_Layer(X)

	neuron2 = Perceptron(8, 0, "relu")
	Layer2 = Layer(neuron2, 5)
	Layer2.start_layer()
	Layer2.layer_training(out1, Y, 1000, 0.25)

	out2 = Layer2.out_Layer(out1)

	neuron3 = Perceptron(5, 0, "relu")
	Layer3 = Layer(neuron3, 8)
	Layer3.start_layer()
	Layer3.layer_training(out2, Y, 1000, 0.25)

	out3 = Layer3.out_Layer(out2)

	print(Layer3.predict(out3))

	parseTest("TestFile.txt", singleLayer)

	print()

	#print ("Prediction: " + str(singleLayer.out_Layer([1, 0, 0]))) #0,0,0,1,0,0,0,0]
