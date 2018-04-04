#PERCEPTRON

import copy
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import spline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


class Perceptron:	
	#CONSTRUTOR
	def __init__(self, num_inputs, bias, func_type):
		self.w = self.weight_init(num_inputs)
		self.b = bias
		self.func_type = func_type

	
	#INICIALIZACAO DOS PESOS E BIAS
	def weight_init(self, num_inputs): 
		"""
		Funcao que inicializa os pesos e bias aleatoriamente utilizando numpy
		Parametro: num_inputs - quantidade de entradas X
		Retorna: w,b - pesos e bias da rede inicializados
		"""
		### Insira seu cadigo aqui (2 linhas)

		w = np.random.random_sample((num_inputs)) - 0.5
		
		return w


	#IMPLEMENTACAO DA FUNCAO DE ATIVACAO
	def activation_func(self, func_type, z):
		"""
		Funcao que implementa as funcoes de ativacao mais comuns
		Parametros: func_type - uma string que contem a funcao de ativação desejada
					z - vetor com os valores de entrada X multiplicado pelos pesos
		Retorna: saida da funcao de ativacao
		"""
		### Seu codigo aqui (2 linhas)
		if func_type == 'sigmoid':
			return 1/(1 + np.exp(z*(-1)))
		elif func_type == 'tanh':
			return (2/(1 + np.exp(z*(-2))))-1
		elif func_type == 'relu':
			if z<0:
				return 0
			return z
		elif func_type == 'degrau':
			if z>0:
				return 1;
			return 0

	#VISUALIZACAO DA FUNCAO DE ATIVACAO
	def visualizeActivationFunc(self, z):
		func = []
		for i in range(len(z)):
			func.append(activation_func('tanh', z[i]))
		plt.plot(z,func)
		plt.xlabel('Entrada')
		plt.ylabel('Valores de Saída')
		plt.show()


	#CALCULO DA SAIDA DO NEURONIO
	def forward(self, x):
		"""
		Funcao que implementa a etapa forward propagate do neuronio
		Parametros: x - entradas
		"""
		z = np.dot(self.w, x) + self.b
		#print ("Valor de z")
		#print (z)
		out = self.activation_func(self.func_type, z)
		return out

	#FUNCAO DE PREDICAO
	def predict(self, out):
		#print (out)
		if (out>0.5):
			return 1
		return 0


	def input_output_scramble(self, X, Y):
		
		for i in range(len(X)):
			X[i].append(Y[i])

		np.random.shuffle(X)

		for i in range(len(X)):
			Y[i] = X[i][len(X[i]) - 1]
			X[i].pop()

		return X, Y 

	def perc_training(self, X, Y, num_iteration, learning_rate):

		for i in range(num_iteration):
			for j in range(len(X)):
				#print("aqui")
				y_pred = self.forward(X[j])
				erro = Y[j] - y_pred
				self.w = np.add(self.w, np.dot(erro * learning_rate, X[j]))
				
			X, Y = self.input_output_scramble(X, Y)

class Layer:
	def __init__(self, perceptron, num_perceptron):
		self.num_perceptron = num_perceptron
		self.perceptron = perceptron
		self.layer = []

	def start_layer(self):
		for i in range(self.num_perceptron):
			self.layer.append(copy.deepcopy(self.perceptron))

	def out_Layer(self, x):
		out = []
		for i in range(self.num_perceptron):
			out.append(self.layer[i].forward(x))

		return out

	def training_Layer(self, X, Y, num_iteration, learning_rate):
		npY = np.array(Y)
		for i in range(self.num_perceptron):
			self.layer[i].perc_training(X, npY[:,i], num_iteration, learning_rate)


	def input_output_scramble(self, X, Y):
		
		for i in range(len(X)):
			X[i].append(Y[i])

		np.random.shuffle(X)

		for i in range(len(X)):
			Y[i] = X[i][len(X[i]) - 1]
			X[i].pop()

		return X, Y 

	def layer_training(self, X, Y, num_iteration, learning_rate):

		for i in range(num_iteration):
			for j in range(len(X)):
				for innerCount in range(len(self.layer)):
					y_pred = self.layer[innerCount].forward(X[j])
					erro = Y[j][innerCount] - y_pred
					self.layer[innerCount].w = np.add(self.layer[innerCount].w, np.dot(erro * learning_rate, X[j]))
				
			X, Y = self.input_output_scramble(X, Y)




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
	f1 = open(inputFile, "r")
	lines = f1.readlines()
	cnt = i = 0
	for line in lines:
		i += 1
		auxArray = line.split()
		X = [float(auxArray[0]), float(auxArray[1]), float(auxArray[2])]
		Y.append([int(x) for x in auxArray[3:]])

		pred = layer.out_Layer(X)

		print ("Prediction: ")
		print(pred)
		print("Output Expected: ") 
		print(Y[i-1])

		if (pred == Y[i-1]):
			answer = "OK"
			cnt += 1
		else:
			answer = "FAIL!"
		
		print(answer)

	print("Total Correct answers: " + str(cnt) + " = " + str((cnt/i) * 100) + "% of Total")


#FUNCAO MAIN
if __name__ == "__main__":
	
	"""neuron = Perceptron(3, 0, "degrau")

	X = [[0, 0, 0],
	     [0, 0, 1],
	     [0, 1, 0],
	     [0, 1, 1],
	     [0, 1, 1]]
	Y = [0, 0, 0, 1, 1]

	neuron.perc_training(X, Y, 10000, 0.25)


	X = [[0, 0, 0],
	     [0, 0, 1],
	     [0, 1, 0],
	     [1, 0, 0],
	     [0, 1, 1],
	     [1, 0, 1],
	     [1, 1, 0],
	     [1, 1, 1]]
	Y = [[1,0,0,0,0,0,0,0], 
		 [0,1,0,0,0,0,0,0], 
		 [0,0,1,0,0,0,0,0], 
		 [0,0,0,1,0,0,0,0],
		 [0,0,0,0,1,0,0,0],
		 [0,0,0,0,0,1,0,0],
		 [0,0,0,0,0,0,1,0],
		 [0,0,0,0,0,0,0,1]]
	"""


	X = []
	Y = []

	for i in range(1, 9):
		inputFile = "Noise " + str(i) + ".txt"
		parseInput(inputFile, X, Y)


	#Y = [[1],[0],[0],[0],[0],[0],[0],[0]]
	neuron = Perceptron(3, 0, "degrau")
	singleLayer = Layer(neuron, 8)
	singleLayer.start_layer()
	singleLayer.layer_training(X, Y, 1000, 0.35)

	parseTest("TestFile.txt", singleLayer)



	#print ("Prediction: " + str(singleLayer.out_Layer([1, 0, 0]))) #0,0,0,1,0,0,0,0]
