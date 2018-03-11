#PERCEPTRON

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


#INICIALIZACAO DOS PESOS E BIAS
def weight_init(num_inputs): 
	"""
	Funcao que inicializa os pesos e bias aleatoriamente utilizando numpy
	Parametro: num_inputs - quantidade de entradas X
	Retorna: w,b - pesos e bias da rede inicializados
	"""
	### Insira seu cadigo aqui (2 linhas)
	w = None
	b = None
	return w,b


#IMPLEMENTACAO DA FUNCAO DE ATIVACAO
def activation_func(func_type, z):
	"""
	Funcao que implementa as funcoes de ativacao mais comuns
	Parametros: func_type - uma string que contem a funcao de ativação desejada
				z - vetor com os valores de entrada X multiplicado pelos pesos
	Retorna: saida da funcao de ativacao
	"""
	### Seu codigo aqui (2 linhas)
	if func_type == 'sigmoid':
		return 1/(1 + exp(z*(-1)))
	elif func_type == 'tanh':
		return (2/(1 + exp(z*(-2))))-1
	elif func_type == 'relu':
		if z<0:
			return 0
		return z
	elif func_type == 'degrau':
		if z>0:
			return 1;
		return 0

#VISUALIZACAO DA FUNCAO DE ATIVACAO
def visualizeActivationFunc(z):
	z = np.arange(-5., 5., 0.2)
	func = []
	for i in range(len(z)):
		func.append(activation_func('tanh', z[i]))
	plt.plot(z,func)
	plt.xlabel('Entrada')
	plt.ylabel('Valores de Saída')
	plt.show()


if __name__ = "__main__":
	print("ola")
	z = np.arange(-5., 5., 0.2)
	print(z)
	visualizeActivationFunc(z)
