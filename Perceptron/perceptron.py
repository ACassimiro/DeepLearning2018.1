#PERCEPTRON

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import spline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

"""
	PESOS E BIAS QUE FUNCIONARAM:
		Array de pesos: 
		[0.20506415855171234, 0.43438788264418093]
		bias: -0.5

		Array de pesos: 
		[0.31534810166703786, 0.1945739821791681]
		bias: -0.5

		Array de pesos: 
		[0.20449638720354146, 0.09829885487261736]
		bias: -0.3

		Array de pesos: 
		[0.2527685044456509, 0.08500280335193366]
		bias: -0.3

"""



#INICIALIZACAO DOS PESOS E BIAS
def weight_init(num_inputs): 
	"""
	Funcao que inicializa os pesos e bias aleatoriamente utilizando numpy
	Parametro: num_inputs - quantidade de entradas X
	Retorna: w,b - pesos e bias da rede inicializados
	"""
	### Insira seu cadigo aqui (2 linhas)

	count = 0
	w = []
	while (count < num_inputs):
		w.append(random.uniform(-1, 1))
		count += 1
	b = 0

	print("Array de pesos: ")
	print (w) 
	print("bias: " + str(b))

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
def visualizeActivationFunc(z):
	z = np.arange(-5., 5., 0.2)
	func = []
	for i in range(len(z)):
		func.append(activation_func('tanh', z[i]))
	plt.plot(z,func)
	plt.xlabel('Entrada')
	plt.ylabel('Valores de Saída')
	plt.show()


#CALCULO DA SAIDA DO NEURONIO
def forward(w, b, X):
	"""
	Funcao que implementa a etapa forward propagate do neuronio
	Parametros: w - pesos
	            b - bias
	            X - entradas
	"""
	z = np.dot(w, X) + b
	out = activation_func("sigmoid", z)
	return out

#FUNCAO DE PREDICAO
def predict(out):
	if (out>0.5):
		return 1
	return 0


#FUNCAO MAIN
if __name__ == "__main__":
	print("")
	#z = np.arange(-5., 5., 0.2)
	#visualizeActivationFunc(w)
	X = [[0, 0],
	     [1, 0],
	     [0, 1],
	     [1, 1]]

	w, b = weight_init(2);

	print()

	count = 0
	while (1):
		w, b = weight_init(2);

		print()

		count = 0
		auxEmpirico = 0
		while (count < 4):
		    out = forward(w, b, X[count])
		    print("Resultado da iteracao " + str(count) + ":" + str(predict(out)))
		    
		    if (count == 3):
		    	if (predict(out) == 1):
		    		auxEmpirico += 1
		    		#print("Count = 3, predict(out):" + str(predict(out)))
		    else:
		    	if (predict(out) == 0):
		    		auxEmpirico += 1
		    		#print("Count = " + str(count) + ", predict(out):" + str(predict(out)))

		    #print("auxEmpirico: " + str(auxEmpirico))
		    count += 1

		if auxEmpirico == 4:
			break;



    

	
