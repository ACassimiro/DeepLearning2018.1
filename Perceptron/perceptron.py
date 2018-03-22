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
		#SIGMOID
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

		#RELU
		Array de pesos: 
		[0.3705497329614913, 0.24110300672835994]
		bias: 0


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
	out = activation_func("degrau", z)
	return out

#FUNCAO DE PREDICAO
def predict(out):
	#print (out)
	if (out>0.5):
		return 1
	return 0

#FUNCAO DE BUSCA DE PESOS ALEATORIOS EMPIRICAMENTE
def empyric(X):
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


#FUNCAO DE TREINO E AVALIACAO
def perceptron(x, y, num_iteration, learning_rate):
	"""
	Funcao que implementa o loop do treinamento
	Parametros: x - entrada da rede
	            y - rotulos/labels
	            num_iteration - quantidade de iteracoes desejadas para a rede convergir
	            learning_rate - taxa de aprendizado para calculo do erro 
	"""
	
	#TODO: DESCOBRIR VALOR DO RÓTULO


	#Passo 1 - Inicie os pesos e bias (1 linha)
	w, b = weight_init(2)

	#Passo 2 - Loop por X interacoes
	for j in range(num_iteration):
		#Passo 3 - calcule a saida do neuronio (1 linha)
		y_pred = forward(w, b, x[j%4])
		#Passo 4 - calcule o erro entre a saida obtida e a saida desejada nos rotulos/labels (1 linha)
		erro = y[j%4] - y_pred
		#Passo 5 - atualize o valor dos pesos (1 linha)
		#Dica: voce pode utilizar a funcao np.dot e a funcao transpose de numpy
		w = np.add(w, np.dot(erro*learning_rate, x[j%4]))

	#Verifique as saidas
	print('Saida obtida: ', y_pred)
	print('Pesos obtidos: ', w)

	#Metricas de Avaliacao
	y_pred = predict(y_pred)
	print('Matriz de confusao:')
	print(confusion_matrix(y, y_pred))
	print('F1 Score:')
	print(classification_report(y, y_pred))


#FUNCAO MAIN
if __name__ == "__main__":
	X = [[0, 0],
	     [1, 0],
	     [0, 1],
	     [1, 1]]
	Y = [0, 0, 0, 1]

	perceptron(X, Y, 10, 0.2)
	#empyric(X)





    

	
