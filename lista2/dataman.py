import keras
import tensorflow as tf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation


#***********************************************
# parseTrainDataset
#    - Parses the input dataset file, correcting 
#    missing items according to the correction
#    method. 
# 
#    - attributes: Attributes used for training
#    - method: Correction method 
#    - path: dataset file path
#***********************************************

def parseTrainDataset(attributes, method, path):

	# Passo 1 - Leitura do dataset 
	train = pd.read_csv(path)
	test  = pd.read_csv(path)

	#print(train)
	# Passo 2 - Separar atributos e classes

	#SUBSTITUIR INSTANCIAS EM STRING POR NUMERICAS
	if "Sex" in attributes:
		train = train.replace(["male", "female"], [0,1])

	if "Embarked" in attributes:
		train = train.replace(["S", "C", "Q"] , [0,1,2])
	
	#Evita que a coluna "Survived" seja apagada em caso de drop 
	attributes.append("Survived")

	train = train[attributes]

	print(train)

	#REMOVER TODAS AS INSTÂNCIAS ONDE NAN APARECE
	if method == "dropAll":
		train = train.dropna(axis=0, how='any')
	elif method == "fillZero":
		train = train.fillna(0)
	elif method == "correctAVG":
		for attribute in attributes:
			train[attribute] = train[attribute].fillna(train[attribute].mean())
	elif method == "correctMode":
		for attribute in attributes:
			train[attribute] = train[attribute].fillna(train[attribute].mode())
	elif method == "correctMedian":
		for attribute in attributes:
			train[attribute] = train[attribute].fillna(train[attribute].median())

	#print(train)

	#Remove "Survived" de attributes para que este não seja incluido no treinamento
	attributes.remove("Survived")

	X_train = train[attributes]
	Y_train = train["Survived"]

	return X_train, Y_train


#***********************************************
# makeAndTrainNN
#    - Using Keras, builds a neural network and
#    trains it with given X (input) and Y (expected
#    output).  
# 
#    - X_train: Input of the neural network
#    - Y_train: Expected output of the neural network
#***********************************************
def makeAndTrainNN(X_train, Y_train, inputNum):
	model = Sequential()

	#ADICIONA CAMADAS NA REDE
	model.add(Dense(30, input_dim=inputNum, kernel_initializer='uniform', activation='relu'))
	#model.add(Dropout(0.4))
	#model.add(BatchNormalization())
	

	model.add(Dense(16, kernel_initializer='uniform', activation='tanh'))
	#model.add(Dropout(0.4))
	#model.add(BatchNormalization())
	
	
	model.add(Dense(8, kernel_initializer='uniform', activation='tanh'))
	#model.add(Dropout(0.4))
	#model.add(BatchNormalization())
	
	
	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

	#Constroi a rede especificando método de treinamento, otimizador e metrica de progresso
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	#Treina rede com base no X e Y dados
	model.fit(X_train, Y_train, epochs=350, batch_size=15)

	return model

#MAIN FUNCTION

if __name__ == "__main__":
	np.random.seed(1)

	attributes = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
	path = "dataset/train.csv"	

	#METODO DE CORRECAO DO DATASET
	# dropAll - Deleta todas as linhas com um campo NaN
	# fillZero - Substitui todos os NaN por 0 
	# correctAVG - Substitui NaN por valor médio
	# correctMode - Substitui NaN pela moda
	# correctMedian - Substitui NaN pela mediana
	
	#CORRECTMODE NAO FUNCIONA POR ALGUM MOTIVO
	method = "correctAVG"

	X_train, Y_train = parseTrainDataset(attributes, method, path)
	
	model = makeAndTrainNN(X_train, Y_train, len(attributes))

	# evaluate the model
	scores = model.evaluate(X_train, Y_train)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
