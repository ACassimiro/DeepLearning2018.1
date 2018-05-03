import keras
import tensorflow as tf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation


#***********************************************
# parseTestDataset
#    - Parses the input dataset file, correcting 
#    missing items according to the correction
#    method. 
# 
#    - attributes: Attributes used for training
#    - method: Correction method 
#    - pathTest: Path to test file
#    - pathDatabase: Path to database file
#    - discard: Discard or not discard in case of 
#      no match found
#***********************************************

def parseTestDataset(attributes, method, pathTest, pathDatabase, discard):
	# Passo 1 - Leitura do dataset 
	test = pd.read_csv(pathTest)
	dataBase = pd.read_excel(pathDatabase, sheet_name="titanic3")

	survivedList = []
	indexList = []

	#emptyDataFrame = 0
	for i in range(len(test.index)):
		auxRow = dataBase.loc[(dataBase['name'] == test['Name'][i]) & (dataBase['ticket'] == test['Ticket'][i])]
		
		if(auxRow.empty & (discard == 0)):
			survivedList.append(0)
			#emptyDataFrame += 1
			#print()
			#print(test.ix[i])
		elif(auxRow.empty & (discard == 1)): 
			indexList.append(i)
			survivedList.append(0)
		else:
			survivedList.append(auxRow.iloc[0, 1])


	survivedDF = pd.DataFrame({'Survived': survivedList})
	test = test.join(survivedDF)
	
	if(discard == 1):
		indexList = indexList[::-1]

		for i in indexList:
			test = test.drop(test.index[i])


	#SUBSTITUIR INSTANCIAS EM STRING POR NUMERICAS
	if "Sex" in attributes:
		test = test.replace(["male", "female"], [0,1])

	if "Embarked" in attributes:
		test = test.replace(["S", "C", "Q"] , [0,1,2])
	
	#Evita que a coluna "Survived" seja apagada em caso de drop 
	attributes.append("Survived")

	test = test[attributes]

	#REMOVER TODAS AS INSTÂNCIAS ONDE NAN APARECE
	if method == "dropAll":
		test = test.dropna(axis=0, how='any')
	elif method == "fillZero":
		test = test.fillna(0)
	elif method == "correctAVG":
		for attribute in attributes:
			test[attribute] = test[attribute].fillna(test[attribute].mean())
	elif method == "correctMode":
		for attribute in attributes:
			test[attribute] = test[attribute].fillna(test[attribute].mode())
	elif method == "correctMedian":
		for attribute in attributes:
			test[attribute] = test[attribute].fillna(test[attribute].median())

	#Remove "Survived" de attributes para que este não seja incluido no treinamento
	attributes.remove("Survived")

	X_test = test[attributes]
	Y_test = test["Survived"]

	return X_test, Y_test


	return 0



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
	seed = 7
	np.random.seed(seed)
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
	print(kfold.split(X_train, Y_train))
	print("**************\n\n\n\n")
	cvscores = []
	for trn, val in kfold.split(X_train, Y_train):
		model = Sequential()

		#ADICIONA CAMADAS NA REDE
		model.add(Dense(30, input_dim=inputNum, kernel_initializer='uniform', activation='relu'))
		#model.add(Dropout(0.2))
		#model.add(BatchNormalization())
		

		model.add(Dense(16, kernel_initializer='uniform', activation='tanh'))
		#model.add(Dropout(0.2))
		#model.add(BatchNormalization())
		
		
		model.add(Dense(8, kernel_initializer='uniform', activation='tanh'))
		#model.add(Dropout(0.2))
		#model.add(BatchNormalization())
		
		
		model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

		#Constroi a rede especificando método de treinamento, otimizador e metrica de progresso
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

		#Treina rede com base no X e Y dados
		model.fit(X_train.loc[trn], Y_train.loc[trn], epochs=350, batch_size=15)

		scores = model.evaluate(X_train.loc[val], Y_train.loc[val], verbose=0)

		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		cvscores.append(scores[1] * 100)


	print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
	return model

#MAIN FUNCTION

if __name__ == "__main__":
	np.random.seed(1)

	attributes = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
	pathTrain = "dataset/train.csv"	
	pathTest = "dataset/teste.csv"
	pathDatabase = "dataset/titanic3.xls"

	#METODO DE CORRECAO DO DATASET
	# dropAll - Deleta todas as linhas com um campo NaN
	# fillZero - Substitui todos os NaN por 0 
	# correctAVG - Substitui NaN por valor médio
	# correctMode - Substitui NaN pela moda
	# correctMedian - Substitui NaN pela mediana
	
	#CORRECTMODE NAO FUNCIONA POR ALGUM MOTIVO
	method = "correctMedian"
	
	X_train, Y_train = parseTrainDataset(attributes, method, pathTrain)
	
	X_test, Y_test = parseTestDataset(attributes, method, pathTest, pathDatabase, 1)
	
	model = makeAndTrainNN(X_train, Y_train, len(attributes))

	# evaluate the model
	scores = model.evaluate(X_test, Y_test)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	