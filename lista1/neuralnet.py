#PERCEPTRON
import copy
import numpy as np
import matplotlib.pyplot as plt
import random
from ParseInput import *
from scipy.interpolate import spline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class NeuronLayer():
    def __init__(self, n_neurons, n_inputs, func_type):
        self.layer_weights = 2 * np.random.random((n_inputs, n_neurons)) - 1 #Cria uma matriz inputsxneurons
        self.layer_bias = 1 # 2 * np.random.random(n_neurons) - 1 #Cria uma matriz de bias
        self.func_type = func_type
        self.func_type_ = func_type+'_'

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_(self, x):
        return x * (1 - x)
    def step(self, x):
        return np.heaviside(x, 0)
    def step_(self, x):
        return np.where(x>0,1,1)
    def relu(self, x):
        return np.maximum(x, 0)
    def relu_(self, x):
        return self.step(x)
    def tanh(self, x):
        return np.tanh(x)
    def tanh_(self, x):
        return 1.0 - np.tanh(x)**2
    def linear(self, x):
        return x
    def linear_(self, x):
        return np.ones_like(x)

    def activation_func(self, func_type, x):
        if func_type == 'sigmoid':
            return self.sigmoid(x)
        elif func_type == 'tanh':
            return self.tanh(x)
        elif func_type == 'relu':
            return self.relu(x)
        elif func_type == 'degrau':
            return self.step(x)
        elif func_type == 'linear':
            return self.linear(x)
        elif func_type == 'sigmoid_':
            return self.sigmoid_(x)
        elif func_type == 'tanh_':
            return self.tanh_(x)
        elif func_type == 'relu_':
            return self.relu_(x)
        elif func_type == 'degrau_':
            return self.step_(x)
        elif func_type == 'linear_':
            return self.linear_(x)

class NeuralNet():
    def __init__(self, layer, n_layer):
        self.layer = layer
        self.n_layer = n_layer

    def forward(self, input):
        outputLayers = []
        #Para cada camada na rede aplica-se a funcao de ativacao no
        #produto interno dos valores de entrada da camanda com os pesos da camada
        for i in range(self.n_layer):
            if i == 0:
                outputLayers.append(self.layer[i].activation_func(self.layer[i].func_type, np.dot(input, self.layer[i].layer_weights) + self.layer[i].layer_bias))
            else:
                outputLayers.append(self.layer[i].activation_func(self.layer[i].func_type, np.dot(outputLayers[i-1], self.layer[i].layer_weights) + self.layer[i].layer_bias))

        return outputLayers

    def stoc_training(self, input_data, output_data, iterations, learning_rate, moment_constant):
        
        errorChart = []

        for it in range(iterations):        
            old_delta = []
            for k in (range(len(input_data))):

                outs = self.forward([input_data[k]])
                erro = []
                delta = []

                for i in range(len(outs)):
                    erro.append(np.zeros(shape= (len(outs[i]), len(outs[i][0])) ))
                    delta.append(np.zeros(shape= (len(outs[i]), len(outs[i][0])) ))

                #Propaga o erro pra trás
                for i in range((len(outs)-1),-1,-1):
                    #Caso especial pra a saída da rede
                    if i == (len(outs)-1):
                        erro[i] = (output_data[k] - outs[i])
                    else:
                        erro[i] = np.dot(delta[i+1], self.layer[i+1].layer_weights.T)

                    delta[i] = erro[i] * self.layer[i].activation_func(self.layer[i].func_type_, outs[i])


                #Atualiza os pesos
                for i in range(len(outs)):
                    #Caso especial pra entrada da rede
                    if k == 0:
                        if i == 0:
                            self.layer[i].layer_weights += learning_rate * np.array([input_data[k]]).T.dot(delta[i])
                        else:
                            self.layer[i].layer_weights += learning_rate * outs[i-1].T.dot(delta[i])
                    else:
                        if i == 0:
                            self.layer[i].layer_weights += learning_rate * np.array([input_data[k]]).T.dot(delta[i]) + (moment_constant * old_delta[i])
                        else:
                            self.layer[i].layer_weights += learning_rate * outs[i-1].T.dot(delta[i]) + (moment_constant * old_delta[i])

                    self.layer[i].layer_bias += learning_rate * np.sum(delta[i], axis=0)

                old_delta = delta
                #print(np.average(erro[-1]) ) 
                errorChart.append(np.average(erro[-1]) ) 

        self.printErrorChart(errorChart)


    def batch_training(self, input_data, output_data, iterations, learning_rate):

        errorChart = []

        for it in range(iterations):
            outs = self.forward(input_data)
            erro = []
            delta = []
            for i in range(len(outs)):
                erro.append(np.zeros(shape= (len(outs[i]), len(outs[i][0])) ))
                delta.append(np.zeros(shape= (len(outs[i]), len(outs[i][0])) ))

            #Propaga o erro pra trás
            for i in range((len(outs)-1),-1,-1):
                #Caso especial pra a saída da rede
                if i == (len(outs)-1):
                    erro[i] = (output_data - outs[i])
                else:
                    erro[i] = delta[i+1].dot(self.layer[i+1].layer_weights.T)

                delta[i] = erro[i] * self.layer[i].activation_func(self.layer[i].func_type_, outs[i])


            #Atualiza os pesos
            for i in range(len(outs)):
                #Caso especial pra entrada da rede
                if i == 0:
                    self.layer[i].layer_weights += learning_rate * input_data.T.dot(delta[i])
                else:
                    self.layer[i].layer_weights += learning_rate * outs[i-1].T.dot(delta[i])

                self.layer[i].layer_bias += learning_rate * np.sum(delta[i], axis=0)

            errorChart.append(np.average(erro[-1]))
            #print(np.average(erro[-1]) ) 

        self.printErrorChart(errorChart)


    def printErrorChart(self, errorChart):
        plt.title("Matplotlib demo") 
        plt.xlabel("x axis caption") 
        plt.ylabel("y axis caption") 
        plt.plot(range(0, len(errorChart)), errorChart) 
        plt.show()


           
    def print_weights(self):
        for i in range(self.n_layer):
            print("Layer " + str(i))
            print (self.layer[i].layer_weights)

    def predict(self, treshold, x):
        return 1 if x > treshold else 0

    def predictByHigherValue(self, out):
        auxVal = -10000
        auxIndex = 0

        i = -1
        for val in out:
            i += 1

            if(auxVal < val):
                auxVal = val
                auxIndex = i

        for count in range(len(out)):
            if (count == auxIndex):
                out[count] = 1
            else:
                out[count] = 0

        return out
