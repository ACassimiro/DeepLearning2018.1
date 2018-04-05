#PERCEPTRON

import copy
import numpy as np
import matplotlib.pyplot as plt
import random
from ParseInput import parseXORInput
from scipy.interpolate import spline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class NeuronLayer():
    def __init__(self, n_neurons, n_inputs):
        self.layer_weights = 2 * np.random.random((n_inputs, n_neurons)) - 1 #Cria uma matriz inputsxneurons

class NeuralNet():
    def __init__(self, layer, n_layer, func_type):
        self.layer = layer 
        self.n_layer = n_layer
        self.func_type = func_type
        self.func_type_ = func_type+'_'

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_(self, x):
        return x * (1 - x)
    def step(self, x):
        return np.heaviside(x, 0)
    def step_(self, x):
        return 0
    def relu(self, x):
        return np.maximum(x, 0)
    def relu_(self, x):
        return 0
    def tanh(self, x):
        return (2/(1 + np.exp(z*(-2))))-1
    def tanh_(self, x):
        return 0

    def activation_func(self, func_type, x):
        if func_type == 'sigmoid':
            return self.sigmoid(x)
        elif func_type == 'tanh':
            return self.tanh(x)
        elif func_type == 'relu':
            return self.relu(x)
        elif func_type == 'degrau':
            return self.step(x)
        elif func_type == 'sigmoid_':
            return self.sigmoid_(x)
        elif func_type == 'tanh_':
            return self.tanh_(x)
        elif func_type == 'relu_':
            return self.relu_(x)
        elif func_type == 'degrau_':
            return self.step_(x)
        

    def forward(self, input):
        outputLayers = []
        #Para cada camada na rede aplica-se a funcao de ativacao no 
        #produto interno dos valores de entrada da camanda com os pesos da camada
        for i in range(self.n_layer):
            if i == 0:
                outputLayers.append(self.activation_func(self.func_type, np.dot(input, self.layer[i].layer_weights)))
            else:
                outputLayers.append(self.activation_func(self.func_type, np.dot(outputLayers[i-1], self.layer[i].layer_weights)))

        return outputLayers


    def stoc_training(self, input_data, output_data, iterations, learning_rate):
        
        for it in range(iterations):
            for k in (range(len(input_data))):
                
                outs = self.forward([input_data[k]])
                outs[0] = outs[0]
                erro = [None] * len(outs)
                delta = [None] * len(outs)

                print()
                print(outs)
                print()

                #Propaga o erro pra trás
                for i in range((len(outs)-1),-1,-1):
                    #Caso especial pra a saída da rede
                    if i == (len(outs)-1):
                        erro[i] = output_data[k] - outs[i]
                    else:
                        erro[i] = np.dot(delta[i+1], self.layer[i+1].layer_weights.T)

                    delta[i] = erro[i] * self.activation_func(self.func_type_, outs[i])


                #Atualiza os pesos
                for i in range(len(outs)):
                    #Caso especial pra entrada da rede
                    if i == 0:
                        self.layer[i].layer_weights += learning_rate * np.dot(np.array([input_data[k]]).T, delta[i])
                    else:
                        self.layer[i].layer_weights += learning_rate * np.dot(outs[i-1].T, delta[i])



    def batch_training(self, input_data, output_data, iterations, learning_rate):
        for it in range(iterations):
            outs = self.forward(input_data)
            erro = [None] * len(outs)
            delta = [None] * len(outs)

            print(outs)

            #Propaga o erro pra trás
            for i in range((len(outs)-1),-1,-1):
                #Caso especial pra a saída da rede
                if i == (len(outs)-1):
                    erro[i] = output_data - outs[i]
                else:
                    erro[i] = np.dot(delta[i+1], self.layer[i+1].layer_weights.T)

                delta[i] = erro[i] * self.activation_func(self.func_type_, outs[i])


            #Atualiza os pesos
            for i in range(len(outs)):
                #Caso especial pra entrada da rede
                if i == 0:
                    self.layer[i].layer_weights += learning_rate * np.dot(input_data.T, delta[i])
                else:
                    self.layer[i].layer_weights += learning_rate * np.dot(outs[i-1].T, delta[i])


    def print_weights(self):
        for i in range(self.n_layer):
            print("Layer " + str(i))
            print (self.layer[i].layer_weights)

if __name__ == "__main__":
    random.seed(1)
    layer1 = NeuronLayer(4, 2)
    layer2 = NeuronLayer(1, 4)

    nn = NeuralNet([layer1, layer2], 2, "sigmoid")

    #print("1) Random weighs")
    #nn.print_weights()

    X_training,Y_training = parseXORInput("NoiseXOR.txt")

    input_data = np.array(X_training)
    output_data = np.array(Y_training).T
    #print(output_data)

    nn.stoc_training(input_data, output_data, 1000, 0.2)
    
    print("2) Weighs after training")
    nn.print_weights()

    X_test, Y_test = parseXORInput("TestXORFile.txt")

    for x in range(len(X_test)):     
        out = nn.forward(np.array(X_test[x]))
        print ("3) Test")
        for i in range(len(out)):
            print("Outputs from layer " + str(i))
            print(out[i])

        print("Prediceted : " + str(out[len(out)-1]))
        print("Expected : " + str(Y_test[0][x]))
    