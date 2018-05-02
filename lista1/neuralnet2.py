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
    def __init__(self, n_neurons, n_inputs):
        self.layer_weights = 2 * np.random.random((n_inputs, n_neurons)) - 1 #Cria uma matriz inputsxneurons
        self.layer_bias =  2 * np.random.random(n_neurons) - 1 #Cria uma matriz de bias

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
        return self.step(x)
    def tanh(self, x):
        return np.tanh(x)
    def tanh_(self, x):
        return 1.0 - np.tanh(x)**2

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
                outputLayers.append(self.activation_func(self.func_type, np.dot(input, self.layer[i].layer_weights) + self.layer[i].layer_bias))
            else:
                outputLayers.append(self.activation_func(self.func_type, np.dot(outputLayers[i-1], self.layer[i].layer_weights) + self.layer[i].layer_bias))

        return outputLayers




    def mt_training(self, input_data, output_data, iterations, learning_rate, moment_constant):

        for it in range(iterations):
            old_delta = []
            for k in (range(len(input_data))):

                outs = self.forward([input_data[k]])
                erro = [None] * len(outs)
                delta = [None] * len(outs)

                #Propaga o erro pra trás
                for i in range((len(outs)-1),-1,-1):
                    #Caso especial pra a saída da rede
                    if i == (len(outs)-1):
                        erro[i] = (output_data[k] - outs[i])**2
                    else:
                        erro[i] = np.dot(delta[i+1], self.layer[i+1].layer_weights.T)

                    delta[i] = erro[i] * self.activation_func(self.func_type_, outs[i])


                #Atualiza os pesos
                for i in range(len(outs)):
                    #Caso especial pra entrada da rede
                    if i == 0:
                        if k == 0 :
                            self.layer[i].layer_weights += learning_rate * np.dot(np.array([input_data[k]]).T, delta[i])
                        else :
                            self.layer[i].layer_weights += learning_rate * np.dot(np.array([input_data[k]]).T, delta[i]) + (moment_constant * old_delta[i])
                    else:
                        if k == 0 :
                            self.layer[i].layer_weights += learning_rate * np.dot(outs[i-1].T, delta[i])
                        else :
                            self.layer[i].layer_weights += learning_rate * np.dot(outs[i-1].T, delta[i]) + (moment_constant * old_delta[i])

                old_delta = delta


    def stoc_training(self, input_data, output_data, iterations, learning_rate):

        for it in range(iterations):
            for k in (range(len(input_data))):

                outs = self.forward([input_data[k]])
                erro = [None] * len(outs)
                delta = [None] * len(outs)

                #Propaga o erro pra trás
                for i in range((len(outs)-1),-1,-1):
                    #Caso especial pra a saída da rede
                    if i == (len(outs)-1):
                        erro[i] = 1/2 * (output_data[k] - outs[i])**2
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
                    #erro_ = (output_data - outs[i])
                    #delta[i] = 2 * np.average(erro_)

                else:
                    erro[i] = delta[i+1].dot(self.layer[i+1].layer_weights.T)
                    #delta[i] = erro[i] * self.activation_func(self.func_type_, outs[i])


                delta[i] = erro[i] * self.activation_func(self.func_type_, outs[i])


            #Atualiza os pesos
            for i in range(len(outs)):
                #Caso especial pra entrada da rede
                if i == 0:
                    self.layer[i].layer_weights += learning_rate * input_data.T.dot(delta[i])
                else:
                    self.layer[i].layer_weights += learning_rate * outs[i-1].T.dot(delta[i])

                self.layer[i].layer_bias += learning_rate * np.sum(delta[i], axis=0)

            print(np.average(erro[-1]) )

    def print_weights(self):
        for i in range(self.n_layer):
            print("Layer " + str(i))
            print (self.layer[i].layer_weights)

    def predict(self, treshold, x):
        return 1 if x > treshold else 0

if __name__ == "__main__":
    random.seed(1)


    #Testes para XOR
    layer1 = NeuronLayer(4, 2)
    layer2 = NeuronLayer(1, 4)

    nn = NeuralNet([layer1, layer2], 2, "sigmoid")

    print("1) Random weighs")
    nn.print_weights()

    X_training,Y_training = parseXORInput("NoiseXOR.txt")

    input_data = np.array(X_training)
    output_data = np.array(Y_training).T

    nn.batch_training(input_data, output_data, 1000, 0.2)
    #nn.stoc_training(input_data, output_data, 60000, 0.2)
    #nn.mt_training(input_data, output_data, 60000, 0.2, 0.1)

    print("2) Weighs after training")
    nn.print_weights()

    X_test, Y_test = parseXORInput("TestXORFile.txt")

    for x in range(len(X_test)):
        out = nn.forward(np.array(X_test[x]))

        print("Prediceted : " + str( nn.predict(0.5, out[-1])))
        print("Expected : " + str(Y_test[0][x]))
        print("*****")

    '''#Testes para funcao Seno
    layer1 = NeuronLayer(10, 1)
    layer2 = NeuronLayer(10, 10)
    layer3 = NeuronLayer(1, 10)

    nn = NeuralNet([layer1, layer2,layer3], 3, "tanh")

    print("1) Random weighs")
    nn.print_weights()

    X_training,Y_training = parseSinInput("trainingSin.txt")

    input_data = np.array(X_training)
    output_data = np.array(Y_training).T

    nn.batch_training(input_data, output_data, 5000, 0.0001)
    #nn.stoc_training(input_data, output_data, 5000, 0.2)
    #nn.mt_training(input_data, output_data, 2000, 0.0001, 0.05)

    print("2) Weighs after training")
    nn.print_weights()

    X_test, Y_test = parseSinInput("testSin.txt")

    for x in range(len(X_test)):
        out = nn.forward(np.array(X_test[x]))


        print("Prediceted : " + str(out[len(out)-1]))
        print("Expected : " + str(Y_test[0][x]))
        print("*****")'''
