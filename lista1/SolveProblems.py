from neuralnet import *
from ParseInput import *
import matplotlib.pyplot as plt

def solveCube():
	#Teste para o problema do cubo
	layer1 = NeuronLayer(8, 3)

	nn = NeuralNet([layer1], 1, "sigmoid")

	print("1) Random weighs")
	nn.print_weights()


	X_training,Y_training = parseInput("trainingCube.txt")

	input_data = np.array(X_training)
	output_data = np.array(Y_training)

	#showGraph(input_data, output_data, input_data, nn.forward(input_data)[-1])

	nn.batch_training(input_data, output_data, 10000, 0.001)
	#nn.stoc_training(input_data, output_data, 60000, 0.2)
	#nn.mt_training(input_data, output_data, 60000, 0.2, 0.1)

	print("2) Weighs after training")
	nn.print_weights()
	#showGraph(input_data, output_data, input_data, nn.forward(input_data)[-1])


	X_test, Y_test = parseInput("testCube.txt")

	#print(Y_test)

	cnt = 0
	for x in range(len(X_test)):
		out = nn.forward(np.array(X_test[x]))

		print("Prediceted : " + str(nn.predictByHigherValue(out[-1])))
		print("Expected : " + str(Y_test[x]))
		print("*****")

		if(np.all(nn.predictByHigherValue(out[-1]) == Y_test[x])):
			cnt += 1

	print(str(cnt) + " ACERTOS DE " + str(len(X_test)))
	#showGraph(X_test, Y_test[0], X_test, nn.forward(X_test)[-1])

def solveXOR():
	#Testes para XOR
	layer1 = NeuronLayer(4, 2)
	layer2 = NeuronLayer(1, 4)

	nn = NeuralNet([layer1, layer2], 2, "sigmoid")

	print("1) Random weighs")
	nn.print_weights()

	X_training,Y_training = parseXORInput("NoiseXOR.txt")

	input_data = np.array(X_training)
	output_data = np.array(Y_training).T

	showGraph(input_data, output_data, input_data, nn.forward(input_data)[-1])

	#nn.batch_training(input_data, output_data, 1000, 0.2)
	#nn.stoc_training(input_data, output_data, 1000, 0.2)
	nn.mt_training(input_data, output_data, 1000, 0.2, 0.2)

	print("2) Weighs after training")
	nn.print_weights()
	showGraph(input_data, output_data, input_data, nn.forward(input_data)[-1])

	X_test, Y_test = parseXORInput("TestXORFile.txt")
	cnt = 0

	for x in range(len(X_test)):
		out = nn.forward(np.array(X_test[x]))

		print("Prediceted : " + str( nn.predict(0.5, out[-1])))
		print("Expected : " + str(Y_test[0][x])) 
		print("*****")
		if(np.all(nn.predict(0.5, out[-1]) == Y_test[0][x])):
			cnt += 1

	print(str(cnt) + " ACERTOS DE " + str(len(X_test)))

	showGraph(X_test, Y_test[0], X_test, nn.forward(X_test)[-1])

def solveSine():
    layer1 = NeuronLayer(10, 1)
    layer2 = NeuronLayer(10, 10)
    layer3 = NeuronLayer(1, 10)

    nn = NeuralNet([layer1, layer2, layer3], 3, "tanh")

    print("1) Random weighs")
    nn.print_weights()

    X_training,Y_training = parseSinInput("TrainingSin.txt")

    input_data = np.array(X_training)
    output_data = np.array(Y_training).T

    showGraph(input_data, output_data, input_data, nn.forward(input_data)[-1])

    nn.batch_training(input_data, output_data, 5000, 0.0001)
    #nn.stoc_training(input_data, output_data, 500, 0.0001)
    #nn.mt_training(input_data, output_data, 100000, 0.001, 0.015)

    print("2) Weighs after training")
    nn.print_weights()
    showGraph(input_data, output_data, input_data, nn.forward(input_data)[-1])

    X_test, Y_test = parseSinInput("TestSin.txt")

    for x in range(len(X_test)):
        out = nn.forward(np.array(X_test[x]))

        print("Prediceted : " + str(out[len(out)-1]))
        print("Expected : " + str(Y_test[0][x]))
        print("*****") 

    showGraph(X_test, Y_test[0], X_test, nn.forward(X_test)[-1])

def solvePatternRec():
	layer1 = NeuronLayer(6, 2)
	layer2 = NeuronLayer(10, 6)
	layer3 = NeuronLayer(8, 10)

	nn = NeuralNet([layer1, layer2, layer3], 3, "tanh")

	print("1) Random weighs")
	nn.print_weights()


	X_training,Y_training = parsePatternInput("TrainingPattern2.txt")

	input_data = np.array(X_training)
	output_data = np.array(Y_training)
	#showGraph(input_data, output_data, input_data, nn.forward(input_data)[-1])

	nn.batch_training(input_data, output_data, 50000, 0.0001)
	#nn.stoc_training(input_data, output_data, 60000, 0.2)
	#nn.mt_training(input_data, output_data, 60000, 0.2, 0.1)

	print("2) Weighs after training")
	nn.print_weights()
	#showGraph(input_data, output_data, input_data, nn.forward(input_data)[-1])

	X_test, Y_test = parsePatternInput("TestPattern2.txt")
	cnt = 0

	for x in range(len(X_test)):
		print()
		out = nn.forward(np.array(X_test[x]))

		print("Prediceted : " + str(nn.predictByHigherValue(out[len(out)-1])))
		print("Expected : " + str(Y_test[x]))

		if(np.all(nn.predictByHigherValue(out[-1]) == Y_test[x])):
			cnt += 1

	print(str(cnt) + " ACERTOS DE " + str(len(X_test)))
	#showGraph(X_test, Y_test[0], X_test, nn.forward(X_test)[-1])

def showGraph(X1, Y1, X2, Y2):
	plt.plot(X1, Y1, '.')
	plt.plot(X2, Y2, '.')
	plt.show()

if __name__ == "__main__":
	random.seed(1)
	#solveCube()
	solveXOR()
	#solveSine()
    #solvePatternRec()