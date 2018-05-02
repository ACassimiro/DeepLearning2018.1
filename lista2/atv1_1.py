import numpy as np
from keras.models import Sequential #Linear-type model of NN, great for feed-forward CNNs
from keras.layers import Dense, Dropout, Activation, Flatten # Regular layers, used in almost all NN
from keras.layers import Conv2D, MaxPooling2D # Convolutional layers
from keras.utils import np_utils
from keras.datasets import mnist
from keras import optimizers

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Understanding of the dataset

# print(x_train.shape)
#print(y_train.shape)
#print(y_train[0])

#from matplotlib import pyplot as plt
#plt.imshow(x_train[0])
#plt.show()

#print(y_train[0])


# !!!!!!!!!!!!!!!!!!!!!!!!!!!! Part "Leitura, visualização e pré-processamento" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Explicitly declaring the depth of the training set, necessary for the Theano backend

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],  28, 28, 1)

#print(x_train.shape)

# Setting the data type for float32, and normalizing the pixel values to the range [0, 1]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test  /= 255

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Part "One-hot enconding" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# y_train is the expected values of the x_train set, concatenated in a single array
# I want to transform it in a binary class matrix

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Declare the model of the architecture

model = Sequential()

# First convolutional layer, composed of 32 3x3 filters. The step is 1 by default, without padding
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=( 28, 28, 1)))
# Second vonlutional layer, just like the first
model.add(Conv2D(32, (3, 3), activation='relu'))
# Second, comes a pooling layer that uses a 2,2 maxpooling
model.add(MaxPooling2D( pool_size=(2,2)))
# Now, comes a dropout layer, zeroing 25% of my neurons in training
model.add(Dropout(0.25))

# Now, we flattens the input, and initiates our fully conected layers
model.add(Flatten())
# To finish it, we add two fully conected layers separated by a dropout layer.
# Our first dense layers have 128 neurons, and our output layer have 10
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# results : [0.027837168360259328, 0.9929] (loss rate, acuracy) 12 epochs
# results : [0.032599467055701645, 0.9901] 6 epochs
# results : [0.047493071697838604, 0.9834] 1 epoch

sgd = optimizers.SGD(lr=0.01, decay=0, momentum=0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# results : [0.04744950870887842, 0.9842] 12 epochs
# results : [0.07461333639165386, 0.9761] 6 epochs
# result : [0.21740307833105327, 0.9365] 1 epoch

model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=1 )

score = model.evaluate(x_test, y_test, verbose=0)


print(score)


#print(model.output_shape)



np.random.seed(123)
