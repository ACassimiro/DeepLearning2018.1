import numpy as np
from keras.models import Sequential #Linear-type model of NN, great for feed-forward CNNs
from keras.layers import Dense, Dropout, Activation, Flatten # Regular layers, used in almost all NN
from keras.layers import Conv2D, MaxPooling2D # Convolutional layers
from keras.utils import np_utils
from keras.datasets import mnist
from keras import optimizers

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],  28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test  /= 255


y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


# Declare the model of the architecture

# LeNet architecture based in the model described in http://slazebni.cs.illinois.edu/spring17/lec01_cnn_architectures.pdf

model = Sequential()

# First convolutional layer, composed of 20 5x5 filters. The step is 1, with padding
model.add(Conv2D(6, (5, 5), activation='relu', padding='same', input_shape=( 28, 28, 1)))
# Pool Layer mask (2,2), 2,2 stride
model.add(MaxPooling2D( pool_size=(2,2)))
# Second convolutional layer, 50 5x5 filters
model.add(Conv2D(16, (5, 5), activation='relu', padding='same'))
# Second Poo layer, just like the previous one
model.add(MaxPooling2D( pool_size=(2,2)))

# Now, we flattens the input, and initiates our fully conected layers
model.add(Flatten())
# To finish it, we add two fully conected layers separated, with 500 weights in the first,and then outputs.
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

# keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# results : [0.03192821291965374, 0.9893] 6 epochs
# results : [0.09339068892151117, 0.9697] 1 epoch

#sgd = optimizers.SGD(lr=0.01, decay=0, momentum=0, nesterov=False)
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# results : [0.05051871402286925, 0.9835] 6 epochs
# result : [0.1500920630902052, 0.9532] 1 epoch

model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=1 )

score = model.evaluate(x_test, y_test, verbose=0)


print(score)


#print(model.output_shape)



np.random.seed(123)
