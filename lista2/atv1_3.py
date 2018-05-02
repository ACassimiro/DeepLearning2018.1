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

model = Sequential()

# Now, we flattens the input, and initiates our fully conected layers
model.add(Flatten(input_shape=(28, 28, 1)))
# To finish it, we add two fully conected layers separated, with 500 weights in the first,and then outputs.
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# results : [0.09477241398687475, 0.9719] 6 epochs
# results : [0.17003534824103117, 0.9512] 1 epoch

sgd = optimizers.SGD(lr=0.01, decay=0, momentum=0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# results : [0.2080327854797244, 0.9409] 6 epochs
# results : [0.3651481668591499, 0.8994] 1 epoch

model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=1 )

score = model.evaluate(x_test, y_test, verbose=0)


print(score)


#print(model.output_shape)



np.random.seed(123)
