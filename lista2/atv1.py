import numpy as np
from keras.model import Sequential #Linear-type model of NN, great for feed-forward CNNs
from keras.layers import Dense, Dropout, Activation, Flatten # Regular layers, used in almost all NN
from keras.layers import Convolution2D, MaxPooling2D # Convolutional layers
from keras.utils import np_utils



np.random.seed(123)
