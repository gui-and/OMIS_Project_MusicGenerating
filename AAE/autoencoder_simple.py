import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
import cPickle, random, sys, keras
from keras.models import Model

from scipy.stats import norm
from keras.models import load_model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist


batch_size = 32 
original_dim = 784
latent_dim = 32
intermediate_dim = 256
nb_epoch = 1
epsilon_std = 1.0


input_img = Input(shape=(original_dim,))
encoder_1 = Dense(128, activation='relu')
encoder_2 = Dense(64, activation='relu')
encoder_l = Dense(latent_dim, activation='relu')


decoder_1 = Dense(64, activation='relu')
decoder_2 = Dense(128, activation='relu')
decoder_o = Dense(original_dim, activation='sigmoid')

encoded_1 = encoder_1(input_img)
encoded_2 = encoder_2(encoded_1)
encoded_l = encoder_l(encoded_2)

decoded_1 = decoder_1(encoded_l)
decoded_2 = decoder_2(decoded_1)
decoded_o = decoder_o(decoded_2)

autoencoder = Model(input=input_img, output=decoded_o)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

encoder = Model(input=input_img, output=encoded_l)

decoder_input = Input(shape=[latent_dim])
decoded_h = decoder_1(decoder_input)
decoded_g = decoder_2(decoded_h)
decoded_f = decoder_o(decoded_g)

decoder = Model(input=decoder_input, output=decoded_f)



# train the AE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#x_train=x_train[:1000]

#autoencoder = load_model('ae_model_10.h5')

autoencoder.fit(x_train, x_train,
                nb_epoch=nb_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test))

autoencoder.save('ae_save_'+str(nb_epoch)+'_2.h5')

encoder.save('encoder_save.h5')
decoder.save('decoder_save.h5')
