#import matplotlib
#matplotlib.use('Agg')

import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D, Deconv2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
#from keras.datasets import mnist
#import matplotlib.pyplot as plt
import cPickle, random, sys, keras
from keras.models import Model
#from IPython import display

from scipy.stats import norm
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist


batch_size = 50
original_dim = 784
latent_dim = 2
intermediate_dim = 256
nb_epoch = 100
epsilon_std = 1.0

#x = Input(batch_shape=(batch_size, original_dim))
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    #epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
    epsilon = K.random_normal(shape=(latent_dim,), mean=0.,
                              std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss


vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#x_train=x_train[:1000]


vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        verbose=2,
        validation_data=(x_test, x_test))

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
#plt.figure(figsize=(6, 6))
#plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
#plt.colorbar()
#plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=[latent_dim])
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

dropout_rate = 0.25
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-3)

d_input = Input(batch_shape=(batch_size, latent_dim))
#d_input_b = Reshape((batch_size, 1, 28, 28), d_input)
#H = Convolution1D(256, 5, border_mode = 'same', activation='relu')(d_input)
H = Dense(256, activation='relu')(d_input)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
#H = Convolution1D(512, 5, border_mode = 'same', activation='relu')(H)
H = Dense(128, activation='relu')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
#H = Flatten()(H)
H = Dense(256)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(0.25)(H)
d_V = Dense(2,activation='softmax')(H)
discriminator = Model(d_input,d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt, metrics=['accuracy'])
#discriminator.summary()


# Freeze weights in the discriminator for stacked training
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
make_trainable(discriminator, False)

# Build stacked GAN model
gan_input = Input(shape=[latent_dim])
H = encoder(gan_input)
gan_V = discriminator(H)
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#GAN.summary()



ntrain = 1000
trainidx = random.sample(range(0,x_train.shape[0]), ntrain)
XT = x_train

# Pre-train the discriminator network ...
noise_d = np.random.uniform(0,1,size=[XT.shape[0],latent_dim])
generated_vec = encoder.predict(XT)
X = np.concatenate((noise_d, generated_vec))
n = XT.shape[0]
y = np.zeros([2*n,2])
y[:n,1] = 1
y[n:,0] = 1

make_trainable(discriminator,True)
discriminator.fit(X,y, nb_epoch=1, batch_size=128)
y_hat = discriminator.predict(X)


y_hat_idx = np.argmax(y_hat,axis=1)
y_idx = np.argmax(y,axis=1)
diff = y_idx-y_hat_idx
n_tot = y.shape[0]
n_rig = (diff==0).sum()
acc = n_rig*100.0/n_tot
print "Accuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot)

# set up loss storage vector
losses = {"d":[], "g":[]}

def train_for_n(nb_epoch=5000, plt_frq=25, BATCH_SIZE=32):

    for e in range(nb_epoch):  
        
        # Make generative images
        image_batch = x_train[np.random.randint(0,x_train.shape[0],size=BATCH_SIZE)]    
        noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,latent_dim])
        generated_vec = encoder.predict(image_batch)


        # Train vae
        vae_losses = vae.fit(x_train, x_train,
                             shuffle=True,
                             nb_epoch=1,
                             batch_size=BATCH_SIZE,
                             verbose=1,
                             validation_data=(x_test, x_test))

        
        # Train discriminator on generated images
        X0 = np.concatenate((noise_gen, generated_vec))
        y0 = np.zeros([2*BATCH_SIZE,2])
        y0[0:BATCH_SIZE,1] = 1
        y0[BATCH_SIZE:,0] = 1
        
        make_trainable(discriminator,True)
#        print "Training discriminator"
        d_loss, d_lab  = discriminator.train_on_batch(X0,y0)
        losses["d"].append(d_loss)
    
        # train Generator-Discriminator stack on input noise to non-generated output class
        y2 = np.zeros([BATCH_SIZE,2])
        y2[:,1] = 1
        
        make_trainable(discriminator,False)
        # Train generator
#        print "Training generator"
        g_loss, g_lab = GAN.train_on_batch(image_batch, y2 )
        losses["g"].append(g_loss)

        
        # Updates plots
        if e%plt_frq==plt_frq-1:
            d_acc = discriminator.evaluate(X,y, batch_size=BATCH_SIZE, verbose=0) 
            print "\ndiscriminator loss:", losses["d"][-1]
            print "discriminator acc:", d_acc[1]
            print "generator loss:", losses["g"][-1]
            print "vae loss:", vae_losses

def test_for_n(nb_epoch=5000, plt_frq=25, BATCH_SIZE=32):

    for e in range(nb_epoch):

        # Make generative images
        image_batch = x_test[np.random.randint(0,x_test.shape[0],size=BATCH_SIZE)]
        noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,2])
        generated_images = generator.predict(noise_gen)

        # Train discriminator on generated images
        X1 = np.concatenate((image_batch, generated_images))
        y1 = np.zeros([2*BATCH_SIZE,2])
        y1[0:BATCH_SIZE,1] = 1
        y1[BATCH_SIZE:,0] = 1

        d_loss, d_lab  = discriminator.test_on_batch(X1,y1)
        losses["d"].append(d_loss)

        # Updates plots
        if e%plt_frq==plt_frq-1:
            d_acc = discriminator.evaluate(X,y, batch_size=BATCH_SIZE, verbose=0) 
            print "discriminator loss:", losses["d"][-1]
            print "discriminator acc:", d_acc[1]


print "\nTraining AAE"
train_for_n(nb_epoch=100, plt_frq=10, BATCH_SIZE=32)
print "\nEvaluation"
test_for_n(nb_epoch=10, plt_frq=1, BATCH_SIZE=32)

print "\nTraining AAE"
train_for_n(nb_epoch=100, plt_frq=10, BATCH_SIZE=32)
print "\nEvaluation"
test_for_n(nb_epoch=10, plt_frq=1, BATCH_SIZE=32)


print "The End"


