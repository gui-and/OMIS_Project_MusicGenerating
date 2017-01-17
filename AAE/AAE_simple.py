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
from keras.models import load_model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist


batch_size = 32 
original_dim = 784
latent_dim = 32
intermediate_dim = 256
nb_epoch = 10
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

autoencoder = load_model('autoencoder_final_2.h5')

autoencoder.fit(x_train, x_train,
                nb_epoch=nb_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test))


autoencoder.save('ae_1201_2147_'+str(nb_epoch)+'.h5')

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

encoder.save('encoder_before_save_1201.h5')
decoder.save('decoder_before_save_1201.h5')

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
gan_input = Input(shape=[784])
H = encoder(gan_input)
gan_V = discriminator(H)
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
GAN.summary()



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
        autoencoder_losses = autoencoder.fit(x_train, x_train,
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
        image_batch = x_train[np.random.randint(0,x_train.shape[0],size=BATCH_SIZE)]    
        g_loss, g_lab = GAN.train_on_batch(image_batch, y2 )
        losses["g"].append(g_loss)

        
        # Updates plots
        if e%plt_frq==plt_frq-1:
            d_acc = discriminator.evaluate(X,y, batch_size=BATCH_SIZE, verbose=0) 
            print "\ndiscriminator loss:", losses["d"][-1]
            print "discriminator acc:", d_acc[1]
            print "generator loss:", losses["g"][-1]
            print "autoencoder loss:", autoencoder_losses
            autoencoder.save('1201_autoencoder_epoch_'+str(e+1)+'.h5')
            GAN.save('1201_gan_epoch_'+str(e+1)+'.h5')


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
#train_for_n(nb_epoch=1, plt_frq=5, BATCH_SIZE=32)
#GAN.load_model('gan_final.h5')
train_for_n(nb_epoch=20, plt_frq=5, BATCH_SIZE=32)

autoencoder.save('autoencoder_final_1201.h5')
GAN.save('gan_final_1201.h5')

encoder.save('encoder_after_save_1201.h5')
decoder.save('decoder_after_save_1201.h5')


#print "\nEvaluation"
#test_for_n(nb_epoch=1, plt_frq=1, BATCH_SIZE=32)



#print "\nTraining AAE"
#train_for_n(nb_epoch=10, plt_frq=10, BATCH_SIZE=32)
#print "\nEvaluation"
#test_for_n(nb_epoch=1, plt_frq=1, BATCH_SIZE=32)


print "The End"


