import argparse

import numpy as np
from random import shuffle
from keras.utils import np_utils
from keras.layers import Input
from keras.models import Model, load_model
from keras.layers.core import Reshape, Dense,Dropout,Activation,Flatten
from keras.activations import *
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import *
from keras.optimizers import *
import random, sys
import csv

from keras import backend as K
from keras.datasets import mnist

import datetime

now = datetime.datetime.now()

date = now.strftime("%d%m_%H%M%S")

import plot

def model_encoder(latent_dim, input_shape, hidden_dim=512):

    encoder_1 = Dense(hidden_dim, activation='relu')
    encoder_2 = Dense(hidden_dim, activation='relu')
    encoder_l = Dense(latent_dim, activation='linear')


    x = Input(input_shape)
    h = Flatten()(x)
    h = encoder_1(h)
    h = encoder_2(h)
    s = encoder_l(h) 
    
    return Model(x, s, name="encoder")
    

def model_generator(latent_dim, input_shape, hidden_dim=512):
    x = Input(shape=(latent_dim,))

    decoder_1 = Dense(hidden_dim, activation='relu')
    decoder_2 = Dense(hidden_dim, activation='relu')
    decoder_o = Dense(np.prod(input_shape), activation='sigmoid')
    reshaping = Reshape(input_shape)


    h = decoder_1(x)
    h = decoder_2(h)
    s = decoder_o(h)

    s = reshaping(s)

    return Model(x, s, name="discriminator")

def model_discriminator(latent_dim, output_dim=2, hidden_dim=512, batch_size=128):
    dropout_rate = 0.25

    z = Input(batch_shape=(batch_size, latent_dim))
    H = Dense(hidden_dim, activation='relu')(z)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Dense(hidden_dim, activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)

    y = Dense(2, activation='softmax')(H)

    return Model(z, y)



def aae(batch_size=128, intermediate_dim=512, latent_dim=32, nb_epoch=1, nb_epoch_ae=1, plt_frq=50, saved_model=''):

    input_shape = (28, 28)

    opt = Adam(lr=1e-4)
    dopt = Adam(lr=1e-3)

    nb_epoch_pretrain_ae = nb_epoch_ae
    epoch_pretrain_disc = 1

    #generator (z -> x)
    generator = model_generator(latent_dim, input_shape)
    #encoder (x -> z)
    encoder = model_encoder(latent_dim, input_shape)
    #autoencoder
    autoencoder = Model(encoder.inputs, generator(encoder(encoder.inputs)))

    autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy')
    #discriminator (z -> y)
    discriminator = model_discriminator(latent_dim)
    discriminator.compile(loss='binary_crossentropy', optimizer=dopt, metrics=['accuracy'])

    # Build stacked GAN model
    gan_input = encoder.inputs[0]
    H = encoder(gan_input)
    gan_V = discriminator(H)
    GAN = Model(gan_input, gan_V)
    GAN.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Freeze weights in the discriminator for stacked training
    def make_trainable(net, val):
        net.trainable = val
        for l in net.layers:
            l.trainable = val

    make_trainable(discriminator, True)

    # load mnist data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # Pre-train the AE
    print 'pre-training autoencoder'


    autoencoder.fit(x_train, x_train,
                nb_epoch=nb_epoch_pretrain_ae,
                verbose=2,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test))


    autoencoder.save('ae_'+str(date)+str(nb_epoch_pretrain_ae)+'.h5')

    # Pre-train the discriminator network ...

    ntrain = 1000
    XT = np.random.permutation(x_train)[:ntrain]

    zfake = np.random.uniform(low=-2, high=2, size=[XT.shape[0], latent_dim])
    zreal = encoder.predict(XT)
    X = np.concatenate((zreal, zfake))

    n = XT.shape[0]
    y = np.zeros(2*n)
    y[:n] = 1
    y = np_utils.to_categorical(y, 2)

    discriminator.fit(X, y, nb_epoch=epoch_pretrain_disc, verbose=0, batch_size=batch_size)
    
    y_hat = discriminator.predict(X)
    y_hat_idx = np.argmax(y_hat, axis=1)

    y_idx = np.argmax(y, axis=1)

    diff = y_idx-y_hat_idx
    n_tot = y.shape[0]
    n_rig = (diff==0).sum()
    acc = n_rig*100.0/n_tot
    print "Discriminator Accuracy pretrain: %0.02f pct (%d pf %d) right on %d epoch"%(acc, n_rig, n_tot, epoch_pretrain_disc)

    # set up loss storage vector
    losses = {"discriminator":[], "generator":[]}

    def train_for_n(nb_epoch=5000, plt_frq=plt_frq, BATCH_SIZE=32):

        count = 0

        for e in range(nb_epoch):
            # Train ae
            autoencoder_losses = autoencoder.fit(x_train, x_train,
                             shuffle=True,
                             nb_epoch=1,
                             batch_size=BATCH_SIZE,
                             verbose=2,
                             validation_data=(x_test, x_test))

            # Make generative images
            image_batch = x_train[np.random.randint(x_train.shape[0], size=BATCH_SIZE)]
            zfake = np.random.uniform(low=-2, high=2, size=[BATCH_SIZE,latent_dim])
            zreal = encoder.predict(image_batch)
            X0 = np.concatenate((zreal, zfake))
  
            # Train discriminator on generated images
            y0 = np.zeros(2*BATCH_SIZE)
            y0[:2*BATCH_SIZE:2] = 1
            shuffle(y0)
            y0 = np_utils.to_categorical(y0, 2)               
            
            #in order to shuffle batch
            zipped = list(zip(X0, y0))
            shuffle(zipped)
            X0, y0 = zip(*zipped)
            X0 = np.array(X0)
            y0 = np.array(y0)


            print "epoch %d out of %d" % (e+1, nb_epoch)
             
            make_trainable(discriminator, True)
            #print "Training discriminator"
            d_loss, d_lab  = discriminator.train_on_batch(X0, y0)
            
            losses["discriminator"].append(float(d_loss))
        
            # train Generator-Discriminator stack on input noise to non-generated output class
            y2 = np.ones(BATCH_SIZE)
            y2 = np_utils.to_categorical(y2, 2)
        

            make_trainable(discriminator, False)

            # Train generator
            # print "Training generator"
            g_loss, g_lab = GAN.train_on_batch(image_batch, y2)

            image_batch = x_train[np.random.randint(0,x_train.shape[0],size=BATCH_SIZE)]    
            g_loss, g_lab = GAN.train_on_batch(image_batch, y2 )
            losses["generator"].append(float(g_loss))

            
            # Updates plots

            d_acc = discriminator.evaluate(X,y, batch_size=BATCH_SIZE, verbose=0) 
            print "\ndiscriminator loss:", losses["discriminator"][-1]
            print "discriminator acc:", d_acc[1]
            print "generator loss:", losses["generator"][-1]
            #print "autoencoder loss:", autoencoder_losses
            
            count += 1

            if count == plt_frq:
                
                nnow = datetime.datetime.now()

                ndate = now.strftime("%d%m_%H%M%S")
                plot_f_name = "latent_"+ndate+".png"
                print "plotting latent dimensions"
                plot.latent_space(generator, plot_f_name, latent_dim)
                count = 0              

    train_for_n(nb_epoch=nb_epoch, plt_frq=plt_frq, BATCH_SIZE=batch_size)

    autoencoder.save(date+'_autoencoder_final.h5')
    encoder.save(date+'_encoder_save.h5')
    generator.save(date+'_generator_save.h5')

    return losses

parser = argparse.ArgumentParser(description='Learn an AAE')
parser.add_argument('-b', '--batch', action='store', type=int, default=128, 
                    help='batch size to learn AAE')
parser.add_argument('-l', '--latent', action='store', type=int, default=32, 
                    help='latent dimension in autoencoder')
parser.add_argument('-i', '--interm', action='store', type=int, default=512, 
                    help='intermediate dimension in autoencoder')
parser.add_argument('-e', '--epoch', action='store', type=int, default=5, 
                    help='number of epoch to do')
parser.add_argument('-a', '--auto', action='store', type=int, default=5, 
                    help='number of epoch to pretrain autoencoder')
parser.add_argument('-p', '--plot', action='store', type=int, default=5, 
                    help='frequence to plot grid image')

args = parser.parse_args()

batch_size = args.batch
latent_dim = args.latent
intermediate_dim = args.interm
nb_epoch = args.epoch
ae = args.auto
plt_frq = args.plot

loss = aae(latent_dim=latent_dim, 
    intermediate_dim=intermediate_dim, 
    batch_size=batch_size, 
    nb_epoch=nb_epoch,
    nb_epoch_ae=ae,
    plt_frq=plt_frq)

#writing losses in a csv to plot them via plot_loss.py
with open('loss.csv', 'w') as csvfile:
    fieldnames = ['discriminator', 'generator']
    w = csv.DictWriter(csvfile, fieldnames=fieldnames)
    w.writeheader()
    w.writerow(loss)

