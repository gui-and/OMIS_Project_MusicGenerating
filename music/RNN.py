# import from python file in our directory
from preprocess import preprocessMidi
from models import Encoder, Decoder, Autoencoder, Discriminator, Generator, compile
from plot_loss import plot_loss

# general import
import random, sys
from random import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras import backend as K
from keras.utils import np_utils

import os
import numpy as np
import datetime as dt
import csv
import argparse

#output_length = 20
dataDirName = "../MIDI/test"

def importDataSet(dirName, output_length):
	print(":: IMPORTING DATA SET")
	X,y = preprocessMidi(dirName,verbose=0,removeExceptions=True,
							max_sample_len=output_length,
							allowMultipleNotesOnTempo=False,
							allowNoteOnSeveralTempos=False)
	if len(X) == 0:
		raise Exception("The sample is empty.")
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)
	print("Data set splitted into train and test data")
	return X_train,X_test,y_train,y_test

def aae(batch_size=128, dim_list=[100,100],
	latent_dim=32, nb_epoch=1, nb_epoch_ae=1,
	plt_frq=50, saved_model='',
	dataDirName=dataDirName, max_len=20, saved='',
	activation=['tanh', 'tanh']):
	
	# get date
	now = dt.datetime.now()
	date = now.strftime("%d%m_%H%M%S")

	# creating file name with parameters in name
	interm=''
	for i in dim_list:
		interm += str(i)

	name='e'+str(nb_epoch)+'_a'+str(nb_epoch_ae)+'_l'+str(latent_dim)+'_i'+interm+'_o'+str(max_len)


	# import Data
	Y_train, Y_test, Y_train, Y_test = importDataSet(dataDirName, max_len)


        dof = Y_train.shape[2]
        activation_list = activation
        dim_list = dim_list
        output_length = Y_train.shape[1]


	# define different models
        encoder = Encoder(latent_dim, output_length, dof, activation_list, dim_list)
        decoder = Decoder(latent_dim, output_length, dim_list, activation_list, dof)
        autoencoder = Autoencoder(encoder, decoder)
        discriminator = Discriminator(latent_dim)
	generator = Generator(encoder, discriminator)

	# compilating every models necessary
        compile(generator, discriminator, autoencoder)

	# summary of the different models
	encoder.summary()
	decoder.summary()
	discriminator.summary()

	# loading autoencoder if needed
	if saved != '':
		autoencoder = load_model(saved)	
		name += '_s'



        # Pre-train the AE
        print 'pre-training autoencoder'

        autoencoder.fit(Y_train, Y_train,
                                nb_epoch=nb_epoch_ae,
                                verbose=2,
                                batch_size=batch_size,
                                shuffle=True,
                                validation_data=(Y_test, Y_test))


        # Pre-train the discriminator network ...
        ntrain = 1000
        XT = np.random.permutation(Y_train)

	# generating vector (real distribution and encoder output)
        zfake = np.random.uniform(low=-1.0, high=1.0, size=[XT.shape[0], latent_dim])
        zreal = encoder.predict(XT)
        X = np.concatenate((zreal, zfake))

	# putting labels on vectors
        n = XT.shape[0]
        y = np.zeros(2*n)
        y[:n] = 1
        y = np_utils.to_categorical(y, 2)

	# training discriminator
        discriminator.fit(X, y, nb_epoch=1, verbose=0, batch_size=batch_size)
        
        y_hat = discriminator.predict(X)
        y_hat_idx = np.argmax(y_hat, axis=1)

        y_idx = np.argmax(y, axis=1)

        diff = y_idx-y_hat_idx
        n_tot = y.shape[0]
        n_rig = (diff==0).sum()
        acc = n_rig*100.0/n_tot
        print "Discriminator Accuracy pretrain: %0.02f pct (%d pf %d) right on %d epoch"%(acc, n_rig, n_tot, 1)

        # set up loss storage vector
    	losses = {"discriminator":[], "generator":[]}

	# defining function to train aae
        def train_for_n(nb_epoch=5, plt_frq=plt_frq, BATCH_SIZE=32):

                count = 0
                for e in range(nb_epoch):
                        # Train ae
                        print "epoch %d"%(e+1)
			# first we train the autoencoder
                        autoencoder_losses = autoencoder.fit(Y_train, Y_train,
                                                         shuffle=True,
                                                         nb_epoch=1,
                                                         batch_size=BATCH_SIZE,
                                                         verbose=2,
                                                         validation_data=(Y_test, Y_test))


                        # Make generative latent vectors
                        music_batch = Y_train[np.random.randint(Y_train.shape[0], size=BATCH_SIZE)] 
                        noise = np.random.uniform(low=-1.0, high=1.0, size=[BATCH_SIZE,latent_dim])
                        zreal = encoder.predict(music_batch)
                
                        # Train discriminator on generated images
                        nb_misclassified = np.random.randint(BATCH_SIZE)

                        X0 = np.concatenate((zreal, noise))
                        y0 = np.zeros(BATCH_SIZE)
                        y0 = np_utils.to_categorical(y0, 2)

			# noising labels
                        misclass = np.zeros(BATCH_SIZE)
                        misclass[:nb_misclassified] = 0
                        misclass = np_utils.to_categorical(misclass, 2)

                        y0=np.concatenate((misclass, y0))
                           
                        #in order to shuffle labels
                        zipped = list(zip(X0, y0))
                        shuffle(zipped)
                        X0, y0 = zip(*zipped)
                        X0 = np.array(X0)
                        y0 = np.array(y0)

                        # then training discriminator 
                        #make_trainable(discriminator, True)
                        # print "Training discriminator"
                        d_loss, d_lab  = discriminator.train_on_batch(X0, y0)
                        losses["discriminator"].append(float(d_loss))
                
                        # train Generator-Discriminator stack on input noise to non-generated output class
                        y2 = np.ones(BATCH_SIZE)
                        y2 = np_utils.to_categorical(y2, 2)
                

                        #make_trainable(discriminator, False)
                        # Train generator
                        # print "Training generator"
                        g_loss, g_lab = generator.train_on_batch(music_batch, y2)

                        image_batch = Y_train[np.random.randint(0,Y_train.shape[0],size=BATCH_SIZE)]        
                        g_loss, g_lab = generator.train_on_batch(music_batch, y2 )
                        losses["generator"].append(float(g_loss))

                
                        # Updates plots

                        d_acc = discriminator.evaluate(X,y, batch_size=BATCH_SIZE, verbose=0) 
                        print "\ndiscriminator loss:", losses["discriminator"][-1]
                        print "discriminator acc:", d_acc[1]
                        print "generator loss:", losses["generator"][-1]
                        #print "autoencoder loss:", autoencoder_losses
                        
                        count += 1
                        
	# launching previous function to train aae
        train_for_n(nb_epoch=nb_epoch, plt_frq=plt_frq, BATCH_SIZE=batch_size)

	# saving final models
        autoencoder.save(name+'_autoencoder.h5')
        encoder.save(name+'_encoder_save.h5')
        decoder.save(name+'_decoder_save.h5')

	
	#writing losses in a csv to plot them via plot_loss.py
	with open('loss.csv', 'w') as csvfile:
		fieldnames = ['discriminator', 'generator']
		w = csv.DictWriter(csvfile, fieldnames=fieldnames)
		w.writeheader()
		w.writerow(losses)

	# plotting loss
	plot_loss('loss.csv')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Learn an AAE')
	parser.add_argument('-b', '--batch', action='store', type=int, default=128,
        	            help='batch size')
	parser.add_argument('-l', '--latent', action='store', type=int, default=32,
        	            help='latent dimension in autoencoder')
	parser.add_argument('-i', '--interm', action='store', type=int, nargs='+', default=[100,100],
        	            help='intermediate dimension in autoencoder')
	parser.add_argument('-e', '--epoch', action='store', type=int, default=1,
        	            help='number of epoch to do for the AAE')
	parser.add_argument('-a', '--auto', action='store', type=int, default=1,
        	            help='number of epoch to pretrain autoencoder')
	parser.add_argument('-s', '--saved', action='store', default='',
        	            help='autoencoder to load')
	parser.add_argument('-o', '--outputlen', action='store', type=int, default=20,
        	            help='length for the sequence input and output')
	parser.add_argument('-ac', '--activation', action='store', nargs='+', default=['tanh', 'tanh'],
        	            help='list of activations to use')

	args = parser.parse_args()

	batch_size = args.batch
	latent_dim = args.latent
	dim_list = args.interm
	nb_epoch = args.epoch
	ae = args.auto
	max_len = args.outputlen

	saved = args.saved
	activation=args.activation


	aae(latent_dim=latent_dim,
                dim_list=dim_list,
                batch_size=batch_size,
                nb_epoch=nb_epoch,
                nb_epoch_ae=ae,
                saved_model=saved,
		max_len=max_len,
		saved=saved,
		activation=activation)

