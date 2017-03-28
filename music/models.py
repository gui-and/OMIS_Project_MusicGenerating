from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, TimeDistributed, Input, SimpleRNN
from keras import backend as K
from keras.layers import LSTM
from keras.layers import Input,LSTM,RepeatVector,Dense,SimpleRNN,GRU
from keras.optimizers import Adam,RMSprop,SGD
import numpy as np


def Autoencoder(encoder, decoder):
	autoencoder = decoder(encoder.output)
        model = Model(input=encoder.input, output=autoencoder)
        return model


def Encoder(latent_dim, seq_length, seq_dim, activation_list, dim_list):
        input = Input(shape= (seq_length, seq_dim),name='encoder_input')
	emb = Embedding(input_dim=128,output_dim=128, input_length=seq_length)(input)
        for i,(dim,activation) in enumerate(zip(dim_list, activation_list)):
            if i ==0:
                encoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(input)
            else:
                encoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(encoded)

        encoded = LSTM(output_dim=latent_dim,activation='tanh',name = 'encoded_layer',return_sequences=False)(encoded)
        return Model(input=input, output= encoded,name='Encoder')


def Decoder(latent_dim, seq_length, dim_list, activation_list, dof):
        input = Input(shape=(latent_dim,),name='decoder_input')
        decoded = RepeatVector(seq_length)(input)
        for i,(dim, activation) in enumerate(zip(dim_list, activation_list)):
            decoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(decoded)

        decoded = SimpleRNN(output_dim=dof,activation='softmax',
                            name='decoder_output',return_sequences=True)(decoded)
        return Model(input = input,output=decoded,name='Decoder')

def Discriminator(latent_dim):
        input = Input(shape=(latent_dim,),name='discmt_input')
        discmt = Dense(100,activation='relu',name='discmt_h1')(input)
        discmt = Dense(2,activation='sigmoid',name='discmt_output')(discmt)
        return Model(input=input,output=discmt,name='Discmt')

def Generator( encoder, discriminator):
	discriminator.trainable = False
	generator = discriminator(encoder.output)
	model = Model(input=encoder.input, output=generator)
	return model

def compile(generator, discriminator, autoencoder):
	# compilation of autoencoder
	autoencoder.compile('rmsprop',loss='mse',metrics=['accuracy'])

	# compilation of discriminator
	# we need to make sure discriminator is trainable
	discriminator.trainable = True
        optimizer_discmt = SGD(lr=0.01)
        discriminator.compile(optimizer_discmt ,loss='binary_crossentropy',metrics=['accuracy'])

	# compilation of generator = encoder + discriminator
      	generator.compile('rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

