from preprocess import addMidiToList,predict_sample,train_sample
from mido_encodage import vectList2midi,midiChannels2midi,saveMidi
import numpy as np
import sys,getopt
import datetime as dt
from keras.models import load_model
import argparse
import matplotlib.pyplot as plt
from RNN import importDataSet

# function to generate a MIDI file and associated image using decoder
def generateMIDI(dir_name, model_name,saveName, latent_dim, dim_list, max_len):
	output_length = max_len
	dof = 128
	activation_list = ['tanh', 'tanh']

	model = load_model(model_name)
	MIDIlist = []
	#addMidiToList(MIDIinput,MIDIlist,verbose=0)
	channels = []
	nb = 0

        noise = np.random.uniform(low=-1.0, high=1.0, size=[1,latent_dim])
	
	print noise
	music = model.predict(noise)
	print music.shape
	y = music
	for j in range(music.shape[0]):
		one_hot = np.zeros((music.shape[1],music.shape[2]))
		for i in range(music.shape[1]):
			index = np.argmax(music[0,i])
			one_hot[i,index]=1

		plt.figure()
		plt.imshow(one_hot)
		plt.gray()
		plt.savefig(saveName[:-4]+str(j)+'.png')
		plt.clf()

	ch_list = []
	for batch in y:
		ch_list.extend(batch)
	ch_mid = vectList2midi(ch_list)
	channels.append(ch_mid)
	midi = midiChannels2midi(channels)
	saveMidi(midi, saveName)

# function to generate a csv instead of an image
def generateMIDIcsv(dir_name, model_name,saveName, latent_dim, dim_list, max_len):
	output_length = max_len
	dof = 128
	activation_list = ['tanh', 'tanh']

	model = load_model(model_name)
	MIDIlist = []
	#addMidiToList(MIDIinput,MIDIlist,verbose=0)
	channels = []
	nb = 0

        noise = np.random.uniform(low=-1.0, high=1.0, size=[1,latent_dim])
	
	print noise
	music = model.predict(noise)
	print music.shape
	for j in range(music.shape[0]):
		one_hot = np.zeros((music.shape[1],music.shape[2]))
		for i in range(music.shape[1]):
			index = np.argmax(music[0,i])
			one_hot[i,index]=1

		a = np.asarray(one_hot)
		np.savetxt(saveName[:-4]+'-'+str(j)+'.csv', a, delimiter=",")

# generating a CSV and MIDI file using autoencoder
# autoencoder used with MIDIinput as input
def generatefromMIDItoCsv(MIDIinput,dir_name, model_name,saveName, latent_dim, dim_list, max_len):
	output_length = max_len
	dof = 128
	activation_list = ['tanh', 'tanh']
	
	dataDirName = "../MIDI/test"
	x_train, x_test, Y_train, Y_test = importDataSet(dataDirName, max_len)

	model = load_model(model_name)
	MIDIlist = []
	addMidiToList(MIDIinput,MIDIlist,verbose=0)
	channels = []
	nb = 0
	np.savetxt(saveName[:-4]+'-train.csv', Y_train[0], delimiter=",")

	for channel in MIDIlist:
		print("Channel "+str(nb))
		y = model.predict(Y_train[0:1])
		#X,y = train_sample([channel],max_sample_len=20)
		#y = y.tolist()
		ch_list = []
		for batch in y:
			ch_list.extend(batch)
		ch_mid = vectList2midi(ch_list)
		channels.append(ch_mid)
		#saveMidi(ch_mid,saveName[:-4]+"_ch"+str(nb)+".mid")
		nb+=1
	print y
	print len(y[0])
	for j in range(1):
		one_hot = np.zeros((50,128))
		for i in range(one_hot.shape[0]):
			index = np.argmax(y[0][i])
			one_hot[i,index]=1

		a = np.asarray(one_hot)
		np.savetxt(saveName[:-4]+'-'+str(j)+'autoencoder.csv', a, delimiter=",")
	
	ch_list = []
	for batch in y:
		ch_list.extend(batch)
	ch_mid = vectList2midi(ch_list)
	channels.append(ch_mid)
	midi = midiChannels2midi(channels)
	saveMidi(midi, saveName)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Learn an AAE')
	parser.add_argument('-l', '--latent', action='store', type=int, default=32,
        	            help='latent dimension in autoencoder')
	parser.add_argument('-s', '--save', action='store', default='',
        	            help='model to load')
	parser.add_argument('-m', '--model', action='store', default='',
        	            help='model to load')
	parser.add_argument('-i', '--interm', action='store', type=int, nargs='+', default=[100,100],
        	            help='intermediate dimension in autoencoder')
	parser.add_argument('-o', '--mlen', action='store', type=int, default=20,
        	            help='intermediate dimension in autoencoder')

	args = parser.parse_args()

	latent_dim = args.latent
	dim_list = args.interm
	saveName = args.save
	max_len = args.mlen

	model_name = args.model
	print model_name
	dir_name=''

	dec_name = model_name+'decoder_save.h5'
	auto_name = model_name+'autoencoder_final.h5'
	
	generateMIDI(dir_name, dec_name,  saveName, latent_dim, dim_list, max_len)
	generateMIDIcsv(dir_name, dec_name,  saveName, latent_dim, dim_list, max_len)

	midiInput = '../MIDI/test/A/Aguado_Rondo_in_A_minor.mid'
	generatefromMIDItoCsv(midiInput,dir_name, auto_name,saveName, latent_dim, dim_list, max_len)
