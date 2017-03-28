import mido_encodage as me
import numpy as np
import os

def scanDir(directory=".",midiSample=[],verbose=1,removeExceptions=False,allowMultipleNotesOnTempo=False):
	if verbose:
		print("Sanning "+directory)
	for file in os.listdir(directory):
		if os.path.isdir(directory+"/"+file):
			scanDir(directory+"/"+file,midiSample,verbose)
		else:
			addMidiToList(directory+"/"+file,midiSample,verbose,removeExceptions,allowMultipleNotesOnTempo)
	if verbose:
		print("Scanned "+directory)
	return midiSample

def addMidiToList(midiFile,midiList,verbose=1,removeExceptions=False,allowMultipleNotesOnTempo=False):
	if verbose:
		print("Adding "+midiFile)
	try:
		channels,metas=me.parseMidi(midiFile,allowMultipleNotesOnTempo=allowMultipleNotesOnTempo)
		for channel in channels:
			if len(channel.tracks[0]) > 0:
				midiList.append(channel)
		if verbose:
			print("Added "+midiFile+" ** "+str(len(channels))+" channels")
	except:
		if removeExceptions:
			os.remove(midiFile)
			print(midiFile+" REMOVED")
		else:
			print(midiFile+" is not correct")

def train_sample(midiFileList,max_sample_len=0,allowNoteOnSeveralTempos=False):
	sampleList = []
	max_len = 0
	counter = 0
	print(":: Formatting MIDI files")
	for midiFile in midiFileList:
		encodedMidi = me.note_egal_int(midiFile,max_len=max_sample_len,allowNoteOnSeveralTempos=allowNoteOnSeveralTempos)
		for subsample in encodedMidi:
			max_len = max(max_len,len(subsample))
			sampleList.append(subsample)
			counter+=1
	print("Sample shape = "+str(counter)+":"+str(max_len)+":1")
	sampleArray_X = np.zeros((counter,max_len),np.int)
	sampleArray_y = np.zeros((counter,max_len,128),np.int)
	print(":: Formatting sample")
	for mid in range(counter):
		for tempo in range(len(sampleList[mid])):
			sampleArray_X[mid,tempo] = sampleList[mid][tempo]
			sampleArray_y[mid,tempo,sampleList[mid][tempo]] = 1
	print(":: Formatted")
	return sampleArray_X,sampleArray_y

def predict_sample(midiFileList,max_sample_len=0,allowNoteOnSeveralTempos=False):
	sampleList = []
	max_len = 0
	counter = 0
	print(":: Formatting MIDI file")
	for midiFile in midiFileList:
		encodedMidi = me.note_egal_int(midiFile,max_len=max_sample_len,allowNoteOnSeveralTempos=allowNoteOnSeveralTempos)
		for subsample in encodedMidi:
			max_len = max(max_len,len(subsample))
			sampleList.append(subsample)
			counter+=1
	sampleArray_X = np.zeros((counter,max_len),np.int)
	print(":: Formatting sample")
	for mid in range(counter):
		for tempo in range(len(sampleList[mid])):
			sampleArray_X[mid,tempo] = sampleList[mid][tempo]
	print(":: Formatted")
	return sampleArray_X


def preprocessMidi(directory="130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive",
                   verbose=1,removeExceptions=False,max_sample_len=0,
                   allowMultipleNotesOnTempo=False,allowNoteOnSeveralTempos=False):
	print(":: Pre-processing MIDI files from "+directory)
	midiList = scanDir(directory=directory,verbose=verbose,removeExceptions=removeExceptions,allowMultipleNotesOnTempo=allowMultipleNotesOnTempo)
	return train_sample(midiList,max_sample_len=max_sample_len,allowNoteOnSeveralTempos=allowNoteOnSeveralTempos)

if __name__ == "__main__":
	print("Pre-Processing of Midi files")
