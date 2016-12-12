import mido_encodage as me
import numpy as np

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
	sampleArray = np.zeros((counter,max_len,1),np.int)
	print(":: Formatting sample")
	for mid in range(counter):
		for tempo in range(len(sampleList[mid])):
			sampleArray[mid,tempo] = sampleList[mid][tempo]
	print(":: Formatted")
	return sampleArray
