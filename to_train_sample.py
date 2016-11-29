import mido_encodage as me

def train_sample(midiFileList):
	sampleList = []
	max_len = 0
	counter = 0
	for midiFile in midiFileList:
		encodedMidi = me.note_egal_vect(midiFile)
		max_len = max(max_len,len(encodedMidi))
		sampleList.append(encodedMidi)
		counter+=1
	sampleArray = np.zeros((counter,max_len,128),np.int)
	for mid in range(counter):
		for tempo in range(len(sampleList[mid])):
			for note in range(128):
				sampleArray[mid,tempo,note] = sampleList[mid][tempo][note]
	return sampleArray