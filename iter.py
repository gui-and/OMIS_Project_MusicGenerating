import mido_encodage as me
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
			midiList.append(channel)
		if verbose:
			print("Added "+midiFile+" ** "+str(len(channels))+" channels")
	except:
		if removeExceptions:
			os.remove(midiFile)
			print(midiFile+" REMOVED")
		else:
			print(midiFile+" is not correct")


if __name__ == "__main__":
	print("Scan a directory and add every channel of every MIDI file of this directory and its sub-directories to a list.")