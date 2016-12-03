from mido import MidiFile,MidiTrack,MetaMessage,Message
import numpy as np
from array import array

def parseMidi(midiFile,allowMultipleNotesOnTempo=False):
	chan_max = -1
	
	channels = []
	metas = []

	mid = MidiFile(midiFile)
	for track in mid.tracks:
		alreadyANoteOnTempo = False
		for message in track:
			if isinstance(message,MetaMessage):
				metas.append(message)
			elif message.type != 'sysex':
				if message.type == 'note_on' and message.velocity != 0 and message.time == 0 and alreadyANoteOnTempo:
					if not allowMultipleNotesOnTempo:
						raise Exception("multiple notes on a tempo")
				elif message.type == 'note_on' and message.velocity != 0:
					alreadyANoteOnTempo = True
				elif message.time > 0:
					alreadyANoteOnTempo = False
				chan = message.channel
				while chan > chan_max:
					mid = MidiFile()
					track = MidiTrack()
					mid.tracks.append(track)
					channels.append(mid)
					chan_max+=1
				channels[chan].tracks[0].append(message)
	return (channels,metas)

def note_egal_vect(mid,max_len=0,allowNoteOnSeveralTempos=False):
	notevect = np.zeros(128,np.int)
	listnote=[]
	listsample=[]
	counter=0
	for message in mid.tracks[0]:
		if not isinstance(message,MetaMessage):
			if allowNoteOnSeveralTempos:
				if message.time > 0:
					for i in range(message.time):
						listnote.append(np.array(notevect))
						counter+=1
						if max_len > 0 and counter >= max_len:
							listsample.append(listnote)
							listnote=[]
							counter=0
				if message.type == 'note_on' and message.velocity != 0:
					notevect[message.note]=1
				if message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
					notevect[message.note]=0
			else:
				if message.time > 0:
					listenote.append(np.array(notevect))
					counter+=1
					notevect = np.zeros(128,np.int)
					if max_len > 0 and counter >= max_len:
						listsample.append(listnote)
						listnote=[]
						counter=0
				if message.type == 'note_on' and message.velocity != 0:
					notevect[message.note]=1
	else:
		try:
			if message.time == 0:
				listnote.append(notevect)
		except NameError:
			raise Exception("Empty MIDI file")
	listsample.append(listnote)
	return listsample

def getMidiFile(midiFile):
	return MidiFile(midiFile)

def newMidiFile():
	mid = MidiFile()
	track = MidiTrack()
	mid.tracks.append(track)
	return mid

def saveMidi(mid,file):
	mid.save(file)


if __name__ == "__main__":
	print("MIDI files encoder using MIDO")