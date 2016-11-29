from mido import MidiFile,MidiTrack,MetaMessage,Message
import numpy as np
from array import array

def parseMidi(midiFile):
	chan_max = -1
	
	channels = []
	metas = []

	mid = MidiFile(midiFile)
	for track in mid.tracks:
		for message in track:
			if isinstance(message,MetaMessage):
				metas.append(message)
			elif message.type != 'sysex':
				chan = message.channel
				while chan > chan_max:
					mid = MidiFile()
					track = MidiTrack()
					mid.tracks.append(track)
					channels.append(mid)
					chan_max+=1
				channels[chan].tracks[0].append(message)
	return (channels,metas)

def note_egal_vect(mid):
	notevect = np.zeros(128,np.int)
	listnote=[]
	for message in mid.tracks[0]:
		if not isinstance(message,MetaMessage):
			if message.time > 0:
				for i in range(message.time):
					listnote.append(np.array(notevect))
			if message.type == 'note_on' and message.velocity != 0:
				notevect[message.note]=1
			if message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
				notevect[message.note]=0
	return listnote

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