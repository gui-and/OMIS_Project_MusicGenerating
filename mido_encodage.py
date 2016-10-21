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

def select_notes(mid):
	res = MidiFile()
	res.tracks.append(MidiTrack())
	for track in mid.tracks:
		for message in track:
			if message.type == 'note_on' or message.type == 'note_off':
				res.tracks[0].append(message)
	return res

msg_note = lambda message:message.note

def note_egal_val(mid):
	return map(msg_note,mid.tracks[0])

def note2vect(note):
	res = np.zeros(128,np.int)
	res[note]=1
	return res

msg2vect = lambda message:note2vect(message.note)

def note_egal_vect(mid):
	return map(msg2vect,mid.tracks[0])

def getMidiFile(midiFile):
	return MidiFile(midiFile)

def newMidiFile():
	mid = MidiFile()
	track = MidiTrack()
	mid.tracks.append(track)
	return mid

def saveMidi(mid,file):
	mid.save(file)

def saveMidiList(midis,base_name='chan_'):
	for i in range(len(midis)):
		mid = midis[i]
		fileName = base_name + str(i) + '.mid'
		mid.save(fileName)

if __name__ == "__main__":
	print("MIDI files encoder using MIDO")