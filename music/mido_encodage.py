from mido import MidiFile,MidiTrack,MetaMessage,Message
import numpy as np
from array import array

def parseMidi(midiFile,allowMultipleNotesOnTempo=False):
	chan_max = -1
	
	channels = []
	metas = []

	mid = MidiFile(midiFile)
	for track in mid.tracks:
		alreadyANoteOnTempo = [False] * (chan_max+1)
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
					alreadyANoteOnTempo.append(False)
					chan_max+=1
				if message.type == 'note_on' and message.velocity != 0 and message.time == 0 and alreadyANoteOnTempo[chan]:
					if not allowMultipleNotesOnTempo:
						#raise Exception("multiple notes on a tempo")
						continue
				elif message.type == 'note_on' and message.velocity != 0:
					alreadyANoteOnTempo[chan] = True
				elif message.time > 0:
					alreadyANoteOnTempo[chan] = False
				channels[chan].tracks[0].append(message)
	return (channels,metas)

def note2vect(note):
	res = np.zeros(128,np.int)
	res[note]=1
	return res

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
					listnote.append(np.array(notevect))
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

def note_egal_int(mid,max_len=0,allowNoteOnSeveralTempos=False):
	note = 0
	listnote=[]
	listsample=[]
	counter=0
	for message in mid.tracks[0]:
		if not isinstance(message,MetaMessage):
			if allowNoteOnSeveralTempos:
				if message.time > 0:
					for i in range(message.time):
						listnote.append(note)
						counter+=1
						if max_len > 0 and counter >= max_len:
							listsample.append(listnote)
							listnote=[]
							counter=0
				if message.type == 'note_on' and message.velocity != 0:
					note=message.note
			else:
				if message.time > 0:
					listnote.append(note)
					counter+=1
					if max_len > 0 and counter >= max_len:
						listsample.append(listnote)
						listnote=[]
						counter=0
				if message.type == 'note_on' and message.velocity != 0:
					note=message.note
	else:
		try:
			if message.time == 0:
				listnote.append(note)
		except NameError:
			raise Exception("Empty MIDI file")
	listsample.append(listnote)
	return listsample

def vect2note(vect):
	return np.asscalar(np.argmax(vect))

def vectList2midi(vectList):
	mid = newMidiFile()
	for vect in vectList:
		note=vect2note(vect)
		mid.tracks[0].append(Message('note_on',note=note,time=0))
		mid.tracks[0].append(Message('note_off',note=note,time=200))
	return mid

def midiChannels2midi(channels):
	mid = newMidiFile()
	mid.tracks.pop()
	for channel in channels:
		track = channel.tracks[0]
		mid.tracks.append(track)
	return mid

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
