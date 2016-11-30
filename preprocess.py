from iter import scanDir
from to_train_sample import train_sample

def preprocessMidi(directory="130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive"):
	print(":: Pre-processing MIDI files from "+directory)
	midiList = scanDir(directory=directory)
	sample = train_sample(midiList)
	return sample

if __name__ == "__main__":
	print("Pre-Processing of Midi files")
