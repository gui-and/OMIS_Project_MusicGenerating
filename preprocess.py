from iter import scanDir
from to_train_sample import train_sample

def preprocessMidi(path="./",directory="MIDI"):
	midiList = scanDir(path=path,directory=directory)
	sample = test_sample(midiList)
	return sample

if __name__ == "__main__":
	print("Pre-Processing of Midi files")