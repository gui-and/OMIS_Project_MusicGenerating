from iter import scanDir
from to_train_sample import train_sample

def preprocessMidi(directory="130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive",
                   verbose=1,removeExceptions=False,max_sample_len=0,
                   allowMultipleNotesOnTempo=False,allowNoteOnSeveralTempos=False):
	print(":: Pre-processing MIDI files from "+directory)
	midiList = scanDir(directory=directory,verbose=verbose,removeExceptions=removeExceptions,allowMultipleNotesOnTempo=allowMultipleNotesOnTempo)
	return train_sample(midiList,max_sample_len=max_sample_len,allowNoteOnSeveralTempos=allowNoteOnSeveralTempos)

if __name__ == "__main__":
	print("Pre-Processing of Midi files")
