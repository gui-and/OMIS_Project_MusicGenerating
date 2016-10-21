import mido_encodage as me
import os

def splitChannels(midiName,midiSource):
	outDir = 'channels/' + midiSource
	if not os.path.exists(outDir):
	  os.makedirs(outDir)
	
	outDir = outDir + '/' + midiName[:-4]
	if not os.path.exists(outDir):
	  os.makedirs(outDir)
	
	midiFile = 'MIDI/' + midiSource + '/' + midiName
	
	channels,metas=me.parseMidi(midiFile)
	me.saveMidiList(channels,outDir+'/chan_')

if __name__ == "__main__":
	for sourceDir in os.listdir('MIDI'):
		print(sourceDir)
		for midiFile in os.listdir('MIDI/'+sourceDir):
			if not os.path.exists('channels/'+sourceDir+'/'+midiFile[:-4]):
				print('    '+midiFile)
				try:
					splitChannels(midiFile,sourceDir)
				except:
					os.remove('MIDI/'+sourceDir+'/'+midiFile)
					os.rmdir('channels/'+sourceDir+'/'+midiFile[:-4])
					print('      --> REMOVED')