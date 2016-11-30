# OMIS_Project_MusicGenerating

The main code is **RNN.py** : run it with `python RNN.py`

The whole code is in **python3**

* iter.py is the code for iterating over a directory
* mido_encodage.py is the code for turning MIDIs into lists of message using MIDO
* to_train_sample.py is the code for creating the 3D tensor sample from the list of MIDIs encoded
* preprocess.py is the code for doing all the pre-processing of all MIDI files in a directory, resulting in the 3D tensor input of the RNN
