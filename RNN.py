from preprocess import preprocessMidi
from seq2seq.models import SimpleSeq2Seq
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.embeddings import Embedding
import numpy as np

print(":: IMPORTING DATA SET")
X,y = preprocessMidi("MIDI/test",verbose=1,removeExceptions=False,max_sample_len=100,allowMultipleNotesOnTempo=False,allowNoteOnSeveralTempos=False)

if len(X) == 0:
  raise Exception("The sample is empty.")

y = np.concatenate((y,y),axis=2) # Can't have a 1-dim output

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)

print("Data set splitted into train and test data")

model = Sequential()
model.add(Embedding(input_dim=128,output_dim=128))
model.add(SimpleSeq2Seq(input_dim=128,output_dim=2,output_length=len(X[0])))

print(":: COMPILING MODEL")
model.compile(loss='mse',optimizer='rmsprop')

print(":: FITTING MODEL")
model.fit(X_train,y_train,nb_epoch=5)

print(":: EVALUATING MODEL ON TEST DATA")
loss = model.evaluate(X_test,y_test)

print("Loss : "+str(loss))
