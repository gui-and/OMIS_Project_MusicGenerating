from preprocess import preprocessMidi
from seq2seq.models import SimpleSeq2Seq
from sklearn.model_selection import train_test_split
import numpy as np

print(":: IMPORTING DATA SET")
sample = preprocessMidi("MIDI/test")

X = np.array(sample)
y = np.array(sample)

X_train,y_train,X_test,y_test = train_test_split(X,y,test_size=0.20)

print("Data set splitted into train and test data")

model = SimpleSeq2Seq(input_shape=(len(sample[0]),128),output_dim=128,output_length=len(sample[0]))

print(":: COMPILING MODEL")
model.compile(loss='mse',optimizer='rmsproper')

print(":: FITTING MODEL")
model.fit(X_train,y_train,nb_epoch=5)

print(":: EVALUATING MODEL ON TEST DATA")
loss = model.evaluate(X_test,y_test)

print("Loss : "+str(loss))