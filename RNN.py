from preprocess import preprocessMidi
from seq2seq.models import SimpleSeq2Seq
from sklearn.model_selection import train_test_split
import numpy as np

print(":: Data set :")
sample = preprocessMidi("MIDI/storth")

X = np.array(sample)
y = np.array(sample)
X_train,y_train,X_test,y_test = train_test_split(X,y,test_size=0.20)

model = SimpleSeq2Seq(input_shape=(len(sample[0]),128),output_dim=128,output_length=len(sample[0]))

print(":: Compiling model")
model.compile(loss='mse',optimizer='rmsproper')

print(":: Fitting model")
model.fit(X_train,y_train,nb_epoch=5)

print(":: Evaluating model on test data")
loss = model.evaluate(X_test,y_test)

print("% Loss : "+str(loss))