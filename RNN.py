from preprocess import preprocessMidi
from seq2seq.models import SimpleSeq2Seq
import numpy as np

print(":: Training sample :")
train_sample = preprocessMidi("MIDI/storth")

print(":: Testing sample :")
test_sample = preprocessMidi("MIDI/comyr")

X = np.array(train_sample)
y = np.array(train_sample)

X_test = np.array(test_sample)
y_test = np.array(test_sample)

model = SimpleSeq2Seq(input_shape=(len(sample[0]),128),output_dim=128,output_length=len(sample[0]))

print(":: Compiling model")
model.compile(loss='mse',optimizer='rmsproper')

print(":: Fitting model")
model.fit(X,y,nb_epoch=5)

print(":: Evaluating model on test data")
loss = model.evaluate(X_test,y_test)

print("% Loss : "+str(loss))