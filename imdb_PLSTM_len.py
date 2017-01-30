
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.regularizers import l2
from keras.regularizers import l1l2

from phased_lstm_keras.PhasedLSTM import PhasedLSTM as PLSTM

max_features = 20000
maxlen = 200  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(X_train_ori, y_train), (X_test_ori, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train_ori), 'train sequences')
print(len(X_test_ori), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train_ori, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test_ori, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')


# LSTM with timegate
model_PLSTM = Sequential()
model_PLSTM.add(Embedding(max_features, 128, dropout=0.2))
model_PLSTM.add(PLSTM(64, dropout_W=0.3, dropout_U=0.3, W_regularizer=l2(0.08),consume_less='gpu'))
model_PLSTM.add(Dense(1,W_regularizer=l2(0.08), activation='sigmoid'))
model_PLSTM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
model_PLSTM.fit(X_train, y_train, nb_epoch=3, batch_size=batch_size,
                validation_data=(X_test, y_test))
score_PLSTM, acc = model_PLSTM.evaluate(X_test, y_test, batch_size=batch_size)

print('Test score:', score_PLSTM)
print('Test accuracy:', acc)

predictions = model_PLSTM.predict_classes(X_test, batch_size, 1)

max_length = 0
for i in range(0, len(predictions)):
  if(predictions[i] != y_test[i]):
    if max_length < len(X_test_ori[i]):
      max_length = len(X_test_ori[i])
    #print(str(counter) + " : " + str(len(X_test_ori[i])))

histogram_wrong = [0]*max_length
histogram = [0]*max_length

hist_norm = [0]*max_length

print("Max len: " + str(max_length))

counter = 0
for i in range(0, len(predictions)):
  if len(X_test_ori[i]) == 0 or len(X_test_ori[i]) > max_length:
    print("Error: length 0, discard!")

  else:
    histogram[len(X_test_ori[i]) - 1] = histogram[len(X_test_ori[i]) - 1] + 1

  if(predictions[i] != y_test[i]):
    histogram_wrong[len(X_test_ori[i]) - 1] = histogram_wrong[len(X_test_ori[i]) - 1] + 1

for i in range(0, max_length):
  if(histogram[i] != 0):
    hist_norm[i] = histogram_wrong[i]/histogram[i]

file = open("file_PLSTM", 'w')

for item in hist_norm:
  file.write("%s\n" % item)

file.close()