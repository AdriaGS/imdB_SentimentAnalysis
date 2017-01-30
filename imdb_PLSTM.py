
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
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')


# LSTM with timegate
model_PLSTM = Sequential()
model_PLSTM.add(Embedding(max_features, 128, dropout=0.2))
model_PLSTM.add(PLSTM(128, dropout_W=0.3, dropout_U=0.3, W_regularizer=l2(0.002),consume_less='gpu'))
model_PLSTM.add(Dense(1,W_regularizer=l2(0.002), activation='sigmoid'))
model_PLSTM.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])

print('Train...')
model_PLSTM.fit(X_train, y_train, nb_epoch=5, batch_size=batch_size,
                validation_data=(X_test, y_test))
score_PLSTM, acc = model_PLSTM.evaluate(X_test, y_test, batch_size=batch_size)

print('Test score:', score_PLSTM)
print('Test accuracy:', acc)

file = open("PLSTM_trainingSetAccuracyRegularized.csv", 'w')

for item in history.history['acc']:
  file.write("%s\n" % item)

file.close()

file = open("PLSTM_validationSetAccuracyRegularized.csv", 'w')

for item in history.history['val_acc']:
  file.write("%s\n" % item)

file.close()

file = open("PLSTM_trainingSetLossRegularized.csv", 'w')

for item in history.history['loss']:
  file.write("%s\n" % item)

file.close()

file = open("PLSTM_validationSetLossRegularized.csv", 'w')

for item in history.history['val_loss']:
  file.write("%s\n" % item)

file.close()
