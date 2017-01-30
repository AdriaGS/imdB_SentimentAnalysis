'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import sys
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.layers import GRU
from keras.datasets import imdb
from keras.regularizers import l2

max_features = 20000
#maxlen = np.array([80, 160, 240, 320, 400])  # cut texts after this number of words (among top max_features most common words)
maxlen = 80
#n_epoch = np.array([4, 6, 8, 10, 12])
batch_size = 32
#batch_size = np.array([16, 32, 64, 128])

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
model = Sequential()
model.add(Embedding(max_features, 128, dropout=0.2))
model.add(GRU(128, dropout_W=0.3, dropout_U=0.3, W_regularizer=l2(0.01)))  # try using a GRU instead, for fun
model.add(Dense(1, W_regularizer=l2(0.01)))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adamax',
              metrics=['accuracy'])

print('Train...')
history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=5,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

plt.figure(1)
plt.plot(history.history['acc'])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()
