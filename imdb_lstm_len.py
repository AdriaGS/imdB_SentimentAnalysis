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
from keras.datasets import imdb
from keras.regularizers import l2

from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

max_features = 20000
#maxlen = np.array([80, 160, 240, 320, 400])  # cut texts after this number of words (among top max_features most common words)
maxlen = 80
#n_epoch = np.array([4, 6, 8, 10, 12])
batch_size = 32
#batch_size = np.array([16, 32, 64, 128])

print('Loading data...')

########
(X_train_ori, y_train), (X_test_ori, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train_ori), 'train sequences')
print(len(X_test_ori), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train_ori, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test_ori, maxlen=maxlen)
#########

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, dropout=0.2))
model.add(LSTM(128, dropout_W=0.5, dropout_U=0.5, W_regularizer=l2(0.01)))  # try using a GRU instead, for fun
model.add(Dense(1, W_regularizer=l2(0.01)))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

predictions = model.predict_classes(X_test, batch_size, 1)

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
    print("Error:")

  else:
    histogram[len(X_test_ori[i]) - 1] = histogram[len(X_test_ori[i]) - 1] + 1

  if(predictions[i] != y_test[i]):
    histogram_wrong[len(X_test_ori[i]) - 1] = histogram_wrong[len(X_test_ori[i]) - 1] + 1

for i in range(0, max_length):
  if(histogram[i] != 0):
    hist_norm[i] = histogram_wrong[i]/histogram[i]

file = open("file_LSTM", 'w')

for item in hist_norm:
  file.write("%s\n" % item)

file.close()

'''
plt.figure(1)
plt.plot(history.history['acc'])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()
'''