'''This example demonstrates the use of Convolution1D for text classification.

Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.

'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU

from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 4 

print('Loading data...')
(X_train_ori, y_train), (X_test_ori, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train_ori), 'train sequences')
print(len(X_test_ori), 'test sequences')

for i in range(0,50):
  print(str(len(X_train_ori[i])) , str(y_train[i]))

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train_ori, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test_ori, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen,
                    dropout=0.2))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='linear',
                        subsample_length=1))

model.add(LeakyReLU(alpha=0.01))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims,W_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Activation('linear'))
model.add(LeakyReLU(alpha=0.01))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))

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

file = open("file_CNN", 'w')

for item in hist_norm:
  file.write("%s\n" % item)

file.close()
'''
CNN, = plt.plot(hist_norm, 'b.', label='CNN')

plt.legend(handler_map={CNN: HandlerLine2D(numpoints=2)})
plt.show()
'''

print("Number of predictions:" , len(predictions))