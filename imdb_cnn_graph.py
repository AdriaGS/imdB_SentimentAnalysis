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
from keras.layers import Embedding, Flatten
from keras.layers import Convolution1D, GlobalMaxPooling1D, MaxPooling1D
from keras.datasets import imdb
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adagrad, adamax, rmsprop, adam, adadelta, nadam, SGD


# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
nb_filter = 250
filter_length = 5
filter_length2 = 4
filter_length3 = 3
hidden_dims = 250
nb_epoch = 15 

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

# model.add(Flatten())

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
              optimizer='adamax',
              metrics=['accuracy'])
history = model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))

file = open("CNN_trainingSetAccuracyRegularized_1conv1D.csv", 'w')

for item in history.history['acc']:
  file.write("%s\n" % item)

file.close()

file = open("CNN_validationSetAccuracyRegularized_1conv1D.csv", 'w')

for item in history.history['val_acc']:
  file.write("%s\n" % item)

file.close()

file = open("CNN_trainingSetLossRegularized_1conv1D.csv", 'w')

for item in history.history['loss']:
  file.write("%s\n" % item)

file.close()

file = open("CNN_validationSetLossRegularized_1conv1D.csv", 'w')

for item in history.history['val_loss']:
  file.write("%s\n" % item)

file.close()

#########################################################################################################
#########################################################################################################
#########################################################################################################

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

model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length2,
                        border_mode='valid',
                        activation='linear',
                        subsample_length=1))

model.add(LeakyReLU(alpha=0.01))

# model.add(Flatten())

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
              optimizer='adamax',
              metrics=['accuracy'])
history = model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))

file = open("CNN_trainingSetAccuracyRegularized_2conv1D.csv", 'w')

for item in history.history['acc']:
  file.write("%s\n" % item)

file.close()

file = open("CNN_validationSetAccuracyRegularized_2conv1D.csv", 'w')

for item in history.history['val_acc']:
  file.write("%s\n" % item)

file.close()

file = open("CNN_trainingSetLossRegularized_2conv1D.csv", 'w')

for item in history.history['loss']:
  file.write("%s\n" % item)

file.close()

file = open("CNN_validationSetLossRegularized_2conv1D.csv", 'w')

for item in history.history['val_loss']:
  file.write("%s\n" % item)

file.close()


#########################################################################################################
#########################################################################################################
#########################################################################################################

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

model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length2,
                        border_mode='valid',
                        activation='linear',
                        subsample_length=1))
                        
model.add(LeakyReLU(alpha=0.01))

model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length3,
                        border_mode='valid',
                        activation='linear',
                        subsample_length=1))
                        
model.add(LeakyReLU(alpha=0.01))

# model.add(Flatten())

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
              optimizer='adamax',
              metrics=['accuracy'])
history = model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))

file = open("CNN_trainingSetAccuracyRegularized_3conv1D.csv", 'w')

for item in history.history['acc']:
  file.write("%s\n" % item)

file.close()

file = open("CNN_validationSetAccuracyRegularized_3conv1D.csv", 'w')

for item in history.history['val_acc']:
  file.write("%s\n" % item)

file.close()

file = open("CNN_trainingSetLossRegularized_3conv1D.csv", 'w')

for item in history.history['loss']:
  file.write("%s\n" % item)

file.close()

file = open("CNN_validationSetLossRegularized_3conv1D.csv", 'w')

for item in history.history['val_loss']:
  file.write("%s\n" % item)

file.close()


#########################################################################################################
#########################################################################################################
#########################################################################################################

