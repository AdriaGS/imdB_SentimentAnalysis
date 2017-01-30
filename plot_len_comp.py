#!/usr/bin/env python

#hola me llamo alejandro
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

'''
# CNN model
file_CNN = open("file_CNN", 'r')
CNN_hist = [0]*(700)
#x = np.linspace(0,700, 175)

i = 0
aux = float(0)
for line in file_CNN:
	CNN_hist[i] = float(line)
	i = i + 1
	if i == 700:
		break


file_CNN.close()
'''

#  LSTM model
file_LSTM = open("file_LSTM", 'r')
LSTM_hist = [0]*700

i = 0
for line in file_LSTM:
	LSTM_hist[i] = float(line)
	i = i + 1
	if i == 700:
		break

file_LSTM.close()


#  PLSTM model
file_PLSTM = open("file_PLSTM", 'r')
PLSTM_hist = [0]*700

i = 0
for line in file_PLSTM:
	PLSTM_hist[i] = float(line)
	i = i + 1
	if i == 700:
		break

file_PLSTM.close()

#plt.plot(CNN_hist, 'b.')
plt.plot(LSTM_hist, 'r.')
plt.plot(PLSTM_hist, 'g.')

plt.legend(['LSTM', 'PLSTM',], loc='upper left')
plt.title('Comparison LSTM vs PLSTM', fontsize = 22)
plt.ylabel('Error rate', fontsize = 22)
plt.xlabel('Review length', fontsize = 22)	
plt.show()



'''
file_CNN = open("file_CNN", 'r')
CNN_hist = [0]*(175)
x = np.linspace(0,700, 175)

i = 0
aux = float(0)
for line in file_CNN:
	aux = aux + float(line)
	if (i%4) == 3:
		CNN_hist[int(i/4)] = aux/4
		aux = 0
	i = i + 1
	if i == 700:
		break

file_CNN.close()
'''
