import numpy as np
import keras
from keras import backend as K
from sklearn.metrics import mean_absolute_error
import baseline_naive
from tensorflow import set_random_seed
import tensorflow as tf
from random import seed
import os
os.environ['PYTHONHASHSEED']=str(1)
seed(1)
set_random_seed(2)
np.random.seed(42)

keras.backend.clear_session()
sess=tf.Session(graph=tf.get_default_graph())
K.set_session(sess)

layers = [20, 20] # Number of hidden neuros in each layer of the encoder and decoder

learning_rate = 0.01
decay = 0 # Learning rate decay
optimiser = keras.optimizers.Adam(lr=learning_rate, decay=decay) # Other possible optimiser "sgd" (Stochastic Gradient Descent)

num_input_features = 1 # The dimensionality of the input at each time step. In this case a 1D signal.
num_output_features = 1 # The dimensionality of the output at each time step. In this case a 1D signal.
# There is no reason for the input sequence to be of same dimension as the ouput sequence.
# For instance, using 3 input signals: consumer confidence, inflation and house prices to predict the future house prices.

loss = "mean_absolute_error" # Other loss functions are possible, see Keras documentation.

# Regularisation isn't really needed for this application
lambda_regulariser = 0.000001 # Will not be used if regulariser is None
#regulariser = keras.regularizers.l2(lambda_regulariser) # Possible regulariser: keras.regularizers.l2(lambda_regulariser)
regulariser=None
encoder_inputs = keras.layers.Input(shape=(None, num_input_features))

# Create a list of RNN Cells, these are then concatenated into a single layer
# with the RNN layer.
encoder_cells = []
for hidden_neurons in layers:
  encoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                              kernel_regularizer=regulariser,
                                              recurrent_regularizer=regulariser,
                                              bias_regularizer=regulariser))

encoder = keras.layers.RNN(encoder_cells, return_state=True)

encoder_outputs_and_states = encoder(encoder_inputs)

# Discard encoder outputs and only keep the states.
# The outputs are of no interest to us, the encoder's
# job is to create a state describing the input sequence.
encoder_states = encoder_outputs_and_states[1:]
# The decoder input will be set to zero (see random_sine function of the utils module).
# Do not worry about the input size being 1, I will explain that in the next cell.
decoder_inputs = keras.layers.Input(shape=(None, 1))

decoder_cells = []
for hidden_neurons in layers:
  decoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                              kernel_regularizer=regulariser,
                                              recurrent_regularizer=regulariser,
                                              bias_regularizer=regulariser))

decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)

# Set the initial state of the decoder to be the ouput state of the encoder.
# This is the fundamental part of the encoder-decoder.
decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

# Only select the output of the decoder (not the states)
decoder_outputs = decoder_outputs_and_states[0]

# Apply a dense layer with linear activation to set output to correct dimension
# and scale (tanh is default activation for GRU in Keras, our output sine function can be larger then 1)
decoder_dense = keras.layers.Dense(num_output_features,
                                   activation='linear',
                                   kernel_regularizer=regulariser,
                                   bias_regularizer=regulariser)

decoder_outputs = decoder_dense(decoder_outputs)
# Create a model using the functional API provided by Keras.
# The functional API is great, it gives an amazing amount of freedom in architecture of your NN.
# A read worth your time: https://keras.io/getting-started/functional-api-guide/ 
model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
model.compile(optimizer=optimiser, loss=loss)

import pickle as pkl
import numpy as np
import random
import matplotlib.pyplot as plt

train,val,test=pkl.load(open('synthetic_data.pkl','rb'))
train=train.squeeze()[:,0]
#print(train)
#print(train[:,0])
val=val.squeeze()[:,0]
test=test.squeeze()[:,0]
#train=(train-np.mean(train))/np.std(train)
#val=(val-np.mean(val))/np.std(val)
#test=(test-np.mean(test))/np.std(test)
def gen(data,batch_size, steps_per_epoch,
				input_sequence_length, target_sequence_length):
	num_points = input_sequence_length + target_sequence_length

	while True:
		# Reset seed to obtain same sequences from epoch to epoch
		

		for _ in range(steps_per_epoch):
			signals = np.zeros((batch_size, num_points))
			for i in range(batch_size):
				idx=np.random.randint(0,len(data)-num_points-1)
				#print(data[idx:idx+num_points])
				signals[i]=data[idx:idx+num_points]
			signals = np.expand_dims(signals, axis=2)
			
			encoder_input = signals[:, :input_sequence_length, :]
			decoder_output = signals[:, input_sequence_length:, :]
			
			# The output of the generator must be ([encoder_input, decoder_input], [decoder_output])
			decoder_input = np.zeros((decoder_output.shape[0], decoder_output.shape[1], 1))
			yield ([encoder_input, decoder_input], decoder_output)

in_seq_len=30
targ_seq_len=1
epochs = 10
#train_data_generator = gen(data=train[:600],batch_size=512,
train_data_generator = gen(data=train,batch_size=512,
                                   steps_per_epoch=200,
                                   input_sequence_length=in_seq_len,
                                   target_sequence_length=targ_seq_len)
#val_data_generator = gen(data=np.append(train[600:],val),batch_size=50,
val_data_generator = gen(data=val,batch_size=50,
                                   steps_per_epoch=200,
                                   input_sequence_length=in_seq_len,
                                   target_sequence_length=targ_seq_len)
#test_data_generator = gen(data=test,batch_size=50,
                                  # steps_per_epoch=200,
                                   #input_sequence_length=in_seq_len,
                                  # target_sequence_length=targ_seq_len,seed=1969)

model.fit_generator(train_data_generator, steps_per_epoch=200, epochs=epochs,validation_data=val_data_generator,validation_steps=2)

#(x_encoder_test, x_decoder_test), y_test = next(test_data_generator) # x_decoder_test is composed of zeros.
all=np.append(np.append(train,val),test)
idx=len(train)+len(val)
preds=[]
truths=[]
for i in range(len(test)):
	x_decoder_test = np.zeros((1, 1, 1))
	x_encoder_test = np.expand_dims(np.expand_dims(all[idx+i-in_seq_len:idx+i],axis=1),axis=0)
	#print(x_encoder_test.shape)
	truths.append(all[idx+i])
	preds.append(model.predict([x_encoder_test, x_decoder_test]))
#print(y_test.shape) #(50,1,1)
#print(y_test_predicted.shape) #(50,1,1)
#print(x_encoder_test.shape) #(50,in_seq_len,1)
#groundtruth=np.append(x_encoder_test[0],y_test[0])
mae=mean_absolute_error(np.squeeze(truths), np.squeeze(preds))
print(mae)
_,_,maenaive = baseline_naive.naive_forecast('synthetic_data.pkl')
print('mae frac: ',mae/maenaive)
plt.plot(all,'g',label='truth')
plt.plot(np.append(np.append(train,val),preds),'r',label='pred')
plt.legend()
plt.show()