# -*- coding: utf-8 -*-

from __future__ import print_function

import csv
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pickle
import random

# Training Parameters
learning_rate = 0.001
training_steps = 100000
batch_size = 20
display_step = 200
lossMatrix = np.zeros((training_steps))
lossMatrixTest = np.zeros((2150))

# Network Parameters
num_input = 10
timesteps = 10
num_hidden = 32 # hidden layer num of features
num_output = 10

tb_csv = open("H5_Data/PIO/2006.03.23-29/5IM5N11-VbT731.csv", "r", encoding="utf_8")
reader = csv.reader(tb_csv)
header = next(reader)
dataset = []
for row in reader:
    dataset.append(row)
    
tb_csv = open("H5_Data/PIO/2006.06.01-15/5IM5N11-VbT.csv", "r", encoding="utf_8")
reader = csv.reader(tb_csv)
header = next(reader)
datasetAb = []
for row in reader:
    if row[0] != "00:00.0" and row[0] != "10:00.0" and row[0] != "20:00.0" and row[0] != "30:00.0" and row[0] != "40:00.0" and row[0] != "50:00.0":
        continue
    datasetAb.append(row)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_output])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_output]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_output]))
}

def make_batch(batch_size):
    batch = np.zeros((batch_size, 10, 10))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 10))
    output = np.array(output, dtype=np.float32)
    for i in range(batch_size):
        index = random.randint(0, 990)
        for k in range(10):
            output[i, k] = dataset[index + 10][k + 1]
        for k in range(10):
            for j in range(10):
                batch[i, j, k] = dataset[index + j][k + 1]
    return batch, output

def make_batch_test(batch_size, startF):
    batch = np.zeros((batch_size, 10, 10))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 10))
    output = np.array(output, dtype=np.float32)
    for i in range(batch_size):
        index = startF
        for k in range(10):
            output[i, k] = datasetAb[index + 10][k + 1]
        for k in range(10):
            for j in range(10):
                batch[i, j, k] = datasetAb[index + j][k + 1]
    return batch, output

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.nn.sigmoid(tf.matmul(outputs[-1], weights['out']) + biases['out'])


logits = RNN(X, weights, biases)
prediction = logits

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.square(logits - Y))

optimizer = tf.train.AdamOptimizer(1e-4)
train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = make_batch(batch_size)
        
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        # Calculate batch loss and accuracy
        loss = sess.run([loss_op], feed_dict={X: batch_x, Y: batch_y})
        lossMatrix[step-1] = loss[0]
        if step % display_step == 0 or step == 1:
            
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss[0]))
    for step in range(2150):
        batch_x, batch_y = make_batch_test(batch_size, step)
        lossMatrixTest[step] = sess.run([loss_op], feed_dict={X: batch_x, Y: batch_y})[0]

    print("Optimization Finished!")
    saver = tf.train.Saver()
    saver.save(sess, "./LSTMmodel.ckpt")

LSTMdict = {"lossMatrix":lossMatrix}
pickle.dump(LSTMdict, open('LSTMdict.pickle', mode='wb'))