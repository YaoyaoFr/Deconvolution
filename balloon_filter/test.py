import tensorflow as tf
import numpy as np
import balloon_filter.functions as fun
from balloon_filter.balloon_model import gen_BOLD
from balloon_filter.filter import state_filter
from balloon_filter.parameters import InputParameters
from balloon_filter.parameters import BlockDesignParameters
from tensorflow.python.ops.rnn_cell import BasicLSTMCell, BasicRNNCell
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import os


def RNN_structure(input_size, output_size, batch_size, hidden_size, cell_type='LSTM'):
    #   Recurrent Neural Network Cell
    if cell_type is 'LSTM':
        cell = BasicLSTMCell(hidden_size, state_is_tuple=True)
    elif cell_type is 'RNN':
        cell = BasicRNNCell(hidden_size)

    # Place holder
    input_place = tf.placeholder(dtype=tf.float64, shape=[batch_size, input_size],
                                      name='Placeholder/Input')
    output_place = tf.placeholder(dtype=tf.float64, shape=[batch_size, output_size],
                                       name='Placeholder/Output')

    #   Network Variable
    hidden_state = cell.zero_state(batch_size=batch_size, dtype=tf.float64)

    #   Feedforward Process
    value, hidden_state_c = cell(input_place, hidden_state)
    return cell, input_place, output_place, hidden_state_c, hidden_state, value


batch_size = 1
input_size = 1
output_size = 2
hidden_size = 15
with tf.variable_scope('LSTM_V_Q'):
    vq_cell, vq_input_place, vq_output_place, vq_hidden_state_c, vq_hidden_state, vq_value = RNN_structure(input_size=input_size, output_size=output_size,
                                                                  batch_size=batch_size, hidden_size=hidden_size)

init_op = tf.initialize_all_variables()




with tf.Session() as sess:
    sess.run(init_op)
    current_state = sess.run(vq_cell.zero_state(batch_size=batch_size, dtype=tf.float64))
    input_value = np.random.normal(size=[batch_size, input_size])
    val, current_state = sess.run([vq_value, vq_hidden_state_c], feed_dict={
        vq_input_place : input_value,
        vq_hidden_state : current_state
    })

    print(val, current_state)
