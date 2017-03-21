from tensorflow.python.ops.rnn_cell import BasicLSTMCell, BasicRNNCell
import tensorflow as tf
import numpy as np


class RNN_structure:
    def __init__(self, Pa):
        if Pa is None:
            Pa = RNN_structure_parameters
        #   Parameter Reading
        input_size = Pa.input_size
        output_size = Pa.output_size
        hidden_size = Pa.hidden_size
        cell_type = Pa.cell_type
        self.batch_size = Pa.batch_size
        lr = Pa.learning_rate
        decay_step = Pa.decay_step
        decay_rate = Pa.decay_rate

        # Check Arguments
        parameters = [input_size, output_size, hidden_size]
        parameter_strs = ['input size', 'output size', 'hidden size']
        for i in range(len(parameters)):
            if parameters[i] is None:
                raise TypeError('Argument ' + parameter_strs[i] + ' is None')


        #   Recurrent Neural Network Cell
        if cell_type is 'LSTM':
            self.cell = BasicLSTMCell(hidden_size, state_is_tuple=True)
        elif cell_type is 'RNN':
            self.cell = BasicRNNCell(hidden_size)

        #   Place holder
        self.input_place = tf.placeholder(dtype=tf.float64, shape=[self.batch_size, input_size],
                                          name='Placeholder/Input')
        self.output_place = tf.placeholder(dtype=tf.float64, shape=[self.batch_size, output_size],
                                           name='Placeholder/Output')

        #   Network Variable
        self.hidden_state_place = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float64)

        #   Feedforward Process
        self.value, self.hidden_state = self.cell(self.input_place, self.hidden_state_place)

        #   Output
        self.W_output = tf.get_variable(name='Weight/Output', shape=[hidden_size, output_size],
                                        initializer=tf.random_normal_initializer(dtype=tf.float64), dtype=tf.float64)
        self.b_output = tf.get_variable(name='Bias/Output', shape=[output_size],
                                        initializer=tf.random_normal_initializer(dtype=tf.float64),dtype=tf.float64)
        self.output = tf.add(tf.matmul(self.value, self.W_output), self.b_output, name='Feedforward/Output')


        #   Optimize
        self.global_step = tf.get_variable(name='Optimize/Global_step', shape=[],
                                           initializer=tf.constant_initializer(0))
        self.MSE = tf.reduce_mean(tf.square(tf.sub(self.output, self.output_place)), axis=1, name='Optimize/MSE')
        self.learning_rate = tf.train.exponential_decay(lr, global_step=self.global_step, decay_steps=decay_step,
                                                   decay_rate=decay_rate, staircase=True, name='Optimize/Learning_rate')
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate,
                                                           name='Optimize/Optimizer')
        self.minimizer = self.optimizer.minimize(self.MSE, global_step=self.global_step,
                                                 name='Optimize/Minimizer')

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())




    def train_time_point(self, input=None, output=None, state=None, global_step=None):
        with tf.Session() as sess:
            if state is None:
                state = sess.run(self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float64))
            value, state, output, MSE, _ = \
            sess.run(
                [self.value, self.hidden_state, self.output, self.MSE, self.minimizer],
                {
                self.input_place : input,
                self.output_place : output,
                self.hidden_state_place : state,
                self.global_step_place : global_step
            })
            return state, output, MSE


    def test_time_point(self, input=None, output=None, state=None):
        with tf.Session() as sess:
            if state is None:
                state = sess.run(self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float64))
            value, self.hidden_state, RNN_out, MSE = \
            sess.run(
                [self.value, self.hidden_state, self.output, self.MSE],
                {
                self.input_place : input,
                self.output_place : output,
                self.hidden_state_place : state,
                }
            )

    '''
    def train_time_sequence(self, input=None, output=None, state=None):
        if state is None:
            state = self.hidden_state

        [time_length, batch_size, input_size] = np.shape(input)
        [_, _, output_size] = np.shape(output)
        RNN_out = np.zeros(shape=np.shape(output))
        MSE = np.zeros(shape=[time_length, output_size])
        with tf.Session() as sess:
            for inp, out, i in zip(input, output, range(time_length)):
                state, RNN_out, MSE
    '''

class RNN_structure_parameters:


    def __init__(self, input_size=1, output_size=1, hidden_size=10, batch_size=128, cell_type='LSTM',
                 scope='structure', learning_rate=0.05, decay_step=30000, decay_rate=0.96):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.cell_type = cell_type
        self.scope = scope
        self.learning_rate = learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate


