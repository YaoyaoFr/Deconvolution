import tensorflow as tf
import numpy as np
import balloon_filter.functions as fun
from balloon_filter.balloon_model import gen_BOLD
from balloon_filter.filter import state_filter
from balloon_filter.parameters import InputParameters
from tensorflow.python.ops.rnn_cell import BasicLSTMCell
import matplotlib.animation as animation
import matplotlib.pyplot as plt

def RNN_Block_filter():
    plan_path = 'F:/plan_A'
    # hidden_size = 5
    need_init = False
    batch_size = 10
    BOLD_size = 1
    v_size = 1
    q_size = 1
    state_size = 4
    hidden_size = 15
    output_size = 2

    incorrect_path = plan_path+'\\incorrect.txt'

    BOLD_place = tf.placeholder(tf.float64, [batch_size, BOLD_size])
    out_state_place = tf.placeholder(tf.float64, [batch_size, output_size])

    rnn_cell = BasicLSTMCell(hidden_size, state_is_tuple=False)
    initial_state = rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float64)

    val, final_state = rnn_cell(BOLD_place, initial_state)

    weight = tf.Variable(tf.truncated_normal([hidden_size, output_size], dtype=tf.float64), dtype=tf.float64, name='weight')
    bias = tf.Variable(tf.constant(0.1, shape=[output_size], dtype=tf.float64), dtype=tf.float64, name='bias')

    prediction = tf.exp(tf.matmul(val, weight) + bias, name='prediction')

    # Optimizer
    loss_mse = tf.reduce_mean(tf.square(tf.sub(out_state_place, prediction)), name='loss_mse')
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.01, global_step, 100000, 0.96, staircase=True)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='optimizer')
    minimize = optimizer.minimize(loss_mse, global_step=global_step, name='minimize')


    # Mean Square Error


    # Saver
    saver = tf.train.Saver(tf.all_variables())
    init_op = tf.initialize_all_variables()

    restore_epoch = 0
    with tf.Session() as sess:

        if need_init:
            sess.run(init_op)
            file_incorrect = open(incorrect_path, 'w')
            print('Initialize successful.')
        else:
            restore_epoch = 4000
            file_incorrect = open(incorrect_path, 'a')
            """
            # Initial Predicted Value
            [predicted_previous_neural, predicted_previous_state,
             predicted_current_state, predicted_current_BOLD] = fun.next_state()

            # Initial True Value
            [previous_neural, previous_state, current_state, current_BOLD] = fun.next_state()
            """
            saver.restore(sess, plan_path+'/model'+str(restore_epoch)+'.ckpt')
            print('Model restored.')
        epoch = restore_epoch

        # Choose Mode Type
        show_type = 'train'

        # Train Variable
        inp_pa = InputParameters
        sequence_length = int(inp_pa.time_length / inp_pa.step_size)
        state = sess.run(rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float64))
        output = np.zeros([sequence_length, batch_size, output_size])
        predicted_BOLD = np.zeros([sequence_length, batch_size, BOLD_size])
        los = np.zeros([sequence_length, ])

        # Test Variable
        test_output = np.zeros([sequence_length, batch_size, output_size])
        test_predicted_BOLD = np.zeros([sequence_length, batch_size, BOLD_size])
        test_los = np.zeros([sequence_length, ])

        while True:
            # Generate New Noisy Data

            [neural, hemodynamic_state, BOLD] = fun.gen_gaussian_bump_data(batch_size=batch_size)
            # Test Model
            test_state = state
            for i in range(sequence_length):
                test_output[i], test_los[i], test_state = sess.run([prediction, loss_mse, final_state],{
                                                                initial_state: test_state,
                                                                BOLD_place: np.reshape(BOLD[i, :, 0], [batch_size, 1]),
                                                                out_state_place: hemodynamic_state[i, :, 2:4]
                })
                test_predicted_BOLD[i] = gen_BOLD(test_output[i, :, 0:4])

            test_predicted_state = np.zeros(np.shape(hemodynamic_state))
            test_predicted_state[:, :, 2:4] = test_output
            test_predicted_neural = np.zeros([np.shape(hemodynamic_state)[0], np.shape(hemodynamic_state)[1], 1])
            for i in range(sequence_length):
                test_predicted_state, test_predicted_neural = state_filter(state=test_predicted_state[:, :, 0:4],
                                                                 neural=test_predicted_neural, index=i, interval=0.1)
            #test_predicted_neural = fun.move_average(test_predicted_neural, step_size=5)

            # Train Model
            for i in range(sequence_length):
                output[i], los[i], state, lr, _ = sess.run([prediction, loss_mse, final_state, learning_rate, minimize], {
                                                         initial_state: state,
                                                         BOLD_place: np.reshape(BOLD[i, :, 0], [batch_size, 1]),
                                                         out_state_place: hemodynamic_state[i, :, 2:4]
                })
                predicted_BOLD[i] = gen_BOLD(output[i, :, 0:4])

            predicted_state =np.zeros(np.shape(hemodynamic_state))
            predicted_state[:, :, 2:4] = output
            predicted_neural = np.zeros([np.shape(hemodynamic_state)[0], np.shape(hemodynamic_state)[1], 1])
            for i in range(sequence_length):
                predicted_state, predicted_neural = state_filter(state=predicted_state[:, :, 0:4], neural=predicted_neural, index=i, interval=0.1)
            #predicted_neural = fun.move_average(predicted_neural, step_size=0)

            # Show data by Show Type
            if show_type is 'train':
                error_string = 'Epoch - {:2d}  max_loss - {:10f}  min_loss - {:10f}  mean_loss - {:10f}  lr  {:10f} ' \
                    .format(epoch + 1, max(los), min(los), np.mean(los), lr)
                print(error_string)
                yield [predicted_state, hemodynamic_state], [predicted_BOLD, BOLD], [predicted_neural, neural]
            elif show_type is 'test':
                error_string = 'Epoch - {:2d}  max_loss - {:10f}  min_loss - {:10f}  mean_loss - {:10f}  lr  {:10f} ' \
                    .format(epoch + 1, max(test_los), min(test_los), np.mean(test_los), lr)
                print(error_string)
                yield [test_predicted_state, hemodynamic_state], [test_predicted_BOLD, BOLD], [test_predicted_neural, neural]

            # Writer Accuracy Info
            file_incorrect = open(incorrect_path, 'a')
            file_incorrect.write(error_string+'\n')
            file_incorrect.close()

            # Saveing Neural Network Weights
            if (epoch+1) % 500 == 0:
                save_path = plan_path+'/model'+str(epoch+1)+'.ckpt'
                save_path = saver.save(sess, save_path)
                print('Model saved in file', save_path)

            # Iterate Epoch
            epoch += 1

inp_pa = InputParameters
tdata = []
fig = plt.figure()
subplot_name = ['BOLD', 's', 'f', 'v', 'q', 'neural']
axs = list()
lines = list()
for i in range(len(subplot_name)):
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xlabel('t')
    ax.set_ylabel(subplot_name[i])
    axs.append(ax)
    if i < 6:
        lines.append([axs[i].plot([], [], lw=2)[0] for _ in range(2)])
    else:
        lines.append(axs[i].plot([], [], lw=2))


def init():
    for ax, line in zip(axs, lines):
        ax.set_ylim(-0.1, 0.1)
        ax.set_xlim(0, 100)
        ax.grid()
        for l in line:
            l.set_data([], [])

    return lines,


def run(data):
    state, BOLD, neural = data
    # Data Pre-processing
    # state
    state = np.array(state)
    s = state[:, :, 0, 0]
    f = state[:, :, 0, 1]
    v = state[:, :, 0, 2]
    q = state[:, :, 0, 3]
    del state

    # BOLD
    BOLD = np.array(BOLD)
    BOLD = BOLD[:, :, 0]

    # Neural
    neural = np.array(neural)
    neural = neural[:, :, 0]

    datas = [BOLD, s, f, v, q, neural]

    # Neural
    for ax, line, data, type in zip(axs, lines, datas, subplot_name):
        '''
        if type is 'Accuracy':
            pass
            accuracy = list()
            ax.set_ylim(-0.1, 1.1)
            predicted_neural = data[0]
            true_neural = data[1]
            for pre_n, tru_n in zip(predicted_neural, true_neural):
                n = np.sum(np.round(pre_n) == tru_n)
                accuracy.append(n / np.shape(pre_n)[0])
            if len(accuracy) > max_len:
                del accuracy[0]
            tdata = np.arange(t, t + len(accuracy) * 0.5, 0.5)
            ax.set_xlim(min(tdata) - 10, max(tdata) + 10)
            line[0].set_data(tdata, accuracy)
            ax.figure.canvas.draw()
        else:
        '''
        # Axis Range
        min_value = np.min(data)
        min_value -= min_value / 10

        max_value = np.max(data)
        max_value += max_value / 10

        ax.set_ylim(min_value, max_value)

        ax.set_xlim(0, inp_pa.time_length)
        tdata = np.arange(0, inp_pa.time_length, inp_pa.step_size)
        for l, d in zip(line, data):
            l.set_data(tdata, d)
        ax.figure.canvas.draw()
    return lines,

ani = animation.FuncAnimation(fig, run, RNN_Block_filter, blit=False,
                              repeat=False, init_func=init)
plt.show()
