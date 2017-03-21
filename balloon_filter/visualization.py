import tensorflow as tf
import balloon_filter.functions as fun
from tensorflow.python.ops.rnn_cell import BasicLSTMCell
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from balloon_filter.balloon_model import gen_BOLD
from balloon_filter.filter import state_filter


def data_gen_nn(t=0):
    plan_path = 'F:/plan_A'

    input_size = 1
    state_size = 4

    # hidden_size = 5

    learning_rate = 0.01
    need_init = True
    batch_size = 128
    incorrect_path = plan_path+'\\incorrect.txt'

    current_BOLD_place = tf.placeholder(tf.float64, [batch_size, input_size])
    current_state_place = tf.placeholder(tf.float64, [batch_size, state_size])
    predicted_current_BOLD_place = tf.placeholder(tf.float64, [batch_size, input_size])
    predicted_current_state_place = tf.placeholder(tf.float64, [batch_size, state_size])
    predicted_previous_state_place = tf.placeholder(tf.float64, [batch_size, state_size])

    '''
    W_1 = tf.Variable(tf.truncated_normal([state_size, hidden_size], name='weight_1'))
    bias_1 = tf.Variable(tf.truncated_normal([hidden_size]))

    W_2 = tf.Variable(tf.truncated_normal([hidden_size, state_size]))
    bias_2 = tf.Variable(tf.truncated_normal([state_size]))

    input = tf.sub(current_BOLD_place, predicted_current_BOLD_place)

    H_1 = tf.sigmoid(tf.matmul(predicted_previous_state_place, W_1)+bias_1)
    H_2 = tf.sigmoid(tf.matmul(H_1, W_2)+bias_2)
    o = tf.reshape(tf.batch_matmul(tf.reshape(H_2,[batch_size, 4, 1]),
            tf.reshape(input,[batch_size, input_size, 1])),[batch_size, 4]) + predicted_previous_state_place
    '''
    input = tf.sub(current_BOLD_place, predicted_current_BOLD_place)
    W = tf.Variable(tf.truncated_normal([3, 2], dtype=tf.float64), dtype=tf.float64)
    b = tf.Variable(tf.truncated_normal([2], dtype=tf.float64), dtype=tf.float64)

    predicted_previous_f_v_q = tf.slice(predicted_previous_state_place, [0, 1], [batch_size, 3])
    W_o = tf.sigmoid(tf.matmul(predicted_previous_f_v_q, W)+b)
    b_o = tf.Variable(tf.truncated_normal([2], dtype=tf.float64), dtype=tf.float64)

    predicted_current_v_q_place = tf.slice(predicted_previous_state_place, [0, 2], [batch_size, 2])
    o = tf.reshape(tf.batch_matmul(tf.reshape(W_o, [batch_size, 2, 1]), tf.reshape(input, [batch_size, input_size, 1])),
                   [batch_size, 2]) + predicted_current_v_q_place + b_o

    # Applying Different Activation Function

    # Neural Network Output
    o_act_v_q = tf.exp(o)

    # Optimizer
    current_s_f_place = tf.slice(current_state_place, [0, 0], [batch_size, 2])
    current_v_q_place = tf.slice(current_state_place, [0, 2], [batch_size, 2])

    o_act_state = tf.concat(1, [current_s_f_place, o_act_v_q], name='output_state')

    loss_v_q = tf.reduce_mean(tf.square(tf.sub(current_v_q_place, o_act_v_q)))
    loss_v = tf.reduce_mean(tf.square(tf.sub(tf.slice(current_v_q_place,
                                                      [0, 0], [batch_size, 1]), tf.slice(o_act_v_q, [0, 0], [batch_size, 1]))))
    loss_q = tf.reduce_mean(tf.square(tf.sub(tf.slice(current_v_q_place,
                                                      [0, 1], [batch_size, 1]), tf.slice(o_act_v_q, [0, 1], [batch_size, 1]))))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='optimizer')
    minimize = optimizer.minimize(loss_v_q, name='minimize')

    # Accuracy
    corrected_previous_neural_place = tf.placeholder(tf.float32, [batch_size, 1])
    previous_neural_place = tf.placeholder(tf.float32, [batch_size, 1])

    loss_neural = tf.reduce_mean(tf.square(tf.sub(corrected_previous_neural_place, previous_neural_place)))
    true = tf.equal(previous_neural_place, tf.round(corrected_previous_neural_place), name='true')
    accu = tf.reduce_mean(tf.cast(true, tf.float32), name='accu')

    # Saver
    saver = tf.train.Saver(tf.all_variables())
    init_op = tf.initialize_all_variables()

    restore_epoch = 0
    with tf.Session() as sess:
        seq_length = 200
        true_state_sequence = list()
        predicted_state_sequence = list()
        true_neural_sequence = list()
        predicted_neural_sequence = list()
        true_BOLD_sequence = list()
        predicted_BOLD_sequence = list()

        if need_init:
            sess.run(init_op)
            file_incorrect = open(incorrect_path, 'w')

            # Initial Predicted Value
            [predicted_previous_neural, predicted_previous_state,
             predicted_current_state, predicted_previous_BOLD, predicted_current_BOLD] = fun.next_state()

            # Initial Predicted Sequence
            predicted_state_sequence.append(predicted_previous_state)
            predicted_neural_sequence.append(predicted_previous_neural)
            predicted_BOLD_sequence.append(predicted_previous_BOLD)

            # Initial True Value
            [previous_neural, previous_state, current_state, previous_BOLD, current_BOLD] = fun.next_state()

            # Initial True Sequence
            true_state_sequence.append(previous_state)
            true_state_sequence.append(current_state)
            true_neural_sequence.append(previous_neural)
            true_BOLD_sequence.append(previous_BOLD)
            true_BOLD_sequence.append(current_BOLD)

            t = 0
            print('Initialize successful.')
        else:
            restore_epoch = 5000
            file_incorrect = open(incorrect_path, 'a')

            # Initial Predicted Value
            [predicted_previous_neural, predicted_previous_state,
             predicted_current_state, predicted_current_BOLD] = fun.next_state()

            # Initial True Value
            [previous_neural, previous_state, current_state, current_BOLD] = fun.next_state()

            saver.restore(sess, plan_path+'/model'+str(restore_epoch)+'.ckpt')
            print('Model restored.')

        epoch = restore_epoch

        while True:
            # Optimize
            predicted_current_state, los_s, los_v, los_q, _ = sess.run([o_act_state, loss_v_q, loss_v, loss_q, minimize], {
                                                     current_BOLD_place: current_BOLD,
                                                     current_state_place: current_state,
                                                     predicted_current_BOLD_place: predicted_current_BOLD,
                                                     predicted_current_state_place: predicted_current_state,
                                                     predicted_previous_state_place: predicted_previous_state,
                                                 })

            # predicted_state_sequence[len(predicted_state_sequence)-1] = output_current_state
            error_string = 'Epoch - {:2d}  loss_state {:f} loss_v {:f} loss_q {:f}'\
                .format(epoch + 1, los_s, los_v, los_q)
            print(error_string)

            output_current_BOLD = gen_BOLD(predicted_current_state)
            predicted_BOLD_sequence.append(np.reshape(output_current_BOLD, [batch_size, 1]))
            predicted_state_sequence.append(predicted_current_state)

            if len(predicted_state_sequence) > seq_length and len(true_state_sequence) > seq_length:
                del predicted_state_sequence[0]
                del predicted_BOLD_sequence[0]
                del predicted_neural_sequence[0]

                del true_state_sequence[0]
                del true_BOLD_sequence[0]
                del true_neural_sequence[0]

            t += 0.5

            predicted_state_sequence, predicted_neural_sequence = state_filter(predicted_state_sequence,
                                                                               predicted_neural_sequence)
            yield t, \
                  [predicted_neural_sequence, true_neural_sequence], \
                  [predicted_state_sequence, true_state_sequence], \
                  [predicted_BOLD_sequence, true_BOLD_sequence]

            # Next Time

            # Predicted Value
            [predicted_previous_neural, predicted_previous_state, predicted_current_state,
             predicted_previous_BOLD, predicted_current_BOLD] = fun.next_state(predicted_current_state)

            # Append Predicted Sequence
            predicted_neural_sequence.append(predicted_previous_neural)

            # True Value
            [previous_neural, previous_state,
             current_state, previous_BOLD, current_BOLD] = fun.next_state(current_state)
            # Append True Sequence
            true_state_sequence.append(current_state)
            true_neural_sequence.append(previous_neural)
            true_BOLD_sequence.append(current_BOLD)

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


def data_gen_rnn(t=0):
    plan_path = 'F:/plan_A'
    # hidden_size = 5

    learning_rate = 0.01
    need_init = True
    batch_size = 128
    BOLD_size = 1
    v_size = 1
    q_size = 1
    state_size = 4
    hidden_size = 10
    output_size = 2

    incorrect_path = plan_path + '\\incorrect.txt'

    t2_true_BOLD_place = tf.placeholder(tf.float64, [batch_size, BOLD_size])
    t2_pre_BOLD_place = tf.placeholder(tf.float64, [batch_size, BOLD_size])
    t1_out_v_place = tf.placeholder(tf.float64, [batch_size, v_size])
    t1_out_q_place = tf.placeholder(tf.float64, [batch_size, v_size])
    t2_out_v_place = tf.placeholder(tf.float64, [batch_size, v_size])
    t2_out_q_place = tf.placeholder(tf.float64, [batch_size, v_size])

    rnn_cell = BasicLSTMCell(hidden_size, state_is_tuple=False)
    initial_state = rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float64)

    input_place = tf.concat(1, (tf.sub(t2_true_BOLD_place, t2_pre_BOLD_place),
                                t1_out_v_place, t1_out_q_place), name='Input')

    val, final_state = rnn_cell(input_place, initial_state)

    weight = tf.Variable(tf.truncated_normal([hidden_size, output_size], dtype=tf.float64), dtype=tf.float64,
                         name='weight')
    bias = tf.Variable(tf.constant(0.1, shape=[output_size], dtype=tf.float64), dtype=tf.float64, name='bias')

    prediction = tf.exp(tf.matmul(val, weight) + bias, name='prediction')

    output_place = tf.concat(1, (t2_out_v_place, t2_out_q_place))

    # Optimizer
    loss_mse = tf.reduce_mean(tf.square(tf.sub(output_place, prediction)), name='loss_mse')
    loss_v = tf.reduce_mean(tf.square(tf.sub(t2_out_v_place,
                                             tf.slice(prediction, [0, 0], [batch_size, 1]))))
    loss_q = tf.reduce_mean(tf.square(tf.sub(t2_out_q_place,
                                             tf.slice(prediction, [0, 1], [batch_size, 1]))))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='optimizer')
    minimize = optimizer.minimize(loss_mse, name='minimize')

    # Saver
    saver = tf.train.Saver(tf.all_variables())
    init_op = tf.initialize_all_variables()

    restore_epoch = 0
    with tf.Session() as sess:
        seq_length = 200
        true_state_sequence = list()
        predicted_state_sequence = list()
        true_neural_sequence = list()
        predicted_neural_sequence = list()
        true_BOLD_sequence = list()
        predicted_BOLD_sequence = list()

        if need_init:
            sess.run(init_op)
            file_incorrect = open(incorrect_path, 'w')

            # Initial Predicted Value
            [predicted_previous_neural, predicted_previous_state,
             predicted_current_state, predicted_previous_BOLD, predicted_current_BOLD] = fun.next_state()

            # Initial Predicted Sequence
            predicted_state_sequence.append(predicted_previous_state)
            predicted_neural_sequence.append(predicted_previous_neural)
            predicted_BOLD_sequence.append(predicted_previous_BOLD)

            # Initial True Value
            [previous_neural, previous_state, current_state, previous_BOLD, current_BOLD] = fun.next_state()

            # Initial True Sequence
            true_state_sequence.append(previous_state)
            true_state_sequence.append(current_state)
            true_neural_sequence.append(previous_neural)
            true_BOLD_sequence.append(previous_BOLD)
            true_BOLD_sequence.append(current_BOLD)

            t = 0
            print('Initialize successful.')
        else:
            restore_epoch = 5000
            file_incorrect = open(incorrect_path, 'a')

            # Initial Predicted Value
            [predicted_previous_neural, predicted_previous_state,
             predicted_current_state, predicted_current_BOLD] = fun.next_state()

            # Initial True Value
            [previous_neural, previous_state, current_state, current_BOLD] = fun.next_state()

            saver.restore(sess, plan_path + '/model' + str(restore_epoch) + '.ckpt')
            print('Model restored.')

        epoch = restore_epoch
        state = sess.run(rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float32))
        output_v_q = predicted_previous_state[:, 2:4]

        while True:
            # Optimize
            output_v_q, los_v_q, los_v, los_q, state, _ = sess.run([prediction, loss_mse, loss_v, loss_q, final_state, minimize], {
                initial_state: state,
                t2_true_BOLD_place: current_BOLD,
                t2_pre_BOLD_place: predicted_current_BOLD,
                t1_out_v_place: np.reshape(output_v_q[:, 0], [batch_size, 1]),
                t1_out_q_place: np.reshape(output_v_q[:, 1], [batch_size, 1]),
                t2_out_v_place: np.reshape(current_state[:, 2], [batch_size, 1]),
                t2_out_q_place: np.reshape(current_state[:, 3], [batch_size, 1]),
            })

            # predicted_state_sequence[len(predicted_state_sequence)-1] = output_current_state
            error_string = 'Epoch - {:2d}  loss_v_q {:f}  loss_v {:f}  loss_q {:f} ' \
                .format(epoch + 1, los_v_q, los_v, los_q)
            print(error_string)

            predicted_current_state[:, 2:4] = output_v_q
            output_current_BOLD = gen_BOLD(predicted_current_state)
            predicted_BOLD_sequence.append(np.reshape(output_current_BOLD, [batch_size, 1]))
            predicted_state_sequence.append(predicted_current_state)

            if len(predicted_state_sequence) > seq_length and len(true_state_sequence) > seq_length:
                del predicted_state_sequence[0]
                del predicted_BOLD_sequence[0]
                del predicted_neural_sequence[0]

                del true_state_sequence[0]
                del true_BOLD_sequence[0]
                del true_neural_sequence[0]

            predicted_state_sequence, predicted_neural_sequence = state_filter(predicted_state_sequence,
                                                                               predicted_neural_sequence)

            t += 0.5

            yield t, \
                  [predicted_neural_sequence, true_neural_sequence], \
                  [predicted_state_sequence, true_state_sequence], \
                  [predicted_BOLD_sequence, true_BOLD_sequence]
            # Next Time

            # Predicted Value
            [predicted_previous_neural, predicted_previous_state, predicted_current_state,
             predicted_previous_BOLD, predicted_current_BOLD] = fun.next_state(predicted_current_state)

            # Append Predicted Sequence
            predicted_neural_sequence.append(predicted_previous_neural)

            # True Value
            [previous_neural, previous_state,
             current_state, previous_BOLD, current_BOLD] = fun.next_state(current_state)
            # Append True Sequence
            true_state_sequence.append(current_state)
            true_neural_sequence.append(previous_neural)
            true_BOLD_sequence.append(current_BOLD)

            # Writer Accuracy Info
            file_incorrect = open(incorrect_path, 'a')
            file_incorrect.write(error_string + '\n')
            file_incorrect.close()

            # Saveing Neural Network Weights
            if (epoch + 1) % 500 == 0:
                save_path = plan_path + '/model' + str(epoch + 1) + '.ckpt'
                save_path = saver.save(sess, save_path)
                print('Model saved in file', save_path)

            # Iterate Epoch
            epoch += 1


tdata = []
fig = plt.figure()
subplot_name = ['BOLD', 's', 'f', 'v', 'q', 'Accuracy']
axs = list()
lines = list()
for i in range(len(subplot_name)):
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xlabel('t')
    ax.set_ylabel(subplot_name[i])
    axs.append(ax)
    if i < 5:
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
    t, neural, state, BOLD = data
    max_len = 100

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

    datas = [BOLD, s, f, v, q, neural]

    # Neural
    for ax, line, data, type in zip(axs, lines, datas, subplot_name):
        if type is 'Accuracy':
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
            # Axis Range
            min_value = np.minimum(min(data[0]), min(data[1]))
            min_value -= min_value / 10

            max_value = np.maximum(max(data[0]), max(data[1]))
            max_value += max_value / 10

            ax.set_ylim(min_value, max_value)

            for l, d in zip(line, data):
                tdata = np.arange(t, t + len(d) * 0.5, 0.5)
                ax.set_xlim(min(tdata) - 10 , max(tdata) + 10)
                l.set_data(tdata, d)
            ax.figure.canvas.draw()
    return lines,

ani = animation.FuncAnimation(fig, run, data_gen_rnn, blit=False, interval=100,
                              repeat=False, init_func=init)
plt.show()