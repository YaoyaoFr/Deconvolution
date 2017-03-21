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

def RNN_Block_filter():
    plan_path = 'F:/Data/Exp_result/2017.3.8/block_design'
    if not os.path.exists(plan_path):
        os.makedirs(plan_path)
    # hidden_size = 5
    need_init = True
    batch_size = 16
    BOLD_size = 1
    v_size = 1
    q_size = 1
    f_size = 1
    s_size = 1
    u_size = 1
    v_q_size = 2
    vq_hidden_size = 20

    # RNN CELLs
    f_hidden_size = 15
    s_hidden_size = 15
    u_hidden_size = 10
    state_size = 4
    neural_size = 1
    hidden_size = 30
    output_size = 5
    incorrect_path = plan_path+'\\incorrect.txt'

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.05, global_step, 300000, 0.96, staircase=True)


    with tf.variable_scope('LSTM_v_q'):
        # v q
        vq_cell, vq_input_place, vq_output_place, vq_hidden_state_c, vq_hidden_state, vq_value = \
            RNN_structure(BOLD_size, v_size+q_size, batch_size, vq_hidden_size)
        vq_W = tf.Variable(tf.truncated_normal([vq_hidden_size, v_q_size], dtype=tf.float64), dtype=tf.float64,
                            name='Weight_v_q')
        vq_b = tf.Variable(tf.constant(0.1, shape=[v_q_size], dtype=tf.float64), dtype=tf.float64, name='Bias_v_q')
        vq_prediction_t = tf.exp(tf.matmul(vq_value, vq_W) + vq_b)

        vq_loss = tf.reduce_mean(tf.square(tf.sub(vq_prediction_t, vq_output_place)), 0)
        vq_optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='optimizer_v_q')
        vq_minimizer = vq_optimizer.minimize(vq_loss, global_step=global_step, name='minimize_v_q')

    with tf.variable_scope('LSTM_f'):
        # f
        f_cell, f_input_place, f_output_place, f_hidden_state_c, f_hidden_state, f_value = \
            RNN_structure(vq_hidden_size, f_size, batch_size, f_hidden_size)
        f_W = tf.Variable(tf.truncated_normal([f_hidden_size, f_size], dtype=tf.float64), dtype=tf.float64,
                           name='Weight_f')
        f_b = tf.Variable(tf.constant(0.1, shape=[f_size], dtype=tf.float64), dtype=tf.float64, name='Bias_f')
        f_prediction = tf.exp(tf.matmul(f_value, f_W) + f_b)

        f_loss = tf.reduce_mean(tf.square(tf.sub(f_prediction, f_output_place)), 0)
        f_optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='optimizer_f')
        f_minimizer = f_optimizer.minimize(f_loss, global_step=global_step, name='minimize_fq')

    with tf.variable_scope('LSTM_s'):
        print('Github Test')
        # s
        s_cell, s_input_place, s_output_place, s_hidden_state_c, s_hidden_state, s_value = \
            RNN_structure(f_hidden_size, s_size, batch_size, s_hidden_size)
        s_W = tf.Variable(tf.truncated_normal([s_hidden_size, s_size], dtype=tf.float64), dtype=tf.float64,
                           name='Weight_s')
        s_b = tf.Variable(tf.constant(0.1, shape=[s_size], dtype=tf.float64), dtype=tf.float64, name='Bias_s')
        s_prediction = tf.add(x=tf.matmul(s_value, s_W),y=s_b, name='Output/S')

        s_loss = tf.reduce_mean(tf.square(tf.sub(s_prediction, s_output_place)), 0)
        s_optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='optimizer_s')
        s_minimizer = s_optimizer.minimize(s_loss, global_step=global_step, name='minimize_sq')

    # Saver
    saver = tf.train.Saver(tf.all_variables())
    init_op = tf.initialize_all_variables()

    restore_epoch = 0
    with tf.Session() as sess:

        if need_init:
            init = sess.run(init_op)
            file_incorrect = open(incorrect_path, 'w')
            print('Initialize successful.')
        else:
            restore_epoch = 390
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
        show_type = 'test'
        test_data_type = 'Bump'

        # Train Variable
        inp_pa = InputParameters
        if test_data_type is 'Gaussian':
            block_sequence_length = int(inp_pa.time_length / inp_pa.step_size)
            bump_sequence_length = 3001
        else:
            block_sequence_length = 3001
            bump_sequence_length = int(inp_pa.time_length / inp_pa.step_size)


        while True:
            # Generate New Noisy Data
            blo_des_pa = BlockDesignParameters
            blo_des_pa.batch_size = batch_size
            [block_neural, block_hemodynamic_state, block_raw_BOLD, block_noisy_interpolated_BOLD] = fun.gen_block_design_data(blo_des_pa=blo_des_pa)
            [bump_neural, bump_hemodynamic_state, bump_BOLD] = fun.gen_gaussian_bump_data(batch_size=batch_size)

            if test_data_type is 'Block':
                [test_neural, test_hemodynamic_state, test_raw_BOLD, test_noisy_interpolated_BOLD] = fun.gen_block_design_data(blo_des_pa=blo_des_pa)
                test_sequence_length = block_sequence_length
            elif test_data_type is 'Bump':
                [test_neural, test_hemodynamic_state, test_noisy_interpolated_BOLD] = fun.gen_gaussian_bump_data(batch_size=batch_size)
                test_sequence_length = bump_sequence_length


            #   Train by Bump Data
            bump_los = np.zeros([bump_sequence_length, 4])
            bump_pre_neural = np.zeros([np.shape(bump_hemodynamic_state)[0], np.shape(bump_hemodynamic_state)[1], 1])
            bump_pre_hemodynamic_state = np.zeros(np.shape(bump_hemodynamic_state))
            bump_pre_BOLD = np.zeros([bump_sequence_length, batch_size, BOLD_size])

            vq_bump_state = sess.run(vq_cell.zero_state(batch_size=batch_size, dtype=tf.float64))
            f_bump_state = sess.run(f_cell.zero_state(batch_size=batch_size, dtype=tf.float64))
            s_bump_state = sess.run(s_cell.zero_state(batch_size=batch_size, dtype=tf.float64))

            for i in range(bump_sequence_length):
                output_state = fun.gen_output_state(bump_neural, bump_hemodynamic_state, batch_size=batch_size, i=i)

                # V and Q
                vq_val, vq_pre, bump_los[i, 2:4], vq_bump_state , _, lr= sess.run([vq_value, vq_prediction_t, vq_loss, vq_hidden_state_c, vq_minimizer, learning_rate], {
                    vq_input_place : np.reshape(bump_BOLD[i, :, 0], [batch_size, BOLD_size]),
                    #   vq_input_place :vq_input_value,
                    vq_hidden_state: vq_bump_state,
                    vq_output_place : output_state[:, 2:4]
                })

                # F
                f_pre, bump_los[i, 1], f_bump_state, _ = sess.run([f_prediction, f_loss, f_hidden_state_c, f_minimizer],{
                    f_hidden_state : f_bump_state,
                    f_input_place : vq_bump_state.h,
                    f_output_place: np.reshape(output_state[:, 1], [batch_size, 1])
                })

                # S
                s_pre, bump_los[i, 0], s_bump_state, _ = sess.run([s_prediction, s_loss, s_hidden_state_c, s_minimizer], {
                    s_hidden_state : s_bump_state,
                    s_input_place : f_bump_state.h,
                    s_output_place : np.reshape(output_state[:, 0], [batch_size, 1])
                })

                output = np.concatenate((s_pre, f_pre, vq_pre), axis=1)
                bump_pre_BOLD[i] = gen_BOLD(output[:, 0:4])
                bump_pre_neural, bump_pre_hemodynamic_state = \
                    fun.reset_output_state(output, bump_pre_neural, bump_pre_hemodynamic_state, i, batch_size)


            #   Train by Block Data
            block_los = np.zeros([block_sequence_length, 4])
            block_pre_neural = np.zeros([block_sequence_length, batch_size, neural_size])
            block_pre_hemodynamic_state = np.zeros([block_sequence_length, batch_size, state_size])
            block_pre_BOLD = np.zeros([block_sequence_length, batch_size, BOLD_size])

            vq_block_state = sess.run(vq_cell.zero_state(batch_size=batch_size, dtype=tf.float64))
            f_block_state = sess.run(f_cell.zero_state(batch_size=batch_size, dtype=tf.float64))
            s_block_state = sess.run(s_cell.zero_state(batch_size=batch_size, dtype=tf.float64))

            for i in range(block_sequence_length):
                output_state = fun.gen_output_state(block_neural, block_hemodynamic_state, batch_size=batch_size, i=i)
                # V and Q
                vq_val, vq_pre, block_los[i, 2:4], vq_block_state, _ , lr= \
                    sess.run([vq_value, vq_prediction_t, vq_loss,
                              vq_hidden_state_c, vq_minimizer, learning_rate], feed_dict={
                    vq_input_place : np.reshape(block_noisy_interpolated_BOLD[i, :, 0], [batch_size, BOLD_size]),
                    vq_hidden_state: vq_block_state,
                    vq_output_place : output_state[:, 2:4]
                })

                # F
                f_pre, block_los[i, 1], f_block_state, _ = sess.run([f_prediction, f_loss, f_hidden_state_c, f_minimizer],{
                    f_hidden_state : f_block_state,
                    f_input_place : vq_block_state.h,
                    f_output_place: np.reshape(output_state[:, 1], [batch_size, 1])
                })

                # S
                s_pre, block_los[i, 0], s_block_state, _ =sess.run([s_prediction, s_loss, s_hidden_state_c, s_minimizer], {
                    s_hidden_state : s_block_state,
                    s_input_place : f_block_state.h,
                    s_output_place : np.reshape(output_state[:, 0], [batch_size, 1])
                })

                output = np.concatenate((s_pre, f_pre, vq_pre), axis=1)
                block_pre_BOLD[i] = gen_BOLD(output[:, 0:4])
                block_pre_neural, block_pre_hemodynamic_state = \
                    fun.reset_output_state(output, block_pre_neural, block_pre_hemodynamic_state, i, batch_size)

            #test_predicted_neural = fun.move_average(test_predicted_neural, step_size=5)

            #   Test by Test Data Type


            test_los = np.zeros([test_sequence_length, 4])
            test_pre_neural = np.zeros([test_sequence_length, batch_size, neural_size])
            test_pre_hemodynamic_state = np.zeros([test_sequence_length, batch_size, state_size])
            test_pre_BOLD = np.zeros([test_sequence_length, batch_size, BOLD_size])

            vq_test_state = sess.run(vq_cell.zero_state(batch_size=batch_size, dtype=tf.float64))
            f_test_state = sess.run(f_cell.zero_state(batch_size=batch_size, dtype=tf.float64))
            s_test_state = sess.run(s_cell.zero_state(batch_size=batch_size, dtype=tf.float64))

            for i in range(test_sequence_length):
                output_state = fun.gen_output_state(test_neural, test_hemodynamic_state, batch_size=batch_size,
                                                    i=i)
                # V and Q
                vq_val, vq_pre, test_los[i, 2:4], vq_test_state, _, lr = \
                    sess.run([vq_value, vq_prediction_t, vq_loss,
                              vq_hidden_state_c, vq_minimizer, learning_rate], feed_dict={
                        vq_input_place: np.reshape(test_noisy_interpolated_BOLD[i, :, 0], [batch_size, BOLD_size]),
                        vq_hidden_state: vq_test_state,
                        vq_output_place: output_state[:, 2:4]
                    })

                # F
                f_pre, test_los[i, 1], f_test_state, _ = sess.run(
                    [f_prediction, f_loss, f_hidden_state_c, f_minimizer], {
                        f_hidden_state: f_test_state,
                        f_input_place: vq_test_state.h,
                        f_output_place: np.reshape(output_state[:, 1], [batch_size, 1])
                    })

                # S
                s_pre, test_los[i, 0], s_test_state, _ = sess.run(
                    [s_prediction, s_loss, s_hidden_state_c, s_minimizer], {
                        s_hidden_state: s_test_state,
                        s_input_place: f_test_state.h,
                        s_output_place: np.reshape(output_state[:, 0], [batch_size, 1])
                    })

                output = np.concatenate((s_pre, f_pre, vq_pre), axis=1)
                test_pre_BOLD[i] = gen_BOLD(output[:, 0:4])
                test_pre_neural, test_pre_hemodynamic_state = \
                    fun.reset_output_state(output, test_pre_neural, test_pre_hemodynamic_state, i, batch_size)


            # Show data by Show Type

            block_los = np.mean(block_los, 0)
            block_error_string = 'Block:  Epoch - {:2d}  MSE:  s - {:10f}  f - {:10f}  v - {:10f} q - {:10f}  lr - {:10f} ' \
                .format(epoch + 1, block_los[0], block_los[1], block_los[2], block_los[3], lr)
            print(block_error_string)

            bump_los = np.mean(bump_los, 0)
            bump_error_string = 'Bump:   Epoch - {:2d}  MSE:  s - {:10f}  f - {:10f}  v - {:10f} q - {:10f}  lr - {:10f} ' \
                .format(epoch + 1, bump_los[0], bump_los[1], bump_los[2], bump_los[3], lr)
            print(bump_error_string)

            test_los = np.mean(test_los, 0)
            test_error_string = 'Test:   Epoch - {:2d}  MSE:  s - {:10f}  f - {:10f}  v - {:10f} q - {:10f}  lr - {:10f} ' \
                .format(epoch + 1, test_los[0], test_los[1], test_los[2], test_los[3], lr)
            print(test_error_string)
            print('\n')


            yield [test_pre_hemodynamic_state, test_hemodynamic_state], [test_pre_BOLD, test_noisy_interpolated_BOLD]

            # Writer Accuracy Info
            file_incorrect = open(incorrect_path, 'a')
            file_incorrect.write(block_error_string+'\n')
            file_incorrect.write(bump_error_string+'\n')
            file_incorrect.close()


            # Saveing Neural Network Weights
            if (epoch+1) % 50 == 0:
                save_path = plan_path+'/model'+str(epoch+1)+'.ckpt'
                save_path = saver.save(sess, save_path)
                print('Model saved in file', save_path)

            # Iterate Epoch
            epoch += 1

inp_pa = InputParameters
tdata = []
fig = plt.figure()
subplot_name = ['BOLD', 's', 'f', 'v', 'q']
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
    state, BOLD = data
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
    #neural = np.array(neural)
    #neural = neural[:, :, 0]

    datas = [BOLD, s, f, v, q]

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

        """
        ax.set_xlim(0, inp_pa.time_length)
        tdata = np.arange(0, inp_pa.time_length, inp_pa.step_size)
        """
        time_length = 59
        ax.set_xlim(0, time_length)
        tdata = np.arange(0, time_length + 0.1, 0.1)
        for l, d in zip(line, data):
            l.set_data(tdata, d[0:len(tdata)])
        ax.figure.canvas.draw()
    return lines,

ani = animation.FuncAnimation(fig, run, RNN_Block_filter, blit=False,
                              repeat=False, init_func=init)
plt.show()
