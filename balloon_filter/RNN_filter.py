import tensorflow as tf
import numpy as np
import balloon_filter.functions as fun
from balloon_filter.balloon_model import gen_BOLD
from balloon_filter.filter import state_filter
from tensorflow.python.ops.rnn_cell import BasicLSTMCell, MultiRNNCell
from balloon_filter.RNN_structure import RNN_structure, RNN_structure_parameters
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from balloon_filter.parameters import InputParameters, BlockDesignParameters

def RNN_filter():
    plan_path = 'F:/Data/Exp_result/2017.3.8/RNN_filter'
    incorrect_path = plan_path +'/incorrect.txt'
    if not os.path.exists(plan_path):
        os.makedirs(plan_path)

    # Basic Parameter
    need_init = True
    batch_size = 16
    BOLD_size = 1
    v_size = 1
    q_size = 1
    f_size = 1
    s_size = 1
    u_size = 1
    output_size = [v_size + q_size, f_size, s_size]

    # RNN Size
    v_q_hidden_size = 20
    f_hidden_size = 15
    s_hidden_size = 10

    # RNN structures
    v_q_stru_pa = RNN_structure_parameters(input_size=BOLD_size, output_size=v_size+q_size,
                                          hidden_size=v_q_hidden_size, batch_size=batch_size, scope='v_q')
    f_stru_pa = RNN_structure_parameters(input_size=v_q_hidden_size, output_size=f_size,
                                          hidden_size=f_hidden_size, batch_size=batch_size, scope='f')
    s_stru_pa = RNN_structure_parameters(input_size=f_hidden_size, output_size=s_size,
                                          hidden_size=s_hidden_size, batch_size=batch_size, scope='s')

    v_q_structure = RNN_structure(v_q_stru_pa)
    f_structure = RNN_structure(f_stru_pa)
    s_structure = RNN_structure(s_stru_pa)

    saver = tf.train.Saver(tf.all_variables())
    restore_epoch = 0
    with tf.Session() as sess:
        if need_init:
            v_q_structure.initial_all_variable()
            f_structure.initial_all_variable()
            s_structure.initial_all_variable()

            file_incorrect = open(incorrect_path, 'w')
            print('Initialize successful.')
        else:
            restore_epoch = 300
            file_incorrect = open(incorrect_path, 'a')
            saver.restore(sess, plan_path + '/model' + str(restore_epoch) + '.ckpt')
            print('Model resotred.')
        epoch = restore_epoch

        while True:
            # Generate New Noisy Data

            blo_des_pa = BlockDesignParameters
            blo_des_pa.batch_size = batch_size
            [neural, hemodynamic_state, BOLD] = fun.gen_block_design_data(blo_des_pa=blo_des_pa)
            sequence_length = 3001

            predicted_neural = np.zeros(shape=np.shape(neural))
            predicted_state = np.zeros(shape=np.shape(hemodynamic_state))
            predicted_BOLD = np.zeros(shape=np.shape(BOLD))
            loss_pre = np.zeros([sequence_length, 4])


            for i in range(sequence_length):
                output_state = fun.gen_output_state(neural, hemodynamic_state, batch_size=batch_size, i=i)
                state, output, MSE, lr, global_step = v_q_structure.train_time_point(input=BOLD[i, :], output=output_state[:, 2:4])
                pass


            yield [predicted_state, hemodynamic_state], [predicted_BOLD, BOLD]

            loss_pre = np.mean(loss_pre, axis=1)
            error_string = 'Epoch - {:d}    loss-v - {:10f}  loss-q - {:10f}    loss-f - {:10f}     loss-s - {:10f}'\
                .format(epoch, loss_pre[0], loss_pre[1], loss_pre[2], loss_pre[3])
            print(error_string)

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

ani = animation.FuncAnimation(fig, run, RNN_filter(), blit=False,
                              repeat=False, init_func=init)
plt.show()
