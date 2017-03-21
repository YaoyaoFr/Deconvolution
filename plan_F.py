import sys
sys.path.append('F:\\Program\\Python')
import data.preprocess_data as pr
import data.functions as fun
import tensorflow as tf
import random
import numpy as np
import math
import matlab.engine
import matplotlib.pyplot as plt
from tensorflow.python.ops.rnn_cell import BasicRNNCell, BasicLSTMCell
from tensorflow.python.ops import rnn_cell
from datetime import datetime

plan_path = 'F:\\Program\\Python\\Deconvolution\\save_plan_F'
hrf=np.loadtxt('hrf.txt',delimiter=' ')
K = 641
rnn_hidden_num = 32
rnn_layer_num = 1

# Convlution Neural Network Parameter
cnn_input_size = K

# Convolution Layer 1
conv1_input_size = cnn_input_size
conv1_in_channel = 1
conv1_out_channel = 4
conv1_kernel_size = 7

# Convolution Layer 2
conv2_in_channel = conv1_out_channel
conv2_out_channel = 8
conv2_kernel_size = 5

# Convolution Layer 3
conv3_in_channel = conv2_out_channel
conv3_out_channel = 1
conv3_kernel_size = 3

cnn_output_size= conv1_input_size - conv1_kernel_size + 1
output_size = 10

# Traning Parameter
learning_rate = 0.1
need_init = True
batch_size = 128
incorrect_path = plan_path+'/incorrect.txt'
path = 'F:\\Program\\Python'



# Input and Output Placeholder
input_place = tf.placeholder(tf.float32, [batch_size, cnn_input_size])
output_place = tf.placeholder(tf.float32, [batch_size, output_size])


# Convolutional Layer
''' Input and Output Placeholder
    1-D Convolution Input :
        [batch, in_width, in_channels] if data_format is 'NHWC'
    kernel :
        [filter_width, in_channels, out_channels]
    padding :
        'SAME' or 'VALID'
'''


# Convolution Layer 1
I_conv1 = tf.reshape(input_place, (batch_size, cnn_input_size, conv1_in_channel))
W_conv1 = fun.weight_variable([conv1_kernel_size, conv1_in_channel, conv1_out_channel], name='W_conv1')
b_conv1 = fun.bias_variable([conv1_out_channel],name='b_conv1')
h_conv1 = tf.nn.relu(fun.conv1d(I_conv1,W_conv1,padding='VALID')+b_conv1,name='h_conv1')
[h_pool1,conv1_output_size] = fun.max_pool_1x2(h_conv1,padding='VALID')

# Convolution Layer 2
I_conv2 = h_pool1
W_conv2 = fun.weight_variable([conv2_kernel_size, conv2_in_channel, conv2_out_channel], name='W_conv2')
b_conv2 = fun.bias_variable([conv2_out_channel], name='b_conv2')
h_conv2 = tf.nn.relu(fun.conv1d(I_conv2, W_conv2, padding='VALID')+b_conv2, name='h_conv2')
[h_pool2, conv2_output_size] = fun.max_pool_1x2(h_conv2, padding='VALID')

I_conv3 = h_pool2
W_conv3 = fun.weight_variable([conv3_kernel_size, conv3_in_channel, conv3_out_channel], name='W_conv3')
b_conv3 = fun.bias_variable([conv3_out_channel], name='b_conv3')
h_conv3 = tf.nn.relu(fun.conv1d(I_conv3, W_conv3, padding='VALID')+b_conv3, name='h_conv3')
[h_pool3, conv3_output_size] = fun.max_pool_1x2(h_conv3, padding='VALID')

# Full-Connection to RNN Input Layer
h_pool3 = tf.reshape(h_pool3,(conv3_output_size[0],conv3_output_size[1]*conv3_output_size[2]))

'''
# Recurrent Neural Network Layer
lstm = BasicLSTMCell(rnn_hidden_num, state_is_tuple=False)
initial_state = lstm.zero_state(batch_size=batch_size,dtype=tf.float32)
val, final_state = lstm(h_pool3, initial_state)
'''

# Full-connection to Prediction Layer
W_fullC = tf.Variable(tf.truncated_normal([conv3_output_size[1]*conv3_output_size[2], output_size]), name='weight')
b_fullC = tf.Variable(tf.constant(0.1, shape=[output_size]),name='bias')
prediction = tf.matmul(h_pool3, W_fullC) + b_fullC

# Optimizer
loss_mse=tf.reduce_mean(tf.square(tf.sub(output_place,prediction)))
optimizer = tf.train.AdamOptimizer(learning_rate,name='optimizer')
minimize = optimizer.minimize(loss_mse,name='minimize')

# Accuracy
loss_m=tf.sqrt(loss_mse)
accu = tf.reduce_mean(tf.cast(tf.equal(output_place,tf.round(prediction)),tf.float32), name='accu')

saver=tf.train.Saver(tf.all_variables())
init_op = tf.initialize_all_variables()

restore_i=0
with tf.Session() as sess:
    #_, _, test_input, test_output, BOLD_range, _ = pr.preprocess_data(
        # batch_size=batch_size, path=path)
    #test_input = test_input[0:batch_size, :]
    #test_output = test_output[0:batch_size, :]
    #print('Data preprocessed.')
    input_queue=list()
    output_queue=list()

    if need_init:
        sess.run(init_op)
        epoch = 0
        file_incorrect = open(incorrect_path, 'w')
        current_state=None
        print('Initialize successful.')
    else:
        restore_i = 40000
        current_state=np.loadtxt('current_state.txt',delimiter=' ')
        input_queue=np.loadtxt('input.txt',delimiter=' ')
        output_queue=np.loadtxt('output.txt',delimiter=' ')
        file_incorrect = open(incorrect_path,'a')
        saver.restore(sess, plan_path +'/model' + str(restore_i) + '.ckpt')
        print('Model restored.')

    engine = matlab.engine.start_matlab()
    accuracy_value = 0
    loss_value=0;
    epoch = restore_i
    # time_indexs = list(BOLD_range.keys())
    accuracy_mean = 0
    loss_mean = 0
    # state = sess.run(lstm.zero_state(batch_size=batch_size, dtype=tf.float32))
    # eng=matlab.engine.start_matlab()
    while accuracy_mean < 0.9:

        if len(input_queue)<cnn_input_size:
            #[input_queue, output_queue, current_state] = fun.generate_data_balloone_model(input_queue, output_queue, current_state, eng=eng)
           [input_queue, output_queue, current_state] = fun.generateData(input_queue, output_queue,current_state=current_state, hrf=hrf)

        inp=np.transpose(input_queue[0:cnn_input_size+output_size])
        out=np.reshape(output_queue[0:output_size], (batch_size, output_size))
        input_queue.pop(0)

        acc = 0
        i=0
        '''
        inp=np.transpose(output_queue[0:cnn_input_size])
        out=np.reshape(input_queue[cnn_input_size-1],(batch_size,1))
        '''
        while acc<0.9:
            los, acc = sess.run([loss_m, accu], {input_place: inp, output_place: out})
            _ = sess.run([minimize], {input_place: inp, output_place: out})
            los2, acc2 = sess.run([loss_m, accu], {input_place: inp, output_place: out})
            print('I - {:d}  Loss1 - {:f}  Loss2 - {:f}  Acc1 - {:f}  Acc2 - {:f}'.format(i, los, los2, acc, acc2))
            i += 1
        accuracy_value += acc
        loss_value += los

        if (epoch+1) % 1000 == 0:
            accuracy_mean = accuracy_value/1000;
            loss_mean = loss_value/1000;
            error_string = 'Epoch - {:2d}  Accuracy {:2.3f}% loss {:f}' \
                .format(epoch + 1, 100 * accuracy_mean, loss_mean)
            accuracy_value = 0
            loss_value = 0
            # Test step
            print(error_string)
            file_incorrect = open(incorrect_path, 'a')
            file_incorrect.write(error_string + '\n')
            file_incorrect.close()
            if (epoch+1) % 100000 ==0:
                save_path = saver.save(sess, plan_path +'/model' + str(epoch + 1) + '.ckpt')
                np.savetxt('current_state.txt',current_state,delimiter=' ',newline='\r\n')
                np.savetxt('input.txt',input_queue,delimiter=' ',newline='\r\n')
                np.savetxt('output.txt',output_queue,delimiter=' ',newline='\r\n')
                print('Model saved in file', save_path)

        epoch += 1
'''
        accuracy_detail_path=plan_path+'\\detail\\accuracy'+str(epoch)+'.txt'
        loss_detail_path = plan_path + '\\detail\\loss' + str(epoch) + '.txt'
        accuracy_detail_path_before = plan_path + '\\detail\\accuracy_before' + str(epoch) + '.txt'
        loss_detail_path_before = plan_path + '\\detail\\loss_before' + str(epoch) + '.txt'
        accuracy_detail_path_after = plan_path + '\\detail\\accuracy_after' + str(epoch) + '.txt'
        loss_detail_path_after = plan_path + '\\detail\\loss_after' + str(epoch) + '.txt'
        np.savetxt(accuracy_detail_path, accuracy_test, delimiter=' ', newline='\r\n')
        np.savetxt(loss_detail_path, loss_test, delimiter=' ', newline='\r\n')
        epoch += 1
'''


