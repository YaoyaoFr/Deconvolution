import sys
sys.path.append('F:\\Program\\Python')
import data.preprocess_data as pr
import data.functions as fun
import tensorflow as tf
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow.python.ops.rnn_cell import BasicRNNCell, BasicLSTMCell
from tensorflow.python.ops import rnn_cell
from datetime import datetime

plan_path = 'F:\\Program\\Python\\Deconvolution\\save_plan_E'
K = 641
rnn_hidden_num = 32
rnn_layer_num = 1

# Convlution Neural Network Parameter
cnn_input_size = K

# Convolution Layer 1
conv1_input_size = cnn_input_size
conv1_in_channel = 1
conv1_out_channel = 32
conv1_kernel_size = 7

# Convolution Layer 2
conv2_in_channel = conv1_out_channel
conv2_out_channel = 16
conv2_kernel_size = 5

# Convolution Layer 3
conv3_in_channel = conv2_out_channel
conv3_out_channel = 1
conv3_kernel_size = 3

cnn_output_size= conv1_input_size - conv1_kernel_size + 1
output_size = 1

# Traning Parameter
learning_rate = 0.01
need_init = True
batch_size = 200
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


# Recurrent Neural Network Layer
lstm = BasicLSTMCell(rnn_hidden_num, state_is_tuple=False)
initial_state = lstm.zero_state(batch_size=batch_size,dtype=tf.float32)
val, final_state = lstm(h_pool3, initial_state)

# Full-connection to Prediction Layer
W_fullC = tf.Variable(tf.truncated_normal([rnn_hidden_num, output_size]), name='weight')
b_fullC = tf.Variable(tf.constant(0.1, shape=[output_size]),name='bias')
prediction = tf.nn.sigmoid(tf.matmul(val, W_fullC) + b_fullC,name='prediction')

# Optimizer
loss_mse=tf.reduce_mean(tf.square(tf.sub(output_place,prediction)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate,name='optimizer')
minimize = optimizer.minimize(loss_mse,name='minimize')

# Accuracy
loss_m=tf.sqrt(loss_mse)
accu = tf.reduce_mean(tf.cast(tf.equal(output_place,tf.round(prediction)),tf.float32), name='accu')

saver=tf.train.Saver(tf.all_variables())
init_op = tf.initialize_all_variables()

restore_i=0
with tf.Session() as sess:
    train_input, train_output, test_input, test_output, BOLD_range, neural_range = pr.preprocess_data(
        batch_size=batch_size, path=path)
    test_input = test_input[0:batch_size, :]
    test_output = test_output[0:batch_size, :]
    print('Data preprocessed.')
    neural_length = train_output.shape[1]

    if need_init:
        sess.run(init_op)
        epoch = 0
        file_incorrect = open(incorrect_path, 'w')
        print('Initialize successful.')
    else:
        restore_i = 26
        file_incorrect = open(incorrect_path,'a')
        saver.restore(sess, plan_path +'/model' + str(restore_i) + '.ckpt')
        print('Model restored.')

    no_of_batches = int(len(train_input) / batch_size)
    accuracy_value = 0
    epoch = restore_i
    i = epoch * no_of_batches
    time_indexs = list(BOLD_range.keys())
    while accuracy_value < 0.9:
        ptr = 0
        accuracy_test = np.zeros([no_of_batches, len(time_indexs)])
        loss_test = np.zeros([no_of_batches, len(time_indexs)])

        accuracy_train_before = np.zeros([no_of_batches, len(time_indexs)])
        loss_train_before = np.zeros([no_of_batches, len(time_indexs)])
        accuracy_train_after = np.zeros([no_of_batches, len(time_indexs)])
        loss_train_after = np.zeros([no_of_batches, len(time_indexs)])

        for batch in range(no_of_batches):
            state = sess.run(lstm.zero_state(batch_size=batch_size, dtype=tf.float32))
            max=0
            min=0
            mean=0

            for time in time_indexs:
                out = train_output[ptr:ptr+batch_size, time:time+output_size].reshape(batch_size, output_size)
                inp = fun.get_output_range_by_input_range(range(time,time+output_size), BOLD_range, train_input,
                                                          batch_size, ptr, 'wide')
                max+=np.max(inp)
                min+=np.min(inp)
                mean+=np.mean(inp)
                loss_train_before[batch, time], accuracy_train_before[batch, time]= sess.run(
                    [loss_m, accu], {initial_state: state, input_place: inp, output_place: out})

                if time == 0:
                    los, acc, _, state = sess.run([loss_m, accu,val,final_state],
                                                 {initial_state: state, input_place: inp, output_place: out})
                else:
                    los, acc, _, state = sess.run([loss_m, accu,minimize,final_state],
                                                  {initial_state:state,input_place:inp, output_place:out})

                loss_train_after[batch, time], accuracy_train_after[batch, time] = sess.run(
                    [loss_m, accu], {initial_state: state, input_place: inp, output_place: out})

            print('Max  {:f}  Min  {:f}  Mean  {:f}'.format(max/len(time_indexs),min/len(time_indexs),mean/len(time_indexs)))
            accuracy_before = accuracy_train_before[batch, :].mean()
            loss_before = loss_train_before[batch, :].mean()

            accuracy_after = accuracy_train_after[batch, :].mean()
            loss_after = loss_train_after[batch, :].mean()

            ptr += batch_size

            # Test step
            loss_value = 0
            time_test_indexs = time_indexs
            state = sess.run(lstm.zero_state(batch_size=batch_size, dtype=tf.float32))
            for time_test in time_test_indexs:
                out = test_output[ :batch_size, time_test:time_test + output_size].reshape(batch_size, output_size)
                inp = fun.get_output_range_by_input_range(range(time_test, time_test + output_size), BOLD_range, test_input,
                                                          batch_size, 0, 'wide')
                loss_test[batch, time_test], accuracy_test[batch, time_test], state = sess.run([loss_m, accu, final_state],
                                                                                               {initial_state:state, input_place: inp, output_place: out})
            accuracy_value = accuracy_test[batch, :].mean()
            loss_value = loss_test[batch, :].mean()
            error_string = 'I - {:2d} Epoch - {:2d} Batch - {:2d} Accuracy {:2.3f}% loss {:f}' \
                .format(i + 1,epoch+1,batch+1, 100 * accuracy_value, loss_value)
            print(error_string+'   Test:  Accuracy - {:2.3f}% to {:2.3f}  loss - {:f} to {:f} improve {:2.3f} , {:f}'
                  .format(100*accuracy_before, 100*accuracy_after, loss_before, loss_after,
                          100*(accuracy_after-accuracy_before),loss_before-loss_after))
            file_incorrect = open(incorrect_path, 'a')
            file_incorrect.write(error_string + '\n')
            file_incorrect.close()
            i += 1

        if (epoch+1) % 1 == 0:
            save_path = saver.save(sess, plan_path +'/model' + str(epoch + 1) + '.ckpt')
            print('Model saved in file', save_path)

        accuracy_detail_path=plan_path+'\\detail\\accuracy'+str(epoch)+'.txt'
        loss_detail_path = plan_path + '\\detail\\loss' + str(epoch) + '.txt'
        accuracy_detail_path_before = plan_path + '\\detail\\accuracy_before' + str(epoch) + '.txt'
        loss_detail_path_before = plan_path + '\\detail\\loss_before' + str(epoch) + '.txt'
        accuracy_detail_path_after = plan_path + '\\detail\\accuracy_after' + str(epoch) + '.txt'
        loss_detail_path_after = plan_path + '\\detail\\loss_after' + str(epoch) + '.txt'
        np.savetxt(accuracy_detail_path, accuracy_test, delimiter=' ', newline='\r\n')
        np.savetxt(loss_detail_path, loss_test, delimiter=' ', newline='\r\n')
        np.savetxt(accuracy_detail_path_before, accuracy_train_before, delimiter=' ', newline='\r\n')
        np.savetxt(loss_detail_path_before, loss_train_before, delimiter=' ', newline='\r\n')
        np.savetxt(accuracy_detail_path_after, accuracy_train_after, delimiter=' ', newline='\r\n')
        np.savetxt(loss_detail_path_after, loss_train_after, delimiter=' ', newline='\r\n')
        epoch += 1



