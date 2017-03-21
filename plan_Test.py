import sys
sys.path.append('F:\\Program\\Python')
import data.functions as fun
import tensorflow as tf
import numpy as np

plan_path = 'F:\\Program\\Python\\Deconvolution\\save_plan_F'
hrf=np.loadtxt('hrf.txt',delimiter=' ')
K = 641

# Convlution Neural Network Parameter
cnn_input_size = 1000
num_hidden1 = 128
output_size = K+cnn_input_size-1

# Traning Parameter
global_step = tf.Variable(0,trainable=False)
learning_rate = tf.train.exponential_decay(0.01, global_step, 10000, 0.9, staircase=True)
need_init = True
batch_size = 128
incorrect_path = plan_path+'/incorrect.txt'
path = 'F:\\Program\\Python'

# Input and Output Placeholder
input_place = tf.placeholder(tf.float32, [batch_size, cnn_input_size])
output_place = tf.placeholder(tf.float32, [batch_size, output_size])

# Full-connection to Prediction Layer
W_hidden1 = tf.Variable(tf.truncated_normal([cnn_input_size, num_hidden1]), name='weight')
b_hidden1 = tf.Variable(tf.constant(0.1, shape=[num_hidden1]), name='bias')
hidden1 = tf.matmul(input_place, W_hidden1) + b_hidden1

W_output = tf.Variable(tf.truncated_normal([num_hidden1,output_size]))
b_output = tf.Variable(tf.truncated_normal([output_size]))
prediction = tf.matmul(hidden1 , W_output)+b_output

# Optimizer
loss_mse=tf.reduce_mean(tf.square(tf.sub(output_place,prediction)))
optimizer = tf.train.AdamOptimizer(learning_rate,name='optimizer')
minimize = optimizer.minimize(loss_mse, global_step=global_step, name='minimize')

# Accuracy
loss_m=tf.sqrt(loss_mse)
accu = tf.reduce_mean(tf.cast(tf.equal(output_place,tf.round(prediction)),tf.float32), name='accu')

saver=tf.train.Saver(tf.all_variables())
init_op = tf.initialize_all_variables()

restore_i=0
with tf.Session() as sess:
    # _, _, test_input, test_output, BOLD_range, _ = pr.preprocess_data(
        # batch_size=batch_size, path=path)
    # test_input = test_input[0:batch_size, :]
    # test_output = test_output[0:batch_size, :]
    # print('Data preprocessed.')
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

    accuracy_value = 0
    loss_value=0;
    epoch = restore_i
    # time_indexs = list(BOLD_range.keys())
    accuracy_mean = 0
    loss_mean = 0
    # state = sess.run(lstm.zero_state(batch_size=batch_size, dtype=tf.float32))
    # eng=matlab.engine.start_matlab()
    while accuracy_mean < 0.9:

        while len(input_queue)<output_size:
            #[input_queue, output_queue, current_state] = fun.generate_data_balloone_model(input_queue, output_queue, current_state, eng=eng)
           [input_queue, output_queue, current_state] = fun.generateData(input_queue, output_queue,current_state=current_state, hrf=hrf)

        out = np.transpose(output_queue[0:output_size])
        inp = np.reshape(input_queue[output_size - cnn_input_size:output_size], (batch_size, cnn_input_size))

        output_queue.pop(0)
        input_queue.pop(0)

        acc = 0
        los, acc = sess.run([loss_m , accu],{input_place: inp, output_place: out})
        _ = sess.run([minimize], {input_place: inp, output_place: out})
        los2, acc2 = sess.run([loss_m, accu], {input_place: inp, output_place: out})
        if (epoch+1) % 1000 ==0 :
            print('Epoch - {:d} Loss1 - {:f}  Loss2 - {:f}  Acc1 - {:f}  Acc2 - {:f}  LR - {:f}'.format(
                epoch, los, los2,acc,acc2,learning_rate.eval()))
        accuracy_value += acc
        loss_value += los

        epoch +=1
        '''
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


