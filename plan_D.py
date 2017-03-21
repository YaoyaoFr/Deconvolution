import sys
import data.preprocess_data as pr
import data.functions as fun
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import BasicRNNCell
from tensorflow.python.ops import rnn_cell
from datetime import datetime
sys.path.append('F:\\Program\\Python')

plan_path = 'save_plan_D'
K = 641
num_hidden = 300
num_layer = 1
# step_size : the input length of every time step, it must be the divisor of 640(K-1)
output_size = K
input_size = K

learning_rate = 0.2
need_init = False
batch_size = 200
incorrect_path = plan_path+'/incorrect.txt'


train_input, train_output, test_input, test_output, BOLD_range, neural_range = pr.preprocess_data(batch_size=batch_size)
test_input = test_input[0:batch_size, :]
test_output = test_output[0:batch_size, :]
# input_size=fun.get_BOLD_length_by_neural_length(output_size,BOLD_range,range_type='cross')
print('Data preprocessed.')

neural_length = train_output.shape[1]

input_place = tf.placeholder(tf.float32, [batch_size, input_size])
output_place = tf.placeholder(tf.float32, [batch_size, output_size])

rnn_neuron = BasicRNNCell(num_hidden)
rnn_neuron = rnn_cell.MultiRNNCell([rnn_neuron] * num_layer)

initial_state=rnn_neuron.zero_state(batch_size=batch_size, dtype=tf.float32)

val,final_state=rnn_neuron(input_place, initial_state)

weight = tf.Variable(tf.truncated_normal([num_hidden, output_size]),name='weight')
bias = tf.Variable(tf.constant(0.1, shape=[output_size]),name='bias')

prediction = tf.nn.sigmoid(tf.matmul(val, weight) + bias,name='prediction')

# Optimizer
residual=tf.square(tf.sub(output_place,prediction))
loss_mse=tf.reduce_mean(residual, name='loss_mse')
optimizer = tf.train.GradientDescentOptimizer(learning_rate,name='optimizer')
minimize = optimizer.minimize(loss_mse,name='minimize')

# Accuracy
mistakes = tf.not_equal(output_place, tf.round(prediction), name='mistakes')
error = tf.reduce_mean(tf.cast(mistakes, tf.float32),name='error')

saver=tf.train.Saver(tf.all_variables())
init_op = tf.initialize_all_variables()

restore_epoch=0
with tf.Session() as sess:

    if need_init:
        sess.run(init_op)
        file_incorrect = open(incorrect_path, 'w')
        print('Initialize successful.')
    else:
        restore_epoch=28300
        file_incorrect=open(incorrect_path,'a')
        saver.restore(sess,plan_path+'/model'+str(restore_epoch)+'.ckpt')
        print('Model restored.')

    no_of_batches = int(len(train_input) / batch_size)
    incorrect = 1
    epoch = restore_epoch

    time_indexs = range(0, len(neural_range)-K+1, K)

    while incorrect > 0.1:
        state=sess.run(rnn_neuron.zero_state(batch_size=batch_size, dtype=tf.float32))
        ptr = 0

        for batch in range(no_of_batches):
            # batch_start_time=datetime.now()
            for time in time_indexs:
                # inp = train_input[ptr:ptr+batch_size,time:time+input_size].reshape(batch_size,input_size)
                # out = fun.get_output_range_by_input_range(range(time, time + input_size), neural_range, train_output,
                #                                          batch_size, ptr, 'wide')
                out = train_output[ptr:ptr+batch_size, time:time+output_size].reshape(batch_size, output_size)
                inp = fun.get_output_range_by_input_range(range(time,time+output_size),BOLD_range,train_input,
                                                            batch_size, ptr, 'plan_D')

                _, state = sess.run([minimize, final_state], {initial_state:state, input_place:inp, output_place : out})
            '''batch_end_time = datetime.now()
            time_consume = (batch_end_time- batch_start_time).microseconds
            print('Epoch - {epoch} , Batch - {batch} , Time consume(per sample) - {time} '
                  .format(epoch=epoch+1, batch=batch+1, time=time_consume/batch_size))
            # print('Time consume - {time}'.format(time = time_consume))'''
            ptr += batch_size

        if (epoch + 1) % 1 == 0:
            incorrect = 0
            loss = 0
            ptr=0
            for time in time_indexs:
                # inp = test_input[:, time:time + input_size].reshape(batch_size, input_size)
                # out = fun.get_output_range_by_input_range(range(time, time + input_size), neural_range, test_output,
                #                                         batch_size, ptr, 'wide')
                out = test_output[ptr:ptr + batch_size, time:time + output_size].reshape(batch_size, output_size)
                inp = fun.get_output_range_by_input_range(range(time, time + output_size), BOLD_range, test_input,
                                                           batch_size, ptr, 'plan_D')
                los, err = sess.run([loss_mse, error], {input_place:inp, output_place: out})
                incorrect += err
                loss += los
            incorrect /= len(time_indexs)
            error_string = 'Epoch - {:2d} error {:2.3f}% loss {:f}'.format(epoch + 1, 100 * incorrect,loss)
            print(error_string)
            file_incorrect=open(incorrect_path,'a')
            file_incorrect.write(error_string+'\n')
            file_incorrect.close()
            if (epoch+1) % 100 == 0:
                save_path = saver.save(sess, plan_path+'/model' + str(epoch + 1) + '.ckpt')
                print('Model saved in file', save_path)

        epoch += 1



