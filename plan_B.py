import sys

import tensorflow as tf

sys.path.append('F:\\Program\\Python')
import data.preprocess_data as pr
import math
import data.functions as fun
from tensorflow.python.ops.rnn_cell import BasicLSTMCell

K=641
num_hidden = 5
#step_size : the input length of every time step, it must be the divisor of 640(K-1)
step_size=1
output_size=step_size
learning_rate=0.1
need_init=True
batch_size = 200
incorrect_path='save_plan_B/incorrect.txt'


train_input,train_output,test_input,test_output,BOLD_range=pr.preprocess_data(batch_size=batch_size)
test_input=test_input[0:batch_size,:]
test_output=test_output[0:batch_size,:]
input_size=1
print('Data preprocessed.')

neural_length=train_output.shape[1]
step_num=math.floor(neural_length/step_size)

input=tf.placeholder(tf.float32,[batch_size,input_size])
output=tf.placeholder(tf.float32,[batch_size,output_size])

lstm = BasicLSTMCell(num_hidden, state_is_tuple=False)

initial_state=lstm.zero_state(batch_size=batch_size,dtype=tf.float32)

val,final_state=lstm(input, initial_state)

weight = tf.Variable(tf.truncated_normal([num_hidden, output_size]),name='weight')
bias = tf.Variable(tf.constant(0.1, shape=[output_size]),name='bias')

prediction = tf.nn.sigmoid(tf.matmul(val, weight) + bias,name='prediction')

#Optimizer
loss_mse=tf.reduce_mean(tf.square(tf.sub(output, prediction)), name='loss_mse')
optimizer = tf.train.GradientDescentOptimizer(learning_rate,name='optimizer')
minimize = optimizer.minimize(loss_mse,name='minimize')

#Accuracy
mistakes = tf.not_equal(output,tf.round(prediction),name='mistakes')
error = tf.reduce_mean(tf.cast(mistakes, tf.float32),name='error')

saver=tf.train.Saver(tf.all_variables())
init_op = tf.initialize_all_variables()

need_init=True
restore_epoch=0
file_incorrect=open(incorrect_path,'w')
with tf.Session() as sess:

    if(need_init):
        sess.run(init_op)
        print('Initialize successful.')
    else:
        restore_epoch=6
        saver.restore(sess,'save_plan_B/model'+str(restore_epoch)+'.ckpt')
        print('Model restored.')

    no_of_batches = int(len(train_input) / batch_size)
    incorrect = 1
    epoch = restore_epoch
    while(incorrect>0.1):
        state=sess.run(lstm.zero_state(batch_size=batch_size,dtype=tf.float32))
        ptr = 0
        print('Epoch : ',epoch+1)
        for batch in range(no_of_batches):
            print('\tBatch : ',batch+1)
            for time in range(neural_length-1,0,-1):
                out=train_output[ptr:ptr+batch_size,time]
                out=out.reshape(batch_size,output_size)
                inp=train_input[ptr:ptr+batch_size,time+K-1].reshape(batch_size,output_size)
                _,state=sess.run([minimize,final_state],{initial_state:state,input:inp,output:out})
            ptr+=batch_size

        if((epoch+1) % 1==0):
            # Save Model
            save_path = saver.save(sess, 'save_plan_B/model' + str(epoch + 1) + '.ckpt')
            print('Model saved in file', save_path)

            # Caculate the Error Rate
            incorrect = 0
            for time in range(neural_length - 1, 0, -1):
                out = test_output[:, time]
                out = out.reshape(batch_size, output_size)
                inp=test_input[:,time+K-1].reshape(batch_size,output_size)

                incorrect+= sess.run(error, {input: inp, output: out})
            incorrect/=neural_length
            error_string = 'Epoch - {:2d} error {:2.3f}%'.format(epoch + 1, 100 * incorrect)
            print(error_string)
            file_incorrect.write(error_string + '\r\n')

        epoch += 1



