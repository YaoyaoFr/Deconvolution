import sys
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import BasicLSTMCell
sys.path.append('F:\\Program\\Python')
import data.preprocess_data as pr
import data.functions as fun



K=641
num_hidden = 5
#step_size : the input length of every time step, it must be the divisor of 640(K-1)
step_size=1
output_size=step_size
input_size=step_size
learning_rate=0.1
need_init=True
batch_size = 200
incorrect_path='F:\\Program\\Python\\data\\save\\incorrect.txt'

train_input,train_output,test_input,test_output,BOLD_range=pr.preprocess_data(batch_size=batch_size)
print('Data preprocessed.')

neural_length=train_output.shape[1]
BOLD_length=train_input.shape[1]

num_input_init=K-1
step_num_init= int((K - 1) / step_size)

input_init=tf.placeholder(tf.float32, [batch_size, step_num_init, step_size])
#input_init=tf.tranpose(1,num_step_init,input_init)

lstm = BasicLSTMCell(num_hidden,state_is_tuple=True)
initial_state=state=lstm.zero_state(batch_size,tf.float32)

outputs, states = dynamic_rnn(lstm, input_init, initial_state=initial_state)

final_initial_state=states


input = tf.placeholder(tf.float32, [batch_size, input_size])
output=tf.placeholder(tf.float32, [batch_size, output_size])
val,state=lstm(input, final_initial_state)

weight = tf.Variable(tf.truncated_normal([num_hidden, output_size]),name='weight')
bias = tf.Variable(tf.constant(0.1, shape=[output_size]),name='bias')

prediction = tf.nn.sigmoid(tf.matmul(val, weight) + bias,name='prediction')

#cross_entropy = -tf.reduce_sum(target * tf.log(prediction))
loss_mse=tf.reduce_mean(tf.square(tf.sub(output, prediction)), name='loss_mse')

optimizer = tf.train.GradientDescentOptimizer(learning_rate,name='optimizer')
minimize = optimizer.minimize(loss_mse,name='minimize')

mistakes = tf.not_equal(output, tf.round(prediction), name='mistakes')
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
        restore_epoch=7000
        saver.restore(sess,'save/model'+str(restore_epoch)+'.ckpt')
        print('Model restored.')

    no_of_batches = int(len(train_input)/batch_size)
    incorrect=1
    epoch=restore_epoch
    while(incorrect>0.1):
        ptr = 0
        '''
        for j in range(no_of_batches):
            inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
            ptr+=batch_size
            sess.run(minimize, {input: inp, target: out})
        '''
        for batch in range(no_of_batches):
            #init
            inp=train_input[batch*batch_size:(batch+1)*batch_size,0:num_input_init]
            inp=inp.reshape(batch_size, step_num_init, step_size)
            final_state=sess.run(final_initial_state,{input_init:inp})

            #time step
            for time in range(num_input_init+1,BOLD_length):
                inp=train_input[batch*batch_size:(batch+1)*batch_size,time].reshape(batch_size,step_size)
                out=train_output[batch*batch_size:(batch+1)*batch_size,time-num_input_init].reshape(batch_size,step_size)
                state,error=sess.run([state,error],{state:state,final_initial_state:final_state,input:inp,output:out})
            print('ok')



        if((epoch+1) % 50==0):
            incorrect = sess.run(error, {input: test_input, output: test_output})
            error_string='Epoch - {:2d} error {:2.3f}%'.format(epoch + 1, 100 * incorrect)
            print(error_string)
            file_incorrect.write(error_string + '\r\n')

        if((epoch+1) % 1000==0):
            save_path = saver.save(sess, 'save/model' + str(epoch + 1) + '.ckpt')
            print('Model saved in file', save_path)
        epoch+=1


    sess.close()