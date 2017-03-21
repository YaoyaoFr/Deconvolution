import tensorflow as tf
import functions as fun
import tensorflow.tensorflow.python.ops.rnn_cell
import numpy as np
import matlab.engine
plan_path = 'F:/plan_A'


K = 641
input_size=1
state_size = 4
hidden_size=5

learning_rate = 0.01
need_init = True
batch_size = 128
incorrect_path=plan_path+'\\incorrect.txt'
engine = matlab.engine.start_matlab()


current_BOLD_place = tf.placeholder(tf.float32, [batch_size, input_size])
current_state_place = tf.placeholder(tf.float32, [batch_size, state_size])
predicted_current_BOLD_place=tf.placeholder(tf.float32, [batch_size, input_size])
predicted_current_state_place = tf.placeholder(tf.float32, [batch_size, state_size])
predicted_previous_state_place = tf.placeholder(tf.float32, [batch_size, state_size])

'''
W_1 = tf.Variable(tf.truncated_normal([state_size, hidden_size], name='weight_1'))
bias_1 = tf.Variable(tf.truncated_normal([hidden_size]))

W_2 = tf.Variable(tf.truncated_normal([hidden_size, state_size]))
bias_2 = tf.Variable(tf.truncated_normal([state_size]))

input = tf.sub(current_BOLD_place, predicted_current_BOLD_place)

H_1 = tf.sigmoid(tf.matmul(predicted_previous_state_place, W_1)+bias_1)
H_2 = tf.sigmoid(tf.matmul(H_1, W_2)+bias_2)
o = tf.reshape(tf.batch_matmul(tf.reshape(H_2,[batch_size, 4, 1]), tf.reshape(input,[batch_size, input_size, 1])),[batch_size, 4]) + predicted_previous_state_place
'''
input = tf.sub(current_BOLD_place, predicted_current_BOLD_place)
W = tf.Variable(tf.truncated_normal([3, 2]))
b = tf.Variable(tf.truncated_normal([2]))


predicted_previous_f_v_q = tf.slice(predicted_previous_state_place,[0,1],[batch_size,3])
W_o = tf.sigmoid(tf.matmul(predicted_previous_f_v_q, W)+b)
b_o = tf.Variable(tf.truncated_normal([2]))

predicted_current_v_q_place = tf.slice(predicted_previous_state_place,[0,2],[batch_size,2])
o = tf.reshape(tf.batch_matmul(tf.reshape(W_o,[batch_size, 2, 1]), tf.reshape(input,[batch_size, input_size, 1])),
               [batch_size, 2]) + predicted_current_v_q_place + b_o

# Applying Different Activation Function

# Neural
o_act = tf.sigmoid(o)

# Optimizer
current_v_q_place = tf.slice(current_state_place,[0, 2],[batch_size, 2])

loss_v_q = tf.reduce_mean(tf.square(tf.sub(current_v_q_place, o_act)))
loss_v = tf.reduce_mean(tf.square(tf.sub(tf.slice(current_v_q_place, [0, 0], [batch_size, 1]),
                                            tf.slice(o_act, [0, 0], [batch_size, 1]))))
loss_q = tf.reduce_mean(tf.square(tf.sub(tf.slice(current_v_q_place, [0, 1], [batch_size, 1]),
                                            tf.slice(o_act, [0, 1], [batch_size, 1]))))
optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='optimizer')
minimize = optimizer.minimize(loss_v_q, name='minimize')

# Accuracy
corrected_previous_neural_place = tf.placeholder(tf.float32, [batch_size,1])
previous_neural_place = tf.placeholder(tf.float32, [batch_size,1])

loss_neural = tf.reduce_mean(tf.square(tf.sub(corrected_previous_neural_place,previous_neural_place)))
true = tf.equal(previous_neural_place, tf.round(corrected_previous_neural_place), name='true')
accu = tf.reduce_mean(tf.cast(true, tf.float32), name='accu')

# Saver
saver = tf.train.Saver(tf.all_variables())
init_op = tf.initialize_all_variables()

restore_epoch=0
with tf.Session() as sess:

    if need_init:
        sess.run(init_op)
        file_incorrect = open(incorrect_path,'w')
        # Initial Predicted Value
        [predicted_previous_neural, predicted_previous_state, predicted_current_state, predicted_current_BOLD] = fun.next_state(engine=engine)
        # Initial Train Value
        [previous_neural, previous_state, current_state, current_BOLD]=fun.next_state(engine=engine)

        print('Initialize successful.')
    else:
        restore_epoch = 5000
        file_incorrect = open(incorrect_path,'a')

        # Initial Predicted Value
        [predicted_previous_neural, predicted_previous_state, predicted_current_state, predicted_current_BOLD] = fun.next_state(engine=engine)
        # Initial Train Value
        [previous_neural, previous_state, current_state, current_BOLD]=fun.next_state(engine=engine)

        saver.restore(sess,plan_path+'/model'+str(restore_epoch)+'.ckpt')
        print('Model restored.')

    epoch = restore_epoch
    while True:
        # Optimize
        output_current_state, los_s, los_v, los_q, _ = sess.run([o_act, loss_v_q, loss_v, loss_q, minimize], {
                                                 current_BOLD_place: current_BOLD,
                                                 current_state_place: current_state,
                                                 predicted_current_BOLD_place: predicted_current_BOLD,
                                                 predicted_current_state_place: predicted_current_state,
                                                 predicted_previous_state_place: predicted_previous_state,
                                             })


        error_string = 'Epoch - {:2d}  loss_state {:f} loss_v {:f} loss_q {:f}'\
            .format(epoch + 1, los_s, los_v, los_q)
        print(error_string)

        # Next Time
        [predicted_previous_neural, predicted_previous_state, predicted_current_state,
         predicted_current_BOLD] = fun.next_state(predicted_current_state, engine=engine)

        # Initial Train Value
        [previous_neural, previous_state, current_state, current_BOLD] = fun.next_state(current_state,engine=engine)

        file_incorrect=open(incorrect_path, 'a')
        file_incorrect.write(error_string+'\n')
        file_incorrect.close()

        if (epoch+1) % 500 == 0:
            save_path = plan_path+'/model'+str(epoch+1)+'.ckpt'
            save_path = saver.save(sess, save_path)
            print('Model saved in file', save_path)

        epoch += 1



