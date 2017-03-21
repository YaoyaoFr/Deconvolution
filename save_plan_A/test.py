import numpy as np
import matlab.engine
import tensorflow as tf

batch_size=128

x_place = tf.placeholder(tf.float32, [4, 1])
exp_x = tf.log(x_place)


with tf.Session() as sess:
    x = np.random.random([4, 1])

    z_out = sess.run([exp_x],{
        x_place: x,
    })

    pass
