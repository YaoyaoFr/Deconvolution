import tensorflow as tf
import numpy as np
import matlab.engine
import h5py

#functions
def get_BOLD_neural_ranges(neural_seq_length=None, BOLD_seq_length=None, HRF_length=None):
    # Pre_check
    if(neural_seq_length==None):
        raise TypeError('Please input the length of neural sequence!')
    if(BOLD_seq_length==None):
        raise TypeError('Please input the length of BOLD sequence!')
    if(HRF_length==None):
        raise TypeError('Please input the length of Hemodynamic Response Function!')

    #  Generate neural range temporaly
    neural_range_of_BOLD_time_point_temp=[[row-HRF_length+1,row] for row in range(BOLD_seq_length)]

    # Exclude unreasonable neural range
    neural_range_of_BOLD_time_point=dict()
    for i in range(BOLD_seq_length):
        if (neural_range_of_BOLD_time_point_temp[i])[0]>=0 and (neural_range_of_BOLD_time_point_temp[i])[1]<neural_seq_length:
            neural_range_of_BOLD_time_point[i]=neural_range_of_BOLD_time_point_temp[i]

    # Generate BOLD range temporaly
    BOLD_range_of_neural_time_point_temp=[[row,row+HRF_length-1] for row in range(neural_seq_length)]

    # Exclude unrasonable BOLD range
    BOLD_range_of_neural_time_point=dict()
    for i in range(neural_seq_length):
        if((BOLD_range_of_neural_time_point_temp[i])[1]<=neural_seq_length):
            BOLD_range_of_neural_time_point[i]=BOLD_range_of_neural_time_point_temp[i]

    return BOLD_range_of_neural_time_point,neural_range_of_BOLD_time_point

def get_output_range_by_input_range(input_range=None, mapping=None, data=None, batch_size=None, ptr=None, range_type='wide'):
    if (input_range == None):
        raise TypeError('Please input neural range!')
    if (mapping == None):
        raise TypeError('Please input the BOLD range of neural time point!')

    if range_type=='wide':
        output_start=(mapping[input_range.start])[0]
        output_end=(mapping[input_range.stop-1])[1]
    elif range_type=='cross':
        output_start = (mapping[input_range[len(input_range) - 1]])[0]
        output_end = (mapping[input_range[0]])[1]
    elif range_type=='plan_D':
        output_start = (mapping[input_range[0]])[1]
        output_end = (mapping[input_range[len(input_range)-1]])[1]
    else:
        raise TypeError('Range Type Error!')

    if((output_start>output_end) and range_type=='cross'):
        raise TypeError('THe intput seq is too long that the length of output is negative!')


    return data[ptr:ptr+batch_size,output_start:output_end+1]

'''
def get_train_test_input_output(train_BOLD=None,train_neural=None,test_BOLD=None,
                                test_neural=None,neural_range=None,BOLD_range=None):

    train_input=train_BOLD[:,BOLD_range]
    train_output=train_neural[:,neural_range]
    test_input=(test_BOLD[:,BOLD_range])
    test_output=(test_neural[:,neural_range])

    return train_input,train_output,test_input,test_output
    '''

def get_output_length_by_input_length(input_length, mapping=None, range_type='wide'):
    '''
    Get the BOLD length by neural length with PLAN A
    :param neural_length: The length of neural sequence
    :param BOLD_range: The map between neural signal at one time point and range of BOLD signal
    :param range_type: The type of relationship between neural and BOLD
    :return: The length BOLD sequence given the length of neural sequence
    '''

    if(range_type=='wide'):
        output_start = (mapping[0])[0]
        output_end = (mapping[input_length-1])[1]
        return output_end-output_start+1
    elif (range_type == 'cross'):
        output_start = (mapping[input_length - 1])[0]
        output_end = (mapping[0])[1]
        return output_end-output_start+1
    else:
        raise TypeError('Range Type Error!')

def weight_variable(shape,name='weight'):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name='bias'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)

def conv1d(x, W,stride=1, padding='SAME'):
    return tf.nn.conv1d(x, W, stride=stride, padding=padding)

def max_pool_1x2(x,padding='SAME'):
    with tf.Session() as sess:
        shape = sess.run(tf.shape(x))
        x = tf.reshape(x,(shape[0],1,shape[1],shape[2]))
        pool = tf.nn.max_pool(x, ksize=[1,1,2,1], strides=[1,1,2,1], padding=padding)
        shape = sess.run(tf.shape(pool))
        pool = tf.reshape(pool,(shape[0],shape[2],shape[3]))

        return pool, [shape[0],shape[2],shape[3]]

def next_state(previous_state=None, previous_neural=None, engine=None):
    if(previous_state!=None):
        previous_state=matlab.double(previous_state.tolist())
    if(previous_neural!=None):
        previous_neural=matlab.double(previous_neural.tolist())

    if previous_neural!=None and previous_state!=None:
        [previous_neural,previous_state,current_state,current_BOLD]=engine.nextState(previous_state, previous_neural, nargout=4)
    elif previous_state!=None:
        [previous_neural, previous_state, current_state, current_BOLD] = engine.nextState(previous_state, nargout=4)
    else:
        [previous_neural, previous_state, current_state, current_BOLD] = engine.nextState(nargout=4)

    previous_neural = np.array(previous_neural)
    previous_state = np.array(previous_state)
    current_state = np.array(current_state)
    current_BOLD = np.array(current_BOLD)
    return previous_neural, previous_state, current_state, current_BOLD


def cal_previous_neural(predicted_previous_state, predicted_current_state, engine):
    predicted_previous_state = matlab.double(predicted_previous_state.tolist())
    predicted_current_state = matlab.double(predicted_current_state.tolist())
    corrected_previous_neural=engine.calPreviousNeural(predicted_previous_state, predicted_current_state, nargout=1)
    corrected_previous_neural=np.array(corrected_previous_neural)
    return corrected_previous_neural


def cal_jacobian_g(state, engine):
    state = matlab.double(state.tolist())
    jacobian_g = engine.jacobian_g(state, nargout=1)
    jacobian_g = np.array(jacobian_g)
    return jacobian_g
