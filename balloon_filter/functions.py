import tensorflow as tf
import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
import h5py
from balloon_filter.balloon_model import *
from balloon_filter.parameters import *
from scipy import interpolate
from scipy.stats import norm


#functions
def get_BOLD_neural_ranges(neural_seq_length=None, BOLD_seq_length=None, HRF_length=None):
    # Pre_check
    if neural_seq_length is None:
        raise TypeError('Please input the length of neural sequence!')
    if BOLD_seq_length is None:
        raise TypeError('Please input the length of BOLD sequence!')
    if HRF_length is None:
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
    if input_range is None:
        raise TypeError('Please input neural range!')
    if mapping is None:
        raise TypeError('Please input the BOLD range of neural time point!')

    if range_type == 'wide':
        output_start = (mapping[input_range.start])[0]
        output_end = (mapping[input_range.stop-1])[1]
    elif range_type == 'cross':
        output_start = (mapping[input_range[len(input_range) - 1]])[0]
        output_end = (mapping[input_range[0]])[1]
    elif range_type=='plan_D':
        output_start = (mapping[input_range[0]])[1]
        output_end = (mapping[input_range[len(input_range)-1]])[1]
    else:
        raise TypeError('Range Type Error!')

    if output_start>output_end and range_type== 'cross':
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


def get_output_length_by_input_length(input_length, mapping=None, range_type='wide'):
    """
    Get the BOLD length by neural length with PLAN A
    :param neural_length: The length of neural sequence
    :param BOLD_range: The map between neural signal at one time point and range of BOLD signal
    :param range_type: The type of relationship between neural and BOLD
    :return: The length BOLD sequence given the length of neural sequence
    """

    if range_type == 'wide':
        output_start = (mapping[0])[0]
        output_end = (mapping[input_length-1])[1]
        return output_end-output_start+1
    elif range_type == 'cross':
        output_start = (mapping[input_length - 1])[0]
        output_end = (mapping[0])[1]
        return output_end-output_start+1
    else:
        raise TypeError('Range Type Error!')

'''

def next_state(previous_state=None, previous_neural=None, batch_size=None, interval=None):
    [previous_neural, previous_state, current_state, previous_BOLD, current_BOLD] \
        = balloon_odeint(sta_var=previous_state, neural=previous_neural, batch_size=batch_size, interval=interval)

    return previous_neural, previous_state, current_state, previous_BOLD, current_BOLD


def cal_previous_neural(predicted_previous_state, predicted_current_state, engine):
    predicted_previous_state = matlab.double(predicted_previous_state.tolist())
    predicted_current_state = matlab.double(predicted_current_state.tolist())
    corrected_previous_neural=engine.calPreviousNeural(predicted_previous_state, predicted_current_state, nargout=1)
    corrected_previous_neural=np.array(corrected_previous_neural)
    return corrected_previous_neural

'''
def cal_jacobian_g(state, engine):
    state = matlab.double(state.tolist())
    jacobian_g = engine.jacobian_g(state, nargout=1)
    jacobian_g = np.array(jacobian_g)
    return jacobian_g
'''


def gen_block_design_data(blo_des_pa=None, exp_pa=None):
    if blo_des_pa is None:
        blo_des_pa = BlockDesignParameters
    if exp_pa is None:
        exp_pa = ExperimentParameters

    batch_size = blo_des_pa.batch_size
    exp_pa.Vg = 10
    interval = 1/exp_pa.Vg
    current_state = StateVariables
    current_state = current_state.as_list(current_state, batch_size=batch_size)
    previous_neural = None
    current_BOLD = gen_BOLD(current_state, batch_size=batch_size)

    neural = list()
    state = list()
    BOLD = list()
    state.append(current_state)
    BOLD.append(current_BOLD)

    for i in range(blo_des_pa.cycles):
        for _ in np.arange(0, blo_des_pa.stimulu_last, interval):
            exp_pa.probability = 1
            [previous_neural, previous_state, current_state, _, current_BOLD] = \
                balloon_odeint(sta_var=current_state, batch_size=batch_size, exp_pa=exp_pa)
            state.append(current_state)
            neural.append(previous_neural)
            current_BOLD = np.random.normal(current_BOLD, 0.0025)
            BOLD.append(current_BOLD)
        for _ in np.arange(0, blo_des_pa.stimulu_last, interval):
            exp_pa.probability = 0
            [previous_neural, previous_state, current_state, _, current_BOLD] = \
                balloon_odeint(sta_var=current_state, batch_size=batch_size, exp_pa=exp_pa)
            state.append(current_state)
            neural.append(previous_neural)
            current_BOLD = np.random.normal(current_BOLD, 0.0025)
            BOLD.append(current_BOLD)

    neu_var = NeuralVariable
    neu_var = neu_var.as_list(neu_var, batch_size)
    neural.append(neu_var)
    state = np.array(state)
    BOLD = np.array(BOLD)
    neural = np.array(neural)

    downsampled_index = np.arange(0, 3001, 30)
    downsampled_BOLD = BOLD[downsampled_index]
    end = 300
    interpolate_index = np.arange(0, end+0.1, 0.1)
    interpolate_BOLD = np.zeros([len(interpolate_index), batch_size, 1])
    for i in range(batch_size):
        interpolate_BOLD[:, i, 0] = np.interp(interpolate_index, downsampled_index/10, np.reshape(downsampled_BOLD[:, i, 0], [101,]))
    '''
    fig, ax = plt.subplots()
    ax.set_xlim(0, 3000)
    ax.set_ylim(0, 0.1)
    lines = [ax.plot([], [])[0] for _ in range(2)]
    tdata1 = np.arange(0, 300.1, 0.1)
    tdata2 = downsampled_index/10
    lines[0].set_data(tdata1, np.reshape(interpolate_BOLD, [3001,]))
    lines[1].set_data(tdata2, np.reshape(downsampled_BOLD, [101,]))
    plt.show()
    '''
    return neural, state, BOLD, interpolate_BOLD


def gen_gaussian_bump_data(batch_size=128):
    inp_pa = InputParameters
    inputs = np.zeros([int(round(inp_pa.time_length / inp_pa.step_size)), batch_size, 1])
    for i in range(batch_size):
        input = inp_pa.gen_input(inp_pa)
        inputs[:, i, :] = input
    neural, state, BOLD = dynamic_balloone_odeint(neural=inputs, batch_size=batch_size)
    return neural, state, BOLD


def move_average(data=None, step_size=5, circle=False):
    axis = 0
    if data is None:
        raise TypeError('Data is Empty!')

    shape = np.shape(data)
    if axis > len(shape):
        raise TypeError('Axis out of Bound!')

    if not isinstance(step_size, int):
        raise TypeError('Step Size must be Integer!')

    if np.mod(step_size, 2) != 1:
        raise TypeError('Step Size must be Odd!')

    move_range = int((step_size-1)/2)
    new_data = np.zeros(shape)
    for i in range(shape[axis]):
        for j in range(i-move_range, i+move_range+1):
            if j >= 0 and j < shape[axis]:
                new_data[i] += data[j]
    new_data /= step_size
    return new_data

def cal_scale(loc=0, confidence_level=0.95, confidence_bound=0.25, confidence_interval=None, edge_type='bilateral'):
    """
    Caculate the standard deviation by mean value, confidence level and confidence interval
    :param loc:
    :param confidence_level:
    :param confidence_bound:
    :param confidence_interval:
    :param edge_type:
    :return: Standard deviation
    """
    if edge_type == 'bilateral':
        confidence_edge = (1 + confidence_level) / 2
    elif edge_type == 'unilateral':
        confidence_edge = confidence_level

    if confidence_interval is not None:
        loc = np.mean(confidence_interval)


    percent_point = norm.ppf(confidence_edge, loc)
    scale = np.abs(confidence_bound - loc) / percent_point
    return scale


def cal_FWTM(loc=0, scale=1, time_threshold=100, time_length=None, step_size=0.1, percent_of_apmlitude=0.1):
    """
    Caculate Full-Width at Tenth Maximum
    :param loc:
    :param scale:
    :param time_threshold: millisecond
    :param time_length: The edge of Width
    :param step_size:
    :param percent_of_apmlitude:
    :return:The Interval of Full-Width at Tenth Maximum
    """
    if time_length is None:
        raise TypeError('Time Length is None!')
    h = 1 / (np.sqrt(2 * np.pi) * scale)
    h *= percent_of_apmlitude
    HWTM = np.sqrt(-2 * np.square(scale) * np.log(h * scale * np.sqrt(2 * np.pi)))
    left = int(round(max(loc - time_threshold, loc - HWTM, 0) / step_size))
    right = int(round(min(loc + time_threshold, loc + HWTM, time_length) / step_size))
    FWTM =  np.arange(left, right)
    return FWTM


def gen_output_state(neural=None, hemodynamic_state=None, i=None, batch_size=128):
    state_size = 4
    neural_size = 1
    output_state = np.zeros([batch_size, state_size])
    """
    if i >= 3:
        output_state[:, 0] = neural[i - 3, :, 0]
        """
    if i >= 2:
        output_state[:, 0] = hemodynamic_state[i - 2, :, 0]
    if i >= 1:
        output_state[:, 1] = hemodynamic_state[i - 1, :, 1]
    output_state[:, 2:4] = hemodynamic_state[i, :, 2:4]
    return output_state


def reset_output_state(output_state=None, neural=None, hemodynamic_state=None, i=None,batch_size=128):
    """
    if i >= 3:
         neural[i - 3, :, 0] = output_state[:, 0]
         """
    if i >= 2:
        hemodynamic_state[i - 2, :, 0] = output_state[:, 0]
    if i >= 1:
        hemodynamic_state[i - 1, :, 1] = output_state[:, 1]
    hemodynamic_state[i, :, 2:4] = output_state[:, 2:4]
    return neural, hemodynamic_state
