import numpy as np
from balloon_filter import functions as fun
from scipy.stats import norm

class BiophysicalParameters:
    epsilon = 0.54
    kappa = 0.65
    gamma = 0.38
    tau = 0.98
    alpha = 0.34
    phi = 0.32

    def __init__(self, bio_pa):
        if isinstance(bio_pa, BiophysicalParameters):
            self.epsilon = bio_pa.epsilon
            self.kappa = bio_pa.kappa
            self.gamma = bio_pa.gamma
            self.tau = bio_pa.tau
            self.alpha = bio_pa.alpha
            self.phi = bio_pa.phi

    def as_string(self):
        bio_pa = [self.epsilon, self.kappa, self.gamma, self.tau, self.alpha, self.phi]
        return bio_pa


class StateVariables:
    signal = 0
    flow = 1
    volume = 1
    content = 1

    def __init__(self, sta_var):
        if isinstance(sta_var, StateVariables):
            self.signal = sta_var.signal
            self.flow = sta_var.flow
            self.volume = sta_var.volume
            self.content = sta_var.content

    def as_list(self, batch_size):
        sta_var = np.zeros([batch_size, 4])
        for i in range(batch_size):
            sta_var[i, :] = [self.signal, self.flow, self.volume, self.content]
        if batch_size == 1:
            sta_var = np.reshape(sta_var, [batch_size, 4])
        return sta_var

class NeuralVariable():
    neural = 0

    def __init__(self, neu_var):
        if isinstance(neu_var, NeuralVariable):
            self.neural = neu_var.neural

    def as_list(self, batch_size):
        neu_var = np.zeros([batch_size, 1])
        for i in range(batch_size):
            neu_var[i, :] = [self.neural]
        if batch_size == 1:
            neu_var = np.reshape(neu_var, [batch_size, 1])
        return neu_var


class ExperimentParameters:
    probability = 0.5
    Vg = 2

    def __init__(self, exp_pa):
        self.probability = exp_pa.probability
        self.Vg = exp_pa.Vg


class BlockDesignParameters:
    cycles = 5
    step_size = 0.1
    stimulu_last = 30
    stimulu_interval = 30
    downsampling_rate = 1/3
    batch_size = 16


    def __init__(self, blo_des_pa):
        self.cycles = blo_des_pa.cycles
        self.step_size = blo_des_pa.step_size
        self.stimulu_last = blo_des_pa.stimulu_last
        self.stimulu_interval = blo_des_pa.stimulu_interval
        self.downsampling_rate = blo_des_pa.downsampling_rate
        self.batch_size = blo_des_pa.batch_size


class InputParameters:
    # Time Length (s)
    time_length = 60
    # Step Size (s)
    step_size = 0.1
    # The Number of Stimulus
    stimulu_num = 4
    # Full width of percent Maximum
    percent_of_amplitude = 1


    def __init__(self, inp_pa):
        if isinstance(inp_pa):
            self.time_length = inp_pa.time_length
            self.stimulu_num = inp_pa.stimulu_num


    def gen_input(self, amplitude_mean = 0.7, confidence_level=0.99):
        # Set The Amplitude of Stimulu
        sequence_length = round(self.time_length / self.step_size)
        input = np.zeros([sequence_length, 1])
        amplitude_sigma = fun.cal_scale(loc=amplitude_mean, confidence_level=confidence_level, confidence_bound=1.1)
        amplitudes = [np.random.normal(loc=amplitude_mean, scale=amplitude_sigma) for _ in range(self.stimulu_num)]
        input_sigmas = [1 / (np.sqrt(2 * np.pi) * amplitude) for amplitude in amplitudes]

        # Set The Center of Stimulu
        input_centers = np.zeros(shape=[self.stimulu_num, ])
        interval_length = np.round(self.time_length / self.stimulu_num)
        confidence_intervals = [[i * interval_length, (i + 1) * interval_length] for i in range(self.stimulu_num)]
        for i in range(self.stimulu_num):
            center_sigma = fun.cal_scale(confidence_level=0.99, confidence_interval=confidence_intervals[i])
            input_centers[i] = np.random.normal(loc=np.mean(confidence_intervals[i]), scale=center_sigma)

        # Set FWTM
        for loc, scale in zip(input_centers, input_sigmas):
            if self.percent_of_amplitude < 1:
                FWTM = fun.cal_FWTM(loc=loc, scale=scale, time_length=self.time_length, percent_of_apmlitude=self.percent_of_amplitude)
                for i in FWTM:
                    input[i] += norm.pdf(i * self.step_size, loc=loc, scale=scale)
            else:
                for i in range(0, sequence_length):
                    input[i] += norm.pdf(i * self.step_size, loc=loc, scale=scale)


        return input