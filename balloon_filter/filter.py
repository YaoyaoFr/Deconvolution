import numpy as np
from balloon_model import *
from balloon_filter.parameters import *


def state_filter(state, neural, interval=None, bio_pa=None, index=None):
    if state is None:
        raise TypeError('State is Empty!')

    if isinstance(state, list):
        state = np.array(state)

    if neural is None:
        neural = np.zeros(np.shape(state)[0], 1)

    if isinstance(neural, list):
        neural = np.array(neural)

    if interval is None:
        exp_pa = ExperimentParameters
        interval = 1/exp_pa.Vg

    if bio_pa is None:
        bio_pa = BiophysicalParameters
    epsilon = bio_pa.epsilon
    kappa = bio_pa.kappa
    gamma = bio_pa.gamma
    tau = bio_pa.tau
    alpha = bio_pa.alpha
    phi = bio_pa.phi

    # f
    # t = t
    if index is None:
        index = len(state) - 1
    if index > 0:
        state[index - 1, :, 1] = (state[index, :, 2] - state[index - 1, :, 2]) * tau / interval + state[index - 1, :, 2] ** (1/alpha)

    # s
    # t = t-1
    index -= 1
    if index > 0:
        state[index - 1, :, 0] = (state[index, :, 1] - state[index - 1, :, 1]) / interval

    # neural
    # t = t-2
    index -= 1
    if index > 0:
        neural[index - 1, :, 0] = ((state[index, :, 0] - state[index - 1, :, 0]) /
                                interval + kappa * state[index - 1, :, 0] +
                                gamma * (state[index - 1, :, 1] - 1)) / epsilon

    return state, neural

