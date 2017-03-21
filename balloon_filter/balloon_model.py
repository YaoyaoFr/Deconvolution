import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
import random
from balloon_filter.parameters import *


def outflow(v, alpha):
    return v**(1/alpha)


def oxygen_extraction(f, phy):
    return 1-(1-phy)**(1/f)


def gen_neural(probability):
    n = random.random()
    if n <= probability:
        return 1
    else:
        return 0


def gen_BOLD(state, batch_size=None, phi=None):
    if phi is None:
        bio_pa = BiophysicalParameters
        phi = bio_pa.phi

    if batch_size is None:
        batch_size = 128

    if isinstance(state, list):
        state = np.reshape(np.array(state), [batch_size, 4])

    V0 = 0.04
    k1 = 7 * phi
    k2 = 2
    k3 = 2 * phi - 0.2

    batch_size = np.shape(state)[0]
    if np.shape(state)[1] == 4:
        BOLD = V0 * (k1 * (1 - state[:, 3]) + k2 * (1 - state[:, 3] / state[:, 2]) + k3 * (1 - state[:, 2]))
    elif np.shape(state)[1] ==2:
        BOLD = V0 * (k1 * (1 - state[:, 1]) + k2 * (1 - state[:, 1] / state[:, 0]) + k3 * (1 - state[:, 0]))
    BOLD = np.reshape(BOLD, [batch_size, 1])

    return BOLD


def balloon_differential_equations(sta_var=None, t=None, neural=None, bio_pa=None):
    if sta_var is None:
        sta_var = StateVariables
        sta_var = sta_var.as_list(sta_var)
        print('State Varaible Initial Successful.')
    s = sta_var[0]
    f = sta_var[1]
    v = sta_var[2]
    q = sta_var[3]

    if bio_pa is None:
        bio_pa = BiophysicalParameters
        bio_pa = bio_pa.as_string(bio_pa)
        epsilon = bio_pa[0]
        kappa = bio_pa[1]
        gamma = bio_pa[2]
        tau = bio_pa[3]
        alpha = bio_pa[4]
        phi = bio_pa[5]
        print('Parameters Initial Successful.')
    else:
        epsilon = bio_pa[0]
        kappa = bio_pa[1]
        gamma = bio_pa[2]
        tau = bio_pa[3]
        alpha = bio_pa[4]
        phi = bio_pa[5]

    dx = np.zeros(4)
    dx[0] = epsilon*neural - kappa*s - gamma*(f-1)
    dx[1] = s
    dx[2] = (f-outflow(v, alpha))/tau
    dx[3] = (f*oxygen_extraction(f, phi)/phi-outflow(v, alpha)*q/v)
    return dx


def dynamic_balloone_odeint(initial_state=None, neural=None, bio_pa=None, exp_pa=None, batch_size=128):
    if initial_state is None:
        initial_state = StateVariables
        initial_state = initial_state.as_list(initial_state, batch_size=batch_size)

    if neural is None:
        raise TypeError('Neural is Empty!')

    state = initial_state
    states = list()
    BOLD = list()
    for n in neural:
        _, current_state, next_state, current_BOLD, next_BOLD = balloon_odeint(sta_var=state, neural=n, batch_size=batch_size)
        states.append(current_state)
        BOLD.append(current_BOLD)
        state = next_state

    # Transfrom List to NdArray
    states = np.array(states)
    BOLD = np.array(BOLD)

    return neural, states, BOLD



def balloon_odeint(sta_var=None, neural=None, bio_pa=None, exp_pa=None, batch_size=128, time_span=None):
    if sta_var is None:
        sta_var = StateVariables
        sta_var = sta_var.as_list(sta_var, batch_size=batch_size)

    if bio_pa is None:
        bio_pa = BiophysicalParameters

    if exp_pa is None:
        exp_pa = ExperimentParameters
    Vg = exp_pa.Vg
    interval = 1/Vg
    b = exp_pa.probability

    if neural is None:
        neural = np.zeros([batch_size, 1])
        for i in range(batch_size):
            neural[i] = gen_neural(b)

    if time_span is None:
        time_span = np.arange(0, interval, 0.01)

    next_sta_var = np.zeros([batch_size, 4])
    for i in range(batch_size):
        balloon = odeint(balloon_differential_equations, sta_var[i,:], time_span,
                         args=tuple((neural[i], bio_pa.as_string(bio_pa))))
        next_sta_var[i, :] = balloon[len(balloon)-1]

    BOLD = gen_BOLD(sta_var)
    next_BOLD = gen_BOLD(next_sta_var)

    neural = np.reshape(neural, [batch_size, 1])
    next_BOLD = np.reshape(next_BOLD, [batch_size, 1])

    return neural, sta_var, next_sta_var, BOLD, next_BOLD
