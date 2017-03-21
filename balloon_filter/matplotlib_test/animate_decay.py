"""
=====
Decay
=====

This example showcases a sinusoidal decay animation.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def data_gen(t=0):
    cnt = 0
    while cnt < 1000:
        cnt += 1
        t += 0.1
        yield t, np.sin(2*np.pi*t) * np.exp(-t/10.), np.cos(2*np.pi*t) * np.exp(-t/10.)


def init():
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, 10)
    del tdata[:]
    del xdata[:]
    del ydata[:]
    lines[0].set_data(tdata, xdata)
    lines[1].set_data(tdata, ydata)
    return lines,

fig, ax = plt.subplots()
lines = [ax.plot([], [], lw=2)[0], ax.plot([], [], lw=2)[0]]
ax.grid()
tdata, xdata, ydata = [], [], []


def run(data):
    # update the data
    t, x, y = data
    tdata.append(t)
    xdata.append(x)
    ydata.append(y)

    max_len = 20
    if len(tdata) >= max_len:
        del tdata[0]
        del xdata[0]
        del ydata[0]
        ax.set_xlim(t-max_len*0.1, t)
        ax.figure.canvas.draw()
    lines[0].set_data(tdata, xdata)
    lines[1].set_data(tdata, ydata)

    return lines,

ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=10,
                              repeat=False, init_func=init)
plt.show()