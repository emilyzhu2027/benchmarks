'''
Stanley Bak

should match 'GCAS' scenario from matlab version
'''

import math
import pandas as pd

from numpy import deg2rad
import matplotlib.pyplot as plt

from run_f16_sim import run_f16_sim

from visualize import plot

from gcas_autopilot import GcasAutopilot
from datetime import datetime

# seed the pseudorandom number generator
from random import seed
from random import random

def main_run_gcas():
    'main function'
    # Edited the function to use randomized initial parameters

    ### Initial Conditions ###
    power = 4 # engine power level (0-10)
    seed(datetime.now().timestamp())

    # Default alpha & beta
    alpha = random() * deg2rad(15) # ! Trim Angle of Attack (rad)
    beta = random() * deg2rad(10)              # ! Side slip angle (rad)

    # Initial Attitude
    alt = random() * 30000 + 5000      # ! altitude (ft)
    vt = int(random() * 1500) + 500    # ! initial velocity (ft/sec)
    negphi = random()
    if (negphi >= 0.5):
        phi = (random() * deg2rad(5))          # ! Roll angle from wings level (rad)
    else:
        phi = -(random() * deg2rad(5))  

    negtheta = random()
    if (negtheta >= 0.5):
        theta = -(random() * deg2rad(45))       # ! Pitch angle from nose level (rad)
    else:
        theta = random() * deg2rad(45)

    negpsi = random()
    if (negpsi >= 0.5):
        psi = random() * deg2rad(30)  # Yaw angle from North (rad)
    else:
        psi = -(random() * deg2rad(30))

    # Build Initial Condition Vectors
    P = random() * deg2rad(10)
    Q = random() * 0.1
    R = random() * 0.1
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, P, Q, R, 0, 0, alt, power]
    print(init)
    tmax = 5 # simulation time

    ap = GcasAutopilot(init_mode='roll', stdout=True, gain_str='old')

    step = 1/100
    res = run_f16_sim(init, tmax, ap, step=step, extended_states=True)
    print(res['states'])
    print(res['modes'])

    print(f"Simulation Completed in {round(res['runtime'], 3)} seconds")



    times, ys = plot.plot_single(res, 'alt', title='Altitude (ft)')
    df = pd.DataFrame({'alt': ys}, index = times)


    filename = 'alt.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    plot.plot_attitude(res)
    filename = 'attitude.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # plot inner loop controls + references
    plot.plot_inner_loop(res)
    filename = 'inner_loop.png'
    plt.savefig(filename)
    print(f"Made {filename}")

    # plot outer loop controls + references
    plot.plot_outer_loop(res)
    filename = 'outer_loop.png'
    plt.savefig(filename)
    print(f"Made {filename}")
  

    return df

if __name__ == '__main__':
    df = main_run_gcas()
    print(df)
