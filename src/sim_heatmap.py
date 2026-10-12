import numpy as np
import sys

from lib_spike_control import *

from joblib import Parallel, delayed

# Define grid and (zero) target
grid_max = 200
n_grid = 15001

grid = np.linspace(0., grid_max, n_grid, endpoint=True)
target = np.zeros(len(grid))

steps = len(grid)
h = grid[-1]/(steps - 1)

# Define spiking control system
K=1
N=2

x0 = 50.5 # don't choose a multiple of eps0
x_max = 1e6

t_ref = 0
eps0 = 1

# Define spikes
B = np.array([-eps0, eps0])

leak = 3
delay = 0.
leak_vec = leak * np.ones(N)

sc_dict = {
    'x0': x0,
    'G': lambda x,y,z: (np.array([np.max([x-y,0]), np.max([y-x,0])]), 0),
    't_ref': t_ref,
    'input_matrix': B,
    'delta': delay,
    'leak': leak_vec,
    'x_max': x_max,
    }

# Define function to compute growth rate for given a and b
def compute_growth_log_rel(a, b, sc_dict):
    try:
        this_sc_dict = sc_dict.copy()
        # Set a in f_x
        this_sc_dict['f_x'] = lambda x: a * x

        # Set b in theta
        if b > 0:
            this_sc_dict['theta'] = 1/b * np.ones(N)
        else:
            this_sc_dict['theta'] = np.inf * np.ones(N)

        # Simulate the system
        this_scs = spiking_control_system(this_sc_dict)
        z, s, x = this_scs.control_exec(grid, target)

        i_end = len(x)
        grid_x = grid[0:i_end]

        # Compute growth rate as log ratio of final to initial state, divided by time
        val = np.log(np.abs(x[-1]/x[0])) / (grid_x[-1] - grid_x[0])
        if grid_x[-1] - grid_x[0] == 0:
             raise ValueError("No steps")
        elif np.isnan(val) or np.isinf(val):
            raise ValueError("invalid growth value")
        return val
    except Exception as e:
            print(f"Error at (a={a}, b={b}): {e}")
            # Return NaN so everything has consistent type
            return np.nan

# Compute cost matrix

# Set values for a and b
a_values = np.linspace(0.0, 5, 50)
b_values = np.linspace(0.0, 5, 50)

num_cores = -1  
growth_values = Parallel(n_jobs=num_cores)(
    delayed(compute_growth_log_rel)(a, b, sc_dict)
    for a in a_values
    for b in b_values
)
growth_matrix = np.array(growth_values).reshape(len(a_values), len(b_values))

# Save cost matrix for later use
np.save(f"data/growth_matrix_na_{len(a_values)}_nb_{len(b_values)}_x0_{x0}_leak_{leak}.npy", growth_matrix)