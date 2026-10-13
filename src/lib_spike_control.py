import numpy as np
import json

# Define spiking control system class

class spiking_control_system:
    
    #Event-based impulsive controller using the dynamical systems approach:
    #x(t) = \int dx/dt(t) dt, where                      (1)
    #    dx/dt(t) = f_x(x(t)) + B*s(t),                  (2)
    #    s_i(t) = \delta_{z_i=\theta_i},                 (3)
    #    dz/dt = - leak z(t) + G(x(t), x_star(t), G_memory) - \Theta s(t)  (4)
    #x_star is the target, and G is a function that depends on the current state
    #x, the target x_star, and a memory term G_memory.

    def __init__(self, params=None):
        default_params = {
                'K': 1,                               # dimension of state x
                'N': 2,                               # dimension of neurons s
                'x0': 0,                              # initial state x(0)
                'f_x': lambda x: -x,                  # see (2) in description
                'G': lambda x,y: 0,                   # see (4) in description
                't_ref': 0.,                          # refractory period
                'input_matrix': np.array([1,-1]),     # prefactor in front of u, see (2) in description
                'theta': 1.,                          # threshold vector in (3)
                'delta': 0.,                          # delay of feedback, see (4)
                'leak': 0.,                           # leak in integration, see l in (4)
                'lmbda': 1.,                          # relative weight between target and control cost
                'x_max': np.inf,                      # maximum norm of x before simulation is stopped
                }
        # merge default params with user defined params
        self.params = {**default_params, **(params or {})}
        # ensure that f_x is callable
        if not callable(self.params['f_x']):
            raise ValueError("Parameter 'f_user' must be a callable function")
        if not callable(self.params['G']):
            raise ValueError("Parameter 'G' must be a callable function")

    def set_params(self, params):
        self.params.update(params)
        if not callable(self.params['f_x']):
            raise ValueError("Parameter 'f_user' must be a callable function")
        if not callable(self.params['G']):
            raise ValueError("Parameter 'G' must be a callable function")

    def get_params(self):
        return self.params

    def print_params(self, indent=4):
        
        #function to print a dictionary in an easily readable way for debugging purposes
        
        print(json.dumps(self.params, indent=indent, sort_keys=True))

    def _simulate(self, grid, x_star):
        steps = len(grid)
        h = grid[-1]/(steps - 1) # step size for simulation

        # retrieve parameters
        f_x = self.params['f_x']
        G = self.params['G']
        t_ref = self._t_ref_vec
        theta = self.params['theta']
        delta_steps = int(np.floor(self.params['delta']/h))     # delay in number of simulation steps
        B = self.params['input_matrix']
        leak = self.params['leak']
        x0 = self.params['x0']
        x_max = self.params['x_max']

        G_memory = 0 * x0   # possible memory for function G

        # Check dimensions
        if B.ndim == 1:
            K, N = 1, len(B)
        else:
            K, N = B.shape
        
        # Dynamic vectorization
        if self.size > 1:
            shape_latent = (steps+1,N)
            shape = (steps+1,K)
        else:
            shape_latent = (steps+1,N)
            shape = (steps+1,)
            
        z = np.zeros(shape_latent) # init latent
        deriv_x = np.zeros(shape)
        s = np.zeros(shape_latent) # init control signal
        x = np.zeros(shape) # init state
        x[0,] = x0

        i_end = -1
        if np.all(t_ref == 0):
            for i,g in enumerate(grid):
                s[i,], z[i,] = np.divmod(z[i,], theta) # calculate positive (multi)spikes and reset of membrane potential
                
                z_update, G_memory = G(x[i,], x_star[i,], G_memory)
                z[i+1,] = z[i,] - h*leak*z[i,] + h*z_update
    
                # implementation of delay
                if i - delta_steps < 0:
                    s_delayed = 0.
                else:
                    s_delayed = s[i-delta_steps,]/h

                deriv_x[i,] = f_x(x[i,]) + B @ s_delayed
                x[i+1,] = x[i,] + deriv_x[i,]*h # update state

                if np.linalg.norm(x[i+1,]) > x_max:
                    i_end = i
                    break

        elif np.any(np.logical_and(t_ref < h, t_ref >= 0)):
            print("Work in progress, t_ref too small")
        elif np.all(t_ref >= h):
            n_ref = np.ceil(t_ref/h)
            n_counter = 10000 * np.ones(N)
            for i,g in enumerate(grid):
                # Refractory dynamics
                tref_over = n_counter >= n_ref
                z[i,] = np.where(tref_over, np.where(z[i,] >= theta, 1, 0), 0)
                s[i,] = np.where(tref_over, np.remainder(z[i,], theta), z[i,])
                n_counter = np.where(s[i,] > 0, 0, n_counter)
                
                z_update, G_memory = G(x[i,], x_star[i,], G_memory)
                z[i+1,] = z[i,] - z[i,]*leak*h + h*z_update # calculate next time step
    
                n_pos += 1
                n_neg += 1

                # implementation of delay
                if i - delta_steps < 0:
                    s_delayed = 0
                else:
                    s_delayed = s[i-delta_steps,]/h

                deriv_x[i,] = f_x(x[i,]) + B @ s_delayed
                x[i+1,] = x[i,] + deriv_x[i,]*h # update state

                if np.linalg.norm(x[i+1,]) > x_max:
                    i_end = i
                    break

        if i_end == -1:
            return z[1:], s[1:], x[1:]
        else:
            return z[1:i_end], s[1:i_end], x[1:i_end]

    def control_exec(self, grid, x_star, n_cutoff=0, avg_out=False):
        t_ref = self.params['t_ref']
        theta = self.params['theta']
        leak = self.params['leak']

        B = self.params['input_matrix']
        if B.ndim == 1:
            K, N = 1, len(B)
        else:
            K, N = B.shape

        assert np.shape(x_star)[0] == len(grid)
        if K > 1:
            assert len(self.params['x0']) == K

        assert len(theta)==N,        "Length of theta does not match shape of input matrix B"
        assert len(leak)==N,         "Length of leak does not match shape of input matrix B"

        
        if x_star.ndim == 1:
            assert 1 == K
            self.size = 1
        else:
            assert np.shape(x_star)[1] == K
            self.size = np.shape(x_star)[1]     # the number of dimension of the state x

        steps = len(grid)
        h = grid[-1]/(steps - 1)
        
        # check correct size of t_ref and transform into an array for vectorization
        self._t_ref_vec = t_ref

        # simulate
        z, s, x = self._simulate(grid, x_star)
        
        return z, s, x

# Generate 1D trajectory for a simple two-neuron spiking control system with given parameters
def generate_1d_traj(a, eps0, theta, leak, grid, target, x0, x_max=np.inf):
    # INPUTS:
    # a: exponential growth rate of uncontrolled dynamics
    # eps0: magnitude of spikes
    # theta: neuron thresholds
    # leak: neuron leakage
    # grid: time grid for simulation
    # target: target trajectory
    # x0: initial state
    # x_max: maximum norm of x before simulation is stopped
    
    K=1
    N=2

    # Define parameters
    t_ref = 0

    B = np.array([-eps0, eps0])

    theta_vec = theta * np.ones(N)
    delay = 0.
    leak_vec = leak * np.ones(N)

    this_scs_dict = {
        'x0': x0,
        'f_x': lambda x: a * x,
        'G': lambda x,y,z: (np.array([np.max([x-y,0]), np.max([y-x,0])]), 0),
        't_ref': t_ref,
        'input_matrix': B,
        'theta': theta_vec,
        'delta': delay,
        'leak': leak_vec,
        'x_max': x_max,
        }

    this_scs = spiking_control_system(this_scs_dict)
    z, s, x = this_scs.control_exec(grid, target)

    n_grid = len(grid)

    if x_max == np.inf:
        # Separate x in continuous parts (x_cont) and jumps (x_jumps) for plotting purposes
        event_times = np.where(np.max(s, axis=1)>0)
        event_times_connected = np.sort(np.concatenate((event_times[0], event_times[0]+1)))
        event_times_connected = event_times_connected[event_times_connected < n_grid]

        x_cont = x[:,].copy()
        x_cont[event_times] = np.nan
        x_jumps = np.empty((n_grid))
        x_jumps[:] = np.nan
        x_jumps[event_times_connected,] = x[event_times_connected,]

        # Define x_c as in the paper
        x_c = x[1:,] + (B @ (z[:-1,].T / theta_vec[:,np.newaxis])).T

        z_cont = z.copy()
        z_cont[event_times] = np.nan
        
        return z, s, x, x_cont, x_jumps, x_c, z_cont
    
    else:
        i_end = len(x)
        return grid[0:i_end], x[0:i_end]

# Generate 2D trajectory for a simple four-neuron spiking control system with given parameters
def generate_2d_traj(a, kappa, eps0, theta, leak, grid, target, x0):
    # PARAMETERS:
    # a: exponential growth rate of uncontrolled dynamics on the radial direction
    # kappa: rotation rate of uncontrolled dynamics on the angular direction
    # eps0: magnitude of spikes
    # theta: neuron thresholds
    # leak: neuron leakage
    # grid: time grid for simulation
    # target: target trajectory
    # x0: initial state

    K=2
    N=4

    # Define parameters
    t_ref = 0

    # Four neurons with spikes in all axes directions
    B = eps0* np.array([-1, 0,
                        1, 0,
                        0,-1,
                        0, 1]).reshape((N,K)).T

    theta_vec = theta * np.ones(N)
    delay = 0.
    leak_vec = leak * np.ones(N)

    # System dynamics with exponential growth a and radial rotation kappa
    A = np.array([a,kappa,
                -kappa,a]).reshape((K,K))

    this_scs_dict = {
        'x0': x0,
        'f_x': lambda x: A @ x,
        'G': lambda x,y,z: (np.array([np.maximum(x[0]-y[0],0),
                                    np.maximum(y[0]-x[0],0),
                                    np.maximum(x[1]-y[1],0),
                                    np.maximum(y[1]-x[1],0)]),0),
        't_ref': t_ref,
        'input_matrix': B,
        'theta': theta_vec,
        'delta': delay,
        'leak': leak_vec,
        }

    this_scs = spiking_control_system(this_scs_dict)
    z, s, x = this_scs.control_exec(grid, target)

    n_grid = len(grid)

    # Separate x in continuous parts (x_cont) and jumps (x_jumps) for plotting purposes
    event_times = np.where(np.max(s, axis=1)>0)
    event_times_connected = np.sort(np.concatenate((event_times[0], event_times[0]+1)))
    event_times_connected = event_times_connected[event_times_connected<n_grid]

    x_cont = x[:,].copy()
    x_cont[event_times,0], x_cont[event_times,1] = np.nan, np.nan
    x_jumps = np.empty((n_grid,2))
    x_jumps[:] = np.nan
    x_jumps[event_times_connected,:] = x[event_times_connected,:]

    # Define x_c as in the paper
    x_c = x[1:,:] + (B @ (z[:-1].T / theta_vec[:,np.newaxis])).T
    
    return z, s, x, x_cont, x_jumps, x_c