'''
To allow exploration during the learning process we need to add some noise
to the prediction actions. We use Ornstein-Uhlenbeck process for that
'''
import numpy as np

class OrnsteinUhlenbeckProcess():
    '''Ornstein-Uhlenbeck process'''

    def __init__(self, size, theta=0.15, mu=.0, sigma=.02, dt=1e-2):
        '''Initialize noise parameters'''
        self.size = size
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.sigma = sigma

        self.state = None
        self.reset()

    def reset(self):
        '''Reset internal state'''
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        '''Sample the noise accorsing to Ornstein-Uhlenbeck process and update internal state'''
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(len(x))
        self.state = x + dx
        return self.state
