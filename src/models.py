# Shockley equation: I = I_s * (exp(V / (n * V_t)) - 1)
# I_s: saturation current
# V: applied voltage
# n: ideality factor
# V_t: threshold voltage

import numpy as np
from scipy.constants import k as k_B, e as q_e

class DiodeModel:
    def __init__(self, T=300):
        """
        Class constructor for a generic diode device at room temperature, calculates thermal voltage for device as well

        Args:
            T (int, optional): temperature, defaults to 300.
        """
        self.temp = T
        self.V_t = (k_B * T) / q_e
        
    def compute_current(self, V, params):
        """
        Compute diode current using the Shockley equation

        Args:
            V (scalar/numpy array): applied voltage
            params: dict with keys 'I_s' (saturation current) and 'n' (ideality factor)

        Returns:
            A: calculated current at each voltage value
        """
        I_s = params['I_s']
        n = params['n']
        exp = np.clip(V / (n * self.V_t), -50, 50) # prevent exponential overflow
        
        return I_s * (np.exp(exp) - 1)
    
    def get_param_bounds(self):
        """
        Returns standard bounds for device parameter

        Returns:
            Dict: standard saturation current (1e-16 to 1e-6 A) and ideality factor range (1.0 to 2.0)
        """
        return {
            'I_s': (1e-16, 1e-6),
            'n': (1.0, 2.0)
        }