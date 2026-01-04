# Measured I-V data but don't know I_s or n, start with initial guesses for both values and then calculate residuals and optimize
# Residual_I = (I_guess - I_data) / np.abs(I_data)
# Minimize sum(residuals**2) by making better guesses

import numpy as np
from scipy.optimize import least_squares

class ModelExtractor:
    def __init__(self, model):
        """
        ModelExtractor constructor for generic device model

        Args:
            model: Instance of device Model class
        """
        self.model = model
        self.result = None
        self.report = None
        
    def diode_fit(self, V_data, I_data, initial_params=None):
        """
        Diode fit algorithm for resolving I_s and n from given I-V curve data

        Args:
            V_data (scalar/numpy array): applied voltage
            I_data (scalar/numpy array): current data
            initial_params (dict, optional): initial parameter guess, defaults to None and uses device bounds

        Returns:
            report: dict summarizing estimated parameters from least squares algorithm and other things
        """
        if initial_params is None:
            initial_params = {'I_s': 1e-12, 'n': 1.0}
        
        def residuals(param_vector):
            """
            Calculates the residuals for the least squares algorithm

            Args:
                param_vector (dict): dict containing current I_s and n guesses

            Returns:
                residual (scalar/numpy array): residual calculated based on guess and data values
            """
            params = {'I_s': param_vector[0], 'n': param_vector[1]}
            I_guess = self.model.compute_current(V_data, params)
            residual = (I_guess - I_data) / np.maximum(np.abs(I_data), 1e-15)
            
            return residual
        
        x0 = np.array([initial_params['I_s'], initial_params['n']])
        bounds = self.model.get_param_bounds()
        lower_bound = np.array([bounds['I_s'][0], bounds['n'][0]])
        upper_bound = np.array([bounds['I_s'][1], bounds['n'][1]])
        ls = least_squares(residuals, x0, bounds=(lower_bound, upper_bound), method='trf')
        
        ls_params = {'I_s': ls.x[0], 'n': ls.x[1]}
        I_fit = self.model.compute_current(V_data, ls_params)
        res = (I_fit - I_data) / np.maximum(np.abs(I_data), 1e-15)
        rms_err = np.sqrt(np.mean(res**2))
        max_err = np.max(np.abs(res))
        
        report = {
            'parameters': ls_params,
            'rms_err': rms_err,
            'max_err': max_err,
            'success': ls.success,
            'num_iters': ls.nfev,
            'message': ls.message,
        }
        
        self.result = ls
        self.report = report
        
        return report
        