# Measured I-V data but don't know I_s or n, start with initial guesses for both values and then calculate residuals and optimize
# Residual_I = (I_guess - I_data) / np.abs(I_data)
# Minimize sum(residuals**2) by making better guesses

import numpy as np
import torch
import os 

from src.models import *
from scipy.optimize import least_squares
from scipy.constants import k as k_B, e as q_e

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
        
    def _get_diode_ml_guess(self, V_data, I_data):
        model_path = 'models/diode_model_weights.pth'
        if not os.path.exists(model_path):
            return None
        
        try:
            target_v = np.linspace(0, 1.0, 150)
            sort_idx = np.argsort(V_data) # sort data to get interpolation to work
            v_sort = V_data[sort_idx]
            i_sort = I_data[sort_idx]
            i_interp = np.interp(target_v, v_sort, i_sort)
            i_log = np.log10(np.abs(i_interp) + 1e-15) # log transform to match training data preprocessing
            inputs = torch.tensor(i_log, dtype=torch.float32).unsqueeze(0) # (1, 150)
            
            model = DiodeNet(input_size=150)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            with torch.no_grad():
                preds = model(inputs).numpy()[0]
                
            return {'I_s': 10 ** preds[0], 'n': preds[1], 'R_s': preds[2]}
        
        except Exception as e:
            print(f"Diode I-V ML interference warning: {e}")
            return None

    def diode_fit(self, V_data, I_data, T=None, initial_params=None):
        """
        Fit a single I-V curve to extract saturation current, ideality factor and series resistance of a device

        Args:
            V_data (scalar/numpy array): applied voltage
            I_data (scalar/numpy array): measured current
            T (float, optional): temperature in Kelvin, defaults to model temperature if None
            initial_params (dict, optional): initial guess for I_s, n and R_s

        Returns:
            dict: report containing fitted parameters, errors and solver status
        """
        if initial_params is None:
            ml_g = self._get_diode_ml_guess(V_data, I_data)
            if ml_g:
                initial_params = ml_g
            else:
                initial_params = {'I_s': 1e-12, 'n': 1.0, 'R_s': 0.1}
            
        if 'I_s' not in initial_params:
            initial_params['I_s'] = 1e-12
        if 'n' not in initial_params:
            initial_params['n'] = 1.0
        if 'R_s' not in initial_params:
            initial_params['R_s'] = 0.1
        
        def residuals(param_vector, V_data, I_data, T):
            """
            Calculates normalized current residuals
            """
            params = {'I_s': param_vector[0], 'n': param_vector[1], 'R_s': param_vector[2]}
            I_guess = self.model.compute_current(V_data, params, T=T)
            residual = (I_guess - I_data) / np.maximum(np.abs(I_data), 1e-15)
            
            return residual
        
        x0 = np.array([initial_params['I_s'], initial_params['n'], initial_params['R_s']])
        bounds = self.model.get_param_bounds()
        lower_bound = np.array([bounds['I_s'][0], bounds['n'][0], bounds['R_s'][0]])
        upper_bound = np.array([bounds['I_s'][1], bounds['n'][1], bounds['R_s'][1]])
        ls = least_squares(
            residuals, 
            x0, 
            bounds=(lower_bound, upper_bound), 
            args=(V_data, I_data, T), 
            method='trf'
        )
        
        ls_params = {'I_s': ls.x[0], 'n': ls.x[1], 'R_s': ls.x[2]}
        I_fit = self.model.compute_current(V_data, ls_params, T=T)
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
        
    def diode_temp_fit(self, datasets, initial_params=None):
        """
        Perform a simulatenous fit on multiple I-V datasets at different temperatures

        Args:
            datasets (list): list ot tuples (V_data, I_data, T)
            initial_params (dict, optional): initial guesses for I_s, Eg, n and R_s

        Returns:
            dict: report contianing fitted gloabl parameters and combined RMS error
        """
        T_ref = 300.0
        
        if initial_params is None:
            initial_params = {'I_s': 1e-12, 'Eg': 1.12, 'n': 1.0, 'R_s': 0.1}
            
        if 'I_s' not in initial_params:
            initial_params['I_s'] = 1e-12
        if 'Eg' not in initial_params:
            initial_params['Eg'] = 1.12
        if 'n' not in initial_params:
            initial_params['n'] = 1.0
        if 'R_s' not in initial_params:
            initial_params['R_s'] = 0.1
        
        def Is_at_T(Is_ref, Eg, T):
            return Is_ref * (T / T_ref)**3 * (np.exp(((Eg * q_e) / k_B) * (1/T_ref - 1/T)))
        
        def global_residuals(param_vector):
            """
            Calculates normalized current residuals
            """
            Is_ref, Eg, n, Rs = param_vector
            residuals = []
            
            for V, I_measured, T in datasets:
                Is_local = Is_at_T(Is_ref, Eg, T)
                local_params = {'I_s': Is_local, 'n': n, 'R_s': Rs}
                I_data = self.model.compute_current(V, local_params, T=T)
                residual = (I_data - I_measured) / np.maximum(np.abs(I_measured), 1e-15)
                residuals.append(residual)
            
            return np.concatenate(residuals)
        
        x0 = np.array([initial_params['I_s'], initial_params['Eg'], initial_params['n'], initial_params['R_s']])
        bounds = self.model.get_param_bounds()
        lower_bound = np.array([bounds['I_s'][0], bounds['Eg'][0], bounds['n'][0], bounds['R_s'][0]])
        upper_bound = np.array([bounds['I_s'][1], bounds['Eg'][1], bounds['n'][1], bounds['R_s'][1]])
        ls = least_squares(
            global_residuals, 
            x0, 
            bounds=(lower_bound, upper_bound),  
            method='trf'
        )
        
        ls_params = {'I_s': ls.x[0], 'Eg': ls.x[1], 'n': ls.x[2], 'R_s': ls.x[3]}
        res = global_residuals(ls.x)
        rms_err = np.sqrt(np.mean(res**2))
        
        report = {
            'parameters': ls_params,
            'rms_err': rms_err,
            'success': ls.success,
            'num_iters': ls.nfev,
            'message': ls.message,
        }
        
        self.result = ls
        self.report = report
        
        return report
    
    def _get_diode_cv_guess(self, V_data, C_data):
        model_path = 'models/diode_cv_model_weights.pth'
        if not os.path.exists(model_path):
            return None
        
        try:
            target_v = np.linspace(-5.0, 0.0, 150)
            sort_idx = np.argsort(V_data) # sort data to get interpolation to work
            v_sort = V_data[sort_idx]
            c_sort = C_data[sort_idx]
            c_interp = np.interp(target_v, v_sort, c_sort)
            c_log = np.log10(np.abs(c_interp) + 1e-20) # log transform to match training data preprocessing
            inputs = torch.tensor(c_log, dtype=torch.float32).unsqueeze(0) # (1, 150)
            
            model = DiodeNet(input_size=150)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            with torch.no_grad():
                preds = model(inputs).numpy()[0]
                
            return {'C_j': 10 ** preds[0], 'V_bi': preds[1], 'm': preds[2]}
        
        except Exception as e:
            print(f"Diode C-V ML interference warning: {e}")
            return None
    
    def diode_cv_fit(self, V_data, C_data, initial_params=None):
        if initial_params is None:
            ml_g = self._get_diode_cv_guess(V_data, C_data)
            if ml_g:
                initial_params = ml_g
            else:
                initial_params = {'C_j': 1e-12, 'V_bi': 0.7, 'm': 0.5}
        
        if 'C_j' not in initial_params:
            initial_params['C_j'] = 1e-12
        if 'V_bi' not in initial_params:
            initial_params['V_bi'] = 0.7
        if 'm' not in initial_params:
            initial_params['m'] = 0.5
        
        def residuals(param_vector, V_data, C_data):
            local_params = {'C_j': param_vector[0], 'V_bi': param_vector[1], 'm': param_vector[2]}
            C_guess = self.model.compute_capacitance(V_data, local_params)
            residual = (C_guess - C_data) / np.maximum(np.abs(C_data), 1e-15)
            return residual
        
        x0 = np.array([initial_params['C_j'], initial_params['V_bi'], initial_params['m']])
        bounds = self.model.get_param_bounds()
        lower_bound = np.array([bounds['C_j'][0], bounds['V_bi'][0], bounds['m'][0]])
        upper_bound = np.array([bounds['C_j'][1], bounds['V_bi'][1], bounds['m'][1]])
        ls = least_squares(
            residuals,
            x0,
            bounds=(lower_bound, upper_bound),
            args=(V_data, C_data),
            method='trf'
        )
        
        ls_params = {'C_j': ls.x[0], 'V_bi': ls.x[1], 'm': ls.x[2]}
        res = residuals(ls.x, V_data, C_data)
        rms_err = np.sqrt(np.mean(res**2))
        
        report = {
            'parameters': ls_params,
            'rms_err': rms_err,
            'success': ls.success,
            'num_iters': ls.nfev,
            'message': ls.message,
        }
        
        self.result = ls
        self.report = report
        
        return report
    
    def _get_mosfet_transfer_ml_guess(self, V_data, I_data, V_ds):
        model_path = 'models/mosfet_transfer_model_weights.pth'
        if not os.path.exists(model_path):
            return None
        
        try:
            if np.ndim(V_ds) > 0:
                vds = float(np.mean(V_ds))
            else:
                vds = float(V_ds)
                
            v_max = float(np.max(V_data))
            target_v = np.linspace(0, v_max, 150)
            sort_idx = np.argsort(V_data) # sort data to get interpolation to work
            v_sort = V_data[sort_idx]
            i_sort = I_data[sort_idx]
            i_interp = np.interp(target_v, v_sort, i_sort)
            i_log = np.log10(np.abs(i_interp) + 1e-15) # log transform to match training data preprocessing
            
            meta = np.array([vds / 5.0, v_max / 5.0])
            features = np.concatenate([i_log, meta])
            inputs = torch.tensor(features, dtype=torch.float32).unsqueeze(0) # (1, 152)
            
            model = MOSFETNet(input_size=152)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            with torch.no_grad():
                preds = model(inputs).numpy()[0]
                
            return {'V_th': preds[0], 'k_n': 10 ** preds[1], 'lam': preds[2]}
        
        except Exception as e:
            print(f"MOSFET transfer ML interference warning: {e}")
            return None
        
    def _get_mosfet_output_ml_guess(self, V_data, I_data, V_gs):
        model_path = 'models/mosfet_output_model_weights.pth'
        if not os.path.exists(model_path):
            return None
        
        try:
            if np.ndim(V_gs) > 0:
                vgs = float(np.mean(V_gs))
            else:
                vgs = float(V_gs)
                
            v_max = float(np.max(V_data))
            target_v = np.linspace(0, v_max, 150)
            sort_idx = np.argsort(V_data) # sort data to get interpolation to work
            v_sort = V_data[sort_idx]
            i_sort = I_data[sort_idx]
            i_interp = np.interp(target_v, v_sort, i_sort)
            i_log = np.log10(np.abs(i_interp) + 1e-15) # log transform to match training data preprocessing
            
            meta = np.array([vgs / 5.0, v_max / 5.0])
            features = np.concatenate([i_log, meta])
            inputs = torch.tensor(features, dtype=torch.float32).unsqueeze(0) # (1, 152)
            
            model = MOSFETNet(input_size=152)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            with torch.no_grad():
                preds = model(inputs).numpy()[0]
                
            return {'V_th': preds[0], 'k_n': 10 ** preds[1], 'lam': preds[2]}
        
        except Exception as e:
            print(f"MOSFET output ML interference warning: {e}")
            return None

    def mosfet_fit(self, V_gs, I_data, V_ds, initial_params=None):
        """
        Fit a single I-V curve to extract threshold voltage, transconductance, lambda, and drain-to-source voltage of a device

        Args:
            V_gs (scalar/numpy array): gate-to-source voltage
            I_data (scalar/numpy array): collected current values
            V_ds (scalar/numpy array): drain-to-source voltage
            initial_params (dict, optional): initial guesses for V_th, k_n, and lambda

        Returns:
            dict: report containing fitted parameters, errors and solver status
        """
        if initial_params is None:
            is_transfer = np.std(V_gs) > np.std(V_ds)
            ml_g = None
            if is_transfer:
                ml_g = self._get_mosfet_transfer_ml_guess(V_gs, I_data, V_ds)
            else:
                ml_g = self._get_mosfet_output_ml_guess(V_ds, I_data, V_gs)
            
            if ml_g:
                initial_params = ml_g
            else:
                initial_params = {'V_th': 0.5, 'k_n': 1e-4, 'lam': 0.0}
            
        if 'V_th' not in initial_params:
            initial_params['V_th'] = 0.5
        if 'k_n' not in initial_params:
            initial_params['k_n'] = 1e-4
        if 'lam' not in initial_params:
            initial_params['lam'] = 0.0
            
        I_check = self.model.compute_current(V_gs, initial_params)
        if np.all(I_check <= 1e-15) and np.any(I_data > 1e-9):
            print("Warning: initial guess places devices in cutoff, adjusting v_th")
            pos_vgs = V_gs[V_gs > 0]
            if len(pos_vgs) > 0:
                initial_params['V_th'] = max(0.1, np.min(pos_vgs) * 0.5)
            else:
                initial_params['V_th'] = 0.5
            
        def residuals(param_vector, V_gs, I_data, V_ds):
            """
            Calculates normalized current residuals
            """
            params = {'V_th': param_vector[0], 'k_n': param_vector[1], 'lam': param_vector[2], 'V_ds': V_ds}
            I_guess = self.model.compute_current(V_gs, params)
            residual = (I_guess - I_data) / np.maximum(np.abs(I_data), 1e-15)
            
            return residual
        
        x0 = np.array([initial_params['V_th'], initial_params['k_n'], initial_params['lam']])
        bounds = self.model.get_param_bounds()
        lower_bound = np.array([bounds['V_th'][0], bounds['k_n'][0], bounds['lam'][0]])
        upper_bound = np.array([bounds['V_th'][1], bounds['k_n'][1], bounds['lam'][1]])
        ls = least_squares(
            residuals, 
            x0, 
            bounds=(lower_bound, upper_bound), 
            args=(V_gs, I_data, V_ds), 
            method='trf'
        )
        
        ls_params = {'V_th': ls.x[0], 'k_n': ls.x[1], 'lam': ls.x[2], 'V_ds': V_ds}
        I_fit = self.model.compute_current(V_gs, ls_params)
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
    
    def multi_mosfet_fit(self, datasets, initial_params=None):
        """
        Fits multiple I-V curves to extract threshold voltage, transconductance and lambda of a device

        Args:
            datasets (list): list ot tuples (V_ds, I_measured, V_gs)
            initial_params (dict, optional): initial guesses for V_th, k_n and lam

        Returns:
            dict: report containing fitted parameters, errors and solver status
        """
        if initial_params is None:
            try: # make guess using higheest v_gs
                best_idx = np.argmax([d[2] for d in datasets])
                v_ds, i, v_gs = datasets[best_idx]
                ml_g = self._get_mosfet_output_ml_guess(v_ds, i, v_gs)
                
                if ml_g:
                    initial_params = ml_g
            except Exception as e:
                print(f"Multi-curve ML guess failed: {e}")
                
            if initial_params is None:
                initial_params = {'V_th': 0.5, 'k_n': 1e-4, 'lam': 0.0}
            
        if 'V_th' not in initial_params:
            initial_params['V_th'] = 0.5
        if 'k_n' not in initial_params:
            initial_params['k_n'] = 1e-4
        if 'lam' not in initial_params:
            initial_params['lam'] = 0.0
            
        def global_residuals(param_vector):
            """
            Calculates normalized current residuals
            """
            V_th, k_n, lam = param_vector
            residuals = []
            
            for V_ds, I_measured, V_gs in datasets:
                local_params = {'V_th': V_th, 'k_n': k_n, 'lam': lam, 'V_ds': V_ds}
                Vgs_array = np.full_like(V_ds, V_gs)
                I_data = self.model.compute_current(Vgs_array, local_params)
                residual = (I_data - I_measured) / np.maximum(np.abs(I_measured), 1e-15)
                residuals.append(residual)
            
            return np.concatenate(residuals)
        
        x0 = np.array([initial_params['V_th'], initial_params['k_n'], initial_params['lam']])
        bounds = self.model.get_param_bounds()
        lower_bound = np.array([bounds['V_th'][0], bounds['k_n'][0], bounds['lam'][0]])
        upper_bound = np.array([bounds['V_th'][1], bounds['k_n'][1], bounds['lam'][1]])
        ls = least_squares(
            global_residuals, 
            x0, 
            bounds=(lower_bound, upper_bound), 
            method='trf'
        )
        
        ls_params = {'V_th': ls.x[0], 'k_n': ls.x[1], 'lam': ls.x[2]}
        res = global_residuals(ls.x)
        rms_err = np.sqrt(np.mean(res**2))
        
        report = {
            'parameters': ls_params,
            'rms_err': rms_err,
            'success': ls.success,
            'num_iters': ls.nfev,
            'message': ls.message,
        }
        
        self.result = ls
        self.report = report
        
        return report
    