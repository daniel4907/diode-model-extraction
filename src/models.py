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
        Class constructor for a generic diode device

        Args:
            T (int, optional): temperature, defaults to 300.
        """
        self.temp = T
        
    def compute_current(self, V, params, T=None):
        """
        Comptue diode current using Shockley equation with Newton-Raphson iteration to account for series resistance

        Args:
            V (scalar/numpy array): applied voltage
            params (dict): model parameters including saturation current, ideality, and series resistance
            T (float, optional): temperature in Kelvin, defaults to model's temperature if None

        Returns:
            Numpy array: calculated current at each voltage point
        """
        I_s = params['I_s']
        n = params['n']
        R_s = params.get('R_s', 0.0)
        
        if T is None:
            T = self.temp
            
        V = np.asarray(V)
        
        def solve_current(V):
            I = 0.0
            
            for _ in range(50):
                Vd = V - I * R_s
                arg = np.clip(q_e * Vd / (n * k_B * T), -50, 50) # prevent exponential overflow
                f_val = I_s * (np.exp(arg) - 1) - I
                df_val = -(I_s * np.exp(arg) * R_s * q_e / (n * k_B * T)) - 1
                
                if abs(df_val) < 1e-15:
                    break
                
                I_new = I - f_val / df_val
                
                if abs(I_new - I) < 1e-12:
                    return I_new
                
                I = I_new
                
            return I
        
        solve_vec = np.vectorize(solve_current, otypes=[float])(V)
        return solve_vec
    
    def compute_sat_current(self, Is, Eg, T, T_ref=300):
        return Is * (T / T_ref)**3 * np.exp(((Eg * q_e) / k_B) * (1/T_ref - 1/T))
    
    def compute_capacitance(self, V, params):
        C_j = params['C_j']
        V_bi = params['V_bi']
        m = params.get('m', 0.5)
        V = np.asarray(V)
        arg = 1 - V / V_bi
        Cj = C_j / np.power(np.maximum(arg, 1e-3), m)
        return Cj
    
    def get_param_bounds(self):
        """
        Returns standard bounds for device parameter

        Returns:
            Dict: standard saturation current, bandgap, ideality factor and series resistance ranges
        """
        return {
            'I_s': (1e-16, 1e-6),
            'Eg': (0.1, 5.0),
            'n': (1.0, 2.0),
            'R_s': (0.0, 10.0),
            'C_j': (1e-15, 1e-6),
            'V_bi': (0.1, 1.5),
            'm': (0.1, 0.9)
        }
        
# 3 regions for MOSFETs: cutoff, triode and saturation regions
# cutoff: I_D = 0
# triode: I_D = mu_n * C_ox * (W / L) * [(V_GS - V_TH) * V_DS - V_DS**2/2], where k_n = mu_n * C_ox * (W / L)
# saturation: I_D = 1/2 k_n (V_GS - V_TH)**2 (1 + lambda * V_DS)

# I_D: drain current
# mu_n: electron mobility in channel
# C_ox: gate-oxide capacitance
# W: channel width
# L: channel length
# V_GS: gate-to-source voltage
# V_TH: threshold voltage
# V_DS: drain-to-soruce voltage
# k_n: transconductance
# lambda: channel-length modulation parameter
        
class MOSFETModel:
    def __init__(self, T=300):
        """
        Class constructor for a generic MOSFET device

        Args:
            T (int, optional): temperature, defaults to 300.
        """
        self.temp = T
        
    def compute_current(self, V_gs, params, T=None):
        V_th = params['V_th']
        k_n = params['k_n']
        V_ds = params['V_ds']
        lam = params.get('lam', 0.0)
        
        V_gs = np.asarray(V_gs)
        I_d = np.zeros_like(V_gs, dtype=float)
        
        if np.ndim(V_ds) == 0:
            V_ds = np.full_like(V_gs, V_ds)
        
        # Cutoff: V_GS <= V_TH, no code needed since I_D will be zero by default
            
        # Triode: V_GS > V_TH, 0 < V_DS < VGS - VTH
        triode = (V_gs > V_th) & (V_ds > 0) & (V_gs - V_th > V_ds)
        if np.any(triode):
            vgs_triode = V_gs[triode]
            vds_triode = V_ds[triode]
            I_d[triode] = k_n * ((vgs_triode - V_th) * vds_triode - 0.5 * vds_triode**2)
            
        # Saturation: V_GS > V_TH, V_DS >= V_GS - V_TH
        saturation = (V_gs > V_th) & (V_ds >= V_gs - V_th)
        if np.any(saturation):
            vgs_sat = V_gs[saturation]
            vds_sat = V_ds[saturation]
            I_d[saturation] = 0.5 * k_n * (vgs_sat - V_th)**2 * (1 + lam * vds_sat)
            
        return I_d
    
    def get_param_bounds(self):
        """
        Returns standard bounds for MOSFET parameters

        Returns:
            Dict: standard threshold voltage, transconductance, and lambda ranges
        """
        return {
            'V_th': (0.1, 5.0),
            'k_n': (1e-9, 1e-1),
            'lam': (0.0, 0.5)
        }
        