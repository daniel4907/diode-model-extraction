import numpy as np
import pandas as pd
from scipy.constants import e as q_e, k as k_B

from src.models import MOSFETModel, DiodeModel


def generate_diode_csv(filepath, params, v_sweep, T=300):
    """
    Generates SV file for single-temperature diode I-V data
    """
    model = DiodeModel(T=T)
    i = model.compute_current(v_sweep, params, T=T)
    df = pd.DataFrame({
        'Voltage': v_sweep,
        'Current': i
    })
    
    df.to_csv(filepath, index=False)
    
def generate_multitemp_diode_csv(filepath, params, v_sweep, temps):
    """
    Generates SV file for multi-temperature diode I-V data
    """
    model = DiodeModel()
    data = []
    
    Is = params.get('I_s', 1e-10)
    Eg = params.get('Eg', 1.12)
    T_ref = 300
    
    for T in temps:
        Is_T = Is * (T / T_ref)**3 * np.exp(((Eg * q_e) / k_B) * (1/T_ref - 1/T))
        local_params = params.copy()
        local_params['I_s'] = Is_T
        i = model.compute_current(v_sweep, local_params, T=T)
        for v, i_val in zip(v_sweep, i):
            data.append({'Voltage': v, 'Current': i_val, 'Temperature': T})
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

def generate_mosfet_csv(filepath, params, sweep_type='Id-Vgs', sweep=None, val=1.0):
    """
    Generates CSV file for single-curve MOSFET I-V data
    """
    model = MOSFETModel()
    
    if sweep is None:
        sweep = np.linspace(0, 2, 50)
    
    if sweep_type == 'Id-Vgs':
        vgs = sweep
        vds = val
        local_params = params.copy()
        local_params['V_ds'] = vds
        i_d = model.compute_current(vgs, local_params)
        df = pd.DataFrame({
            'V_Gate': vgs,
            'V_Drain': vds,
            'I_Drain': i_d
        })
        
    else:
        vds = sweep
        vgs = val
        local_params = params.copy()
        local_params['V_ds'] = vds
        vgs_array = np.full_like(vds, vgs)
        i_d = model.compute_current(vgs_array, local_params)
        df = pd.DataFrame({
            'V_Gate': vgs,
            'V_Drain': vds,
            'I_Drain': i_d
        })
    
    df.to_csv(filepath, index=False)

def generate_multi_mosfet_csv(filepath, params, vgs_list, vds_sweep):
    """
    Generates CSV file for multi-curve MOSFET I-V data
    """
    model = MOSFETModel()
    data = []
    for vgs in vgs_list:
        vgs_array = np.full_like(vds_sweep, vgs)
        i_d = model.compute_current(vgs_array, {**params, 'V_ds': vds_sweep})
        for vds, id in zip(vds_sweep, i_d):
            data.append({'V_Gate': vgs, 'V_Drain': vds, 'I_Drain': id})
            
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    
def generate_spice_model(params, device_type, model_name='DUT'):
    """
    Generates a SPICE .MODEL from the extracted parameters

    Args:
        params (dict): extracted device parameters
        device_type (str): 'diode' or 'MOSFET' to define .MODEL device type 
        model_name (str, optional): optional name of the model, defaults to 'DUT'
    
    Returns:
        str: Formatted SPICE model string
    """
    if device_type == 'diode':
        I_s = params.get('I_s', 1e-14)
        n = params.get('n', 1.0)
        R_s = params.get('R_s', 0.0)
        
        param_str = f"IS={I_s:.5e} N={n:.4f} RS={R_s:.4f}"
        
        if 'Eg' in params:
            param_str += f" EG={params['Eg']:.4f}"
        
        if 'C_j' in params:
            param_str += f" CJO={params['C_j']:.5e}"
        if 'V_bi' in params:
            param_str += f" VJ={params['V_bi']:.4f}"
        if 'm' in params:
            param_str += f" M={params['m']:.4f}"
            
        return f".MODEL {model_name} D({param_str})\n"

    elif device_type == 'MOSFET':
        V_th = params.get('V_th', 0.7)
        k_n = params.get('k_n', 1e-4)
        lam = params.get('lam', 0.0)
        
        param_str = f"LEVEL=1 VTO={V_th:.4f} KP={k_n:.5e} LAMBDA={lam:.4f}"
        
        return f".MODEL {model_name} D({param_str})\n"
    
    return ""

def generate_training_data_diode(n_samples, v_range):
    v = np.linspace(0, 1.0, 150)
    x_data, y_data = [], []
    
    model = DiodeModel()
    bounds = model.get_param_bounds()
    
    for _ in range(n_samples):
        low_log = np.log10(bounds['I_s'][0])
        high_log = np.log10(bounds['I_s'][1])
        I_s = 10 ** np.random.uniform(low_log, high_log)
        n = np.random.uniform(bounds['n'][0], bounds['n'][1])
        R_s = np.random.uniform(bounds['R_s'][0], bounds['R_s'][1])
        local_params = {'I_s': I_s, 'n': n, 'R_s': R_s}
        I_true = model.compute_current(v, local_params)
        I_noise = I_true * (1 + np.random.normal(scale=0.01, size=I_true.shape)) # add random gaussian noise
        I_noise = np.abs(I_noise) # avoid negatives from noise
        I_final = np.log10(I_noise + 1e-15) # avoid log(0)
        x_data.append(I_final)
        y_data.append([np.log10(I_s), n, R_s])
        
    return np.array(x_data), np.array(y_data)

def generate_training_data_cv_diode(n_samples):
    v = np.linspace(-5.0, 0, 150)
    x_data, y_data = [], []
    model = DiodeModel()
    bounds = model.get_param_bounds()
    
    for _ in range(n_samples):
        low_log = np.log10(bounds['C_j'][0])
        high_log = np.log10(bounds['C_j'][1])
        C_j = 10 ** np.random.uniform(low_log, high_log)
        v_bi = np.random.uniform(bounds['V_bi'][0], bounds['V_bi'][1])
        m = np.random.uniform(bounds['m'][0], bounds['m'][1])
        local_params = {'C_j': C_j, 'V_bi': v_bi, 'm': m}
        C_true = model.compute_capacitance(v, local_params)
        C_noise = C_true * (1 + np.random.normal(scale=0.02, size=C_true.shape))
        C_noise = np.abs(C_noise)
        C_final = np.log10(C_noise + 1e-20)
        x_data.append(C_final)
        y_data.append([np.log10(C_j), v_bi, m])
        
    return np.array(x_data), np.array(y_data)

def generate_training_transfer_mosfet(n_samples):
    x_data, y_data = [], []
    
    model = MOSFETModel()
    bounds = model.get_param_bounds()
    
    for _ in range(n_samples):
        v_max = np.random.uniform(1.0, 5.0)
        vgs = np.linspace(0, v_max, 150)
        vth = np.random.uniform(bounds['V_th'][0], bounds['V_th'][1])
        low_log = np.log10(bounds['k_n'][0])
        high_log = np.log10(bounds['k_n'][1])
        kn = 10 ** np.random.uniform(low_log, high_log)
        lam = np.random.uniform(bounds['lam'][0], bounds['lam'][1])
        vds = np.random.uniform(0.1, 5.0)
        local_params = {'V_th': vth, 'k_n': kn, 'lam': lam, 'V_ds': vds}
        I_true = model.compute_current(vgs, local_params)
        I_noise = I_true * (1 + np.random.normal(scale=0.01, size=I_true.shape))
        I_noise = np.abs(I_noise)
        I_final = np.log10(I_noise + 1e-15)
        features = np.concatenate([I_final, [vds / 5.0, v_max / 5.0]])
        x_data.append(features)
        y_data.append([vth, np.log10(kn), lam])
        
    return np.array(x_data), np.array(y_data)

def generate_training_output_mosfet(n_samples):
    x_data, y_data = [], []
    
    model = MOSFETModel()
    bounds = model.get_param_bounds()
    
    for _ in range(n_samples):
        v_max = np.random.uniform(1.0, 5.0)
        vds = np.linspace(0, v_max, 150)
        vth = np.random.uniform(bounds['V_th'][0], bounds['V_th'][1])
        low_log = np.log10(bounds['k_n'][0])
        high_log = np.log10(bounds['k_n'][1])
        kn = 10 ** np.random.uniform(low_log, high_log)
        lam = np.random.uniform(bounds['lam'][0], bounds['lam'][1])
        vgs = np.random.uniform(1.0, 5.0)
        local_params = {'V_th': vth, 'k_n': kn, 'lam': lam, 'V_ds': vds}
        vgs_array = np.full_like(vds, vgs)
        I_true = model.compute_current(vgs_array, local_params)
        I_noise = I_true * (1 + np.random.normal(scale=0.01, size=I_true.shape))
        I_noise = np.abs(I_noise)
        I_final = np.log10(I_noise + 1e-15)
        features = np.concatenate([I_final, [vgs / 5.0, v_max / 5.0]])
        x_data.append(features)
        y_data.append([vth, np.log10(kn), lam])
        
    return np.array(x_data), np.array(y_data)