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