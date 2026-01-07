import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from scipy.constants import k as k_B, e as q_e

sys.path.append(os.getcwd())

from src.models import DiodeModel, MOSFETModel
from src.extraction import ModelExtractor
from src.utils import generate_spice_model

st.set_page_config(page_title="Compact Model Extractor", layout="wide") # set browser tab title and layout format
st.title("Compact Model Parameter Extractor") # display main heading of app

device_type = st.sidebar.selectbox("Select Device Type", ["Diode", "MOSFET"]) # collapsible side menu with widgets

def generate_synthetic_diode(I_s=1e-10, n=1.5, R_s=2.5, points=50, noise=0.02):
    """
    Function to generate synthetic diode data given user-input parameters
    """
    model = DiodeModel()
    V = np.linspace(0, 0.8, points)
    params = {'I_s': I_s, 'n': n, 'R_s': R_s}
    I_true = model.compute_current(V, params)
    np.random.seed(67)
    I_noise = I_true * (1 + np.random.normal(0, noise, size=I_true.shape))
    return pd.DataFrame({'V': V, 'I': I_noise}), model

def generate_synthetic_diode_multitemp(I_s=1e-10, n=1.5, R_s=2.5, Eg=1.12, temps=[280, 300, 320, 340], points=50, noise=0.02):
    """
    Function to generate multi-temperature synthetic diode data
    """
    model = DiodeModel()
    V_sweep = np.linspace(0, 0.8, points)
    T_ref = 300.0
    data = []
    np.random.seed(67)
    for T in temps:
        Is_T = I_s * (T / T_ref)**3 * np.exp(((Eg * q_e) / k_B) * (1/T_ref - 1/T))
        local_params = {'I_s': Is_T, 'n': n, 'R_s': R_s}
        I_ideal = model.compute_current(V_sweep, local_params, T=T)
        I_noise = I_ideal * (1 + np.random.normal(0, noise, size=I_ideal.shape))
        for v, i in zip(V_sweep, I_noise):
            data.append({'V': v, 'I': i, 'T': T})
    return pd.DataFrame(data), model

def generate_synthetic_mosfet(V_th=0.7, k_n=1e-3, lam=0.02, V_ds=1.0, points=50, noise=0.02):
    """
    Function to generate synthetic MOSFET data given user-input parameters
    """
    model = MOSFETModel()
    V_gs = np.linspace(0, 2.0, points)
    params = {'V_th': V_th, 'k_n': k_n, 'lam': lam, 'V_ds': V_ds}
    I_true = model.compute_current(V_gs, params)
    np.random.seed(67)
    I_noise = I_true * (1 + np.random.normal(0, noise, size=I_true.shape))
    return pd.DataFrame({'V_gs': V_gs, 'I_d': I_noise, 'V_ds': V_ds}), model

def generate_synthetic_mosfet_family(V_th=0.7, k_n=1e-3, lam=0.02, V_gs=[1.0, 1.5, 2.0, 2.5], points=50, noise=0.02):
    """
    Function to generate synthetic MOSFET family curves given user-input parameterss
    """
    model = MOSFETModel()
    data = []
    V_ds = np.linspace(0, 5.0, points)
    np.random.seed(4321)
    for vgs in V_gs:
        vgs_array = np.full_like(V_ds, vgs)
        params = {'V_th': V_th, 'k_n': k_n, 'lam': lam, 'V_ds': V_ds}
        I_true = model.compute_current(vgs_array, params)
        I_noise = I_true * (1 + np.random.normal(0, noise, size=I_true.shape))
        
        for vds, id in zip(V_ds, I_noise):
            data.append({'V_gs': vgs, 'V_ds': vds, 'I_d': id})
            
    return pd.DataFrame(data), model

if device_type == "Diode": # Diode logic
    st.header("Diode Extraction")
    col1, col2 = st.columns([1, 2]) # split layout into vertical columns, 2nd column is twice as wide as 1st
    with col1: # context manager, commands inside rendered in first column
        source = st.radio("Data Source", ["Synthetic", "Upload CSV"])
        fit_mode = st.radio("Fit Mode", ["Single Curve", "Multi-Temperature"])
        df = None
        
        if source == "Synthetic": # generate synthetic data
            true_Is = st.number_input("True I_s (A)", value=1e-10, format="%.2e")
            true_n = st.number_input("True n", value=1.5)
            true_Rs = st.number_input("True R_s (Ω)", value=2.5)
            if fit_mode == "Multi-Temperature":
                true_Eg = st.number_input("True E_g (eV)", value=1.12)
                df, model = generate_synthetic_diode_multitemp(true_Is, true_n, true_Rs, true_Eg)
            else:
                df, model = generate_synthetic_diode(true_Is, true_n, true_Rs)
            
        else:
            # key is necessary for state management, when key is changed (csv_diode -> csv_mosfet), the previous widget is destroyed
            csv = st.file_uploader("Upload CSV", type=['csv'], key="csv_diode")
            if csv:
                df = pd.read_csv(csv)
                cols = df.columns.tolist() # column mapping logic
                v_idx = 0
                i_idx = 1 if len(cols) > 1 else 0
                
                v_col = st.selectbox("Voltage Column", cols, index=v_idx)
                i_col = st.selectbox("Current Column", cols, index=i_idx)
                
                rename_dict = {v_col: 'V', i_col: 'I'}
                if fit_mode == "Multi-Temperature":
                    t_idx = 2 if len(cols) > 2 else 0
                    t_col = st.selectbox("Temperature Column", cols, index=t_idx)
                    rename_dict[t_col] = 'T'
                
                df = df.rename(columns=rename_dict)
                
    if df is not None: # only show if data exists
        with col1:
            st.divider()
            st.markdown("**Initial Guesses**") # input widget for initial param guesses
            g_Is = st.number_input("Guess I_s", value=1e-12, format="%.2e")
            g_n = st.number_input("Guess n", value=1.0)
            g_Rs = st.number_input("Guess R_s", value=0.1)
            if fit_mode == "Multi-Temperature": # allow for initial Eg guess if using multi-temp diode data
                g_Eg = st.number_input("Guess E_g (eV)", value=1.1)
                
            run_btn = st.button("Run Extraction", type="primary")
            
        if run_btn: # extraction logic
            model = DiodeModel()
            extractor = ModelExtractor(model)
            
            if fit_mode == "Multi-Temperature":
                datasets = []
                unique_temps = sorted(df['T'].unique())
                for T in unique_temps:
                    sub = df[df['T'] == T].sort_values('V')
                    datasets.append((sub['V'].values, sub['I'].values, float(T)))
                
                initial = {'I_s': g_Is, 'n': g_n, 'R_s': g_Rs, 'Eg': g_Eg}
                report = extractor.diode_temp_fit(datasets, initial_params=initial)
                
                st.success("Global Extraction Converged") # display success message box
                m1, m2, m3, m4 = st.columns(4)
                m1.metric('I_s (ref)', f"{report['parameters']['I_s']:.2e} A") # display label and value
                m2.metric('E_g', f"{report['parameters']['Eg']:.4f} eV")
                m3.metric('n', f"{report['parameters']['n']:.4f}")
                m4.metric('R_s', f"{report['parameters']['R_s']:.4f} Ω")
                
                st.divider()
                st.subheader("SPICE Model")
                spice_str = generate_spice_model(report['parameters'], "diode", model_name="Multitemp_Diode")
                st.code(spice_str, language='spice')
                
                st.download_button(
                    label="Download Model File",
                    data=spice_str,
                    file_name="diode_model.lib",
                    mime="text/plain"
                )
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.plasma(np.linspace(0, 1, len(datasets)))
                T_ref = 300.0
                fit_params = report['parameters']
                
                for idx, (v_data, i_data, T) in enumerate(datasets):
                    c = colors[idx]
                    ax.semilogy(v_data, i_data, 'o', alpha=0.4, color=c, label=f'{T}K Data')
                    
                    # Compute local fit curve
                    Is_T = fit_params['I_s'] * (T / T_ref)**3 * np.exp(((fit_params['Eg'] * q_e) / k_B) * (1/T_ref - 1/T))
                    p_local = {'I_s': Is_T, 'n': fit_params['n'], 'R_s': fit_params['R_s']}
                    i_fit = model.compute_current(v_data, p_local, T=T)
                    ax.semilogy(v_data, i_fit, '-', color=c, label=f'{T}K Fit')
                
                ax.set_xlabel("Voltage [V]")
                ax.set_ylabel("Current [A]")
                ax.set_title("Global Diode Fit (Temperature Dependence)")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig)
                
            else: # Single Curve Mode
                initial = {'I_s': g_Is, 'n': g_n, 'R_s': g_Rs}
                report = extractor.diode_fit(df['V'].values, df['I'].values, initial_params=initial)
                st.success("Extraction Converged")
                m1, m2, m3 = st.columns(3)
                m1.metric('I_s', f"{report['parameters']['I_s']:.2e} A")
                m2.metric('n', f"{report['parameters']['n']:.4f}")
                m3.metric('R_s', f"{report['parameters']['R_s']:.4f} Ω")
                
                st.divider()
                st.subheader("SPICE Model")
                spice_str = generate_spice_model(report['parameters'], "diode", model_name="Singletemp_Diode")
                st.code(spice_str, language='spice')
                
                st.download_button(
                    label="Download Model File",
                    data=spice_str,
                    file_name="diode_model.lib",
                    mime="text/plain"
                )
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.semilogy(df['V'], df['I'], 'o', alpha=0.5, label='Data')
                I_fit = model.compute_current(df['V'].values, report['parameters'])
                ax.semilogy(df['V'], I_fit, 'r-', label='Fit')
                ax.set_xlabel("Voltage [V]")
                ax.set_ylabel("Current [A]")
                ax.legend()
                st.pyplot(fig)

elif device_type == "MOSFET": # MOSFET logic
    st.header("MOSFET Extraction")
    col1, col2 = st.columns([1, 2])
    with col1:
        source = st.radio("Data Source", ["Synthetic", "Upload CSV"])
        df = None
        
        if source == "Synthetic":
            true_Vth = st.number_input("True V_th (V)", value=0.7)
            true_kn = st.number_input("True n", value=1e-3, format="%.2e")
            true_lam = st.number_input("True lambda", value=0.02)
            synth_type = st.radio("Synthetic Data Type", ["Transfer Curve (Id-Vgs)", "Output Family (Id-Vds)"])
            
            if synth_type == "Transfer Curve (Id-Vgs)":
                df, model = generate_synthetic_mosfet(true_Vth, true_kn, true_lam)
            else:
                df, model = generate_synthetic_mosfet_family(true_Vth, true_kn, true_lam, V_gs=[1.5, 2.0, 2.5, 3.0])
            
        else:
            csv = st.file_uploader("Upload CSV", type=['csv'], key="csv_mosfet")
            if csv:
                df = pd.read_csv(csv)
                cols = df.columns.tolist()
                
                vg_idx = 0
                vd_idx = 1 if len(cols) > 1 else 0
                id_idx = 2 if len(cols) > 2 else 0
                
                vg_col = st.selectbox("V_gs Column", cols, index=vg_idx)
                vd_col = st.selectbox("V_ds Column", cols, index=vd_idx)
                id_col = st.selectbox("I_d Column", cols, index=id_idx)
                
                df = df.rename(columns={vg_col: 'V_gs', id_col: 'I_d', vd_col: 'V_ds'})
                
    if df is not None:
        st.subheader("Configuration")
        fit_mode = st.radio("Fit Mode", ["Single Curve", "Multi-Curve (Output Family)"])
        
        with col1:
            st.divider()
            st.markdown("**Initial Guesses**")
            g_Vth = st.number_input("Guess V_th", value=0.5)
            g_kn = st.number_input("Guess k_n", value=1e-4, format="%.2e")

        if fit_mode == "Multi-Curve (Output Family)": # detect how many unique family curves exist within CSV
            st.info(f"Detected {df['V_gs'].nunique()} unique V_gs curves for global fit.")
            
            if st.button("Run Global Extraction", type="primary"):
                model = MOSFETModel()
                extractor = ModelExtractor(model)
                initial = {'V_th': g_Vth, 'k_n': g_kn, 'lam': 0.0}
                
                datasets = []
                unique_vgs = sorted(df['V_gs'].unique())
                for vgs in unique_vgs:
                    sub = df[df['V_gs'] == vgs].sort_values('V_ds')
                    datasets.append((sub['V_ds'].values, sub['I_d'].values, float(vgs)))
                
                report = extractor.multi_mosfet_fit(datasets, initial_params=initial)
                
                st.success("Global Extraction Converged")
                m1, m2, m3 = st.columns(3)
                m1.metric('V_th', f"{report['parameters']['V_th']:.4f} V")
                m2.metric('k_n', f"{report['parameters']['k_n']:.2e}")
                m3.metric('lambda', f"{report['parameters']['lam']:.4f}")
                
                st.divider()
                st.subheader("SPICE Model")
                spice_str = generate_spice_model(report['parameters'], "MOSFET", model_name="Multicurve_FET")
                st.code(spice_str, language='spice')
                
                st.download_button(
                    label="Download Model File",
                    data=spice_str,
                    file_name="mosfet_model.lib",
                    mime="text/plain"
                )
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.jet(np.linspace(0, 1, len(datasets)))
                
                for idx, (vds_data, id_data, vgs_val) in enumerate(datasets):
                    c = colors[idx]
                    ax.plot(vds_data, id_data, 'o', alpha=0.4, color=c, label=f'Data {vgs_val}V')
                    
                    p = report['parameters'].copy()
                    p['V_ds'] = vds_data
                    vgs_arr = np.full_like(vds_data, vgs_val)
                    I_fit = model.compute_current(vgs_arr, p)
                    
                    ax.plot(vds_data, I_fit, '-', color=c, label=f'Fit {vgs_val}V')
                
                ax.set_xlabel("V_ds [V]")
                ax.set_ylabel("I_d [A]")
                ax.set_title("Global Fit: Output Characteristics")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig)

        else: # Single Curve Mode
            sweep_type = st.radio("Sweep Type", ["Id-Vgs (Transfer)", "Id-Vds (Output)"])
            
            subset = None
            sel_param = 0.0
            
            if sweep_type == "Id-Vgs (Transfer)":
                params = df['V_ds'].unique()
                sel_param = params[0]
                if len(params) > 1:
                    sel_param = st.selectbox("Select V_ds", params)
                subset = df[df['V_ds'] == sel_param].sort_values('V_gs')
                
            else: # Id-Vds
                params = df['V_gs'].unique()
                sel_param = params[0]
                if len(params) > 1:
                    sel_param = st.selectbox("Select V_gs", params)
                subset = df[df['V_gs'] == sel_param].sort_values('V_ds')
            
            if st.button("Run Extraction", type="primary"):
                model = MOSFETModel()
                extractor = ModelExtractor(model)
                initial = {'V_th': g_Vth, 'k_n': g_kn, 'lam': 0.0}
                
                if sweep_type == "Id-Vgs (Transfer)":
                     v_gs_arg = subset['V_gs'].values
                     v_ds_arg = float(sel_param)
                else:
                     v_ds_arg = subset['V_ds'].values
                     v_gs_arg = np.full_like(v_ds_arg, float(sel_param))
                
                report = extractor.mosfet_fit(
                    v_gs_arg,
                    subset['I_d'].values,
                    V_ds=v_ds_arg,
                    initial_params=initial
                )
                
                st.success("Extraction Converged")
                m1, m2, m3 = st.columns(3)
                m1.metric('V_th', f"{report['parameters']['V_th']:.4f} V")
                m2.metric('k_n', f"{report['parameters']['k_n']:.2e}")
                m3.metric('lambda', f"{report['parameters']['lam']:.4f}")
                
                st.divider()
                st.subheader("SPICE Model")
                spice_str = generate_spice_model(report['parameters'], "MOSFET", model_name="Singlecurve_FET")
                st.code(spice_str, language='spice')
                
                st.download_button(
                    label="Download Model File",
                    data=spice_str,
                    file_name="mosfet_model.lib",
                    mime="text/plain"
                )
                
                fig, ax = plt.subplots(figsize=(10, 5))
                
                if sweep_type == "Id-Vgs (Transfer)":
                    x_data = subset['V_gs']
                    x_label = "V_gs [V]"
                else:
                    x_data = subset['V_ds']
                    x_label = "V_ds [V]"
                    
                ax.plot(x_data, subset['I_d'], 'o', alpha=0.5, label='Data')
                
                fit_params = report['parameters']
                fit_params['V_ds'] = v_ds_arg
                I_fit = model.compute_current(v_gs_arg, fit_params)
                
                ax.plot(x_data, I_fit, 'r-', label='Fit')
                ax.set_xlabel(x_label)
                ax.set_ylabel('I_d [A]')
                ax.legend()
                st.pyplot(fig)