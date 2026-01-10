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
from src.visualization import *
from src.physics import DiodePhysics, MOSFETPhysics

st.set_page_config(page_title="Compact Model Extractor", layout="wide") # set browser tab title and layout format
st.title("Compact Model Parameter Extractor") # display main heading of app

device_type = st.sidebar.selectbox("Select Device Type", ["Diode", "MOSFET"]) # collapsible side menu with widgets

if device_type == "Diode" or device_type == "MOSFET":
    app_mode = st.sidebar.radio("App Mode", ["Extraction", "Physics Explorer"])

def generate_synthetic_diode_iv(I_s=1e-10, n=1.5, R_s=2.5, points=50, noise=0.02):
    """
    Function to generate synthetic diode I-V data given user-input parameters
    """
    model = DiodeModel()
    V = np.linspace(0, 0.8, points)
    params = {'I_s': I_s, 'n': n, 'R_s': R_s}
    I_true = model.compute_current(V, params)
    np.random.seed(67)
    I_noise = I_true * (1 + np.random.normal(0, noise, size=I_true.shape))
    return pd.DataFrame({'V': V, 'I': I_noise}), model

def generate_synthetic_diode_cv(C_j=1e-12, V_bi=0.7, m=0.5, points=50, noise=0.02):
    """
    Function to generate synthetic diode C-V data given user-input parameters
    """
    model = DiodeModel()
    V = np.linspace(-5, 0, points)
    params = {'C_j': C_j, 'V_bi': V_bi, 'm': m}
    C_true = model.compute_capacitance(V, params)
    np.random.seed(1273)
    C_noise = C_true * (1 + np.random.normal(0, noise, size=C_true.shape))
    return pd.DataFrame({'V': V, 'C': C_noise}), model

def generate_synthetic_diode_multitemp(I_s=1e-10, n=1.5, R_s=2.5, Eg=1.12, temps=[280, 300, 320, 340], points=50, noise=0.02):
    """
    Function to generate multi-temperature synthetic diode I-V data
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
    if app_mode == "Extraction":
        st.header("Diode Extraction")
        col1, col2 = st.columns([1, 2]) # split layout into vertical columns, 2nd column is twice as wide as 1st
        with col1: # context manager, commands inside rendered in first column
            source = st.radio("Data Source", ["Synthetic", "Upload CSV"])
            fit_mode = st.radio("Fit Mode", ["I-V Curve", "Multi-Temperature I-V", "C-V Curve"])
            df = None
            
            if source == "Synthetic": # generate synthetic data
                if fit_mode == "C-V Curve":
                    true_Cj = st.number_input("True $C_{j}$ (F)", value=1e-12, format="%.2e")
                    true_Vbi = st.number_input("True $V_{bi}$ (V)", value=0.7)
                    true_m = st.number_input("True m", value=0.5)
                    df, model = generate_synthetic_diode_cv(true_Cj, true_Vbi, true_m)
                elif fit_mode == "Multi-Temperature I-V":
                    true_Is = st.number_input("True $I_{s}$ (A)", value=1e-10, format="%.2e")
                    true_n = st.number_input("True n", value=1.5)
                    true_Rs = st.number_input("True $R_{s}$ (Ω)", value=2.5)
                    true_Eg = st.number_input("True $E_{g}$ (eV)", value=1.12)
                    df, model = generate_synthetic_diode_multitemp(true_Is, true_n, true_Rs, true_Eg)
                else: # single temp diode I-V
                    true_Is = st.number_input("True $I_{s}$ (A)", value=1e-10, format="%.2e")
                    true_n = st.number_input("True n", value=1.5)
                    true_Rs = st.number_input("True $R_{s}$ (Ω)", value=2.5)
                    df, model = generate_synthetic_diode_iv(true_Is, true_n, true_Rs)
                
            else:
                # key is necessary for state management, when key is changed (csv_diode -> csv_mosfet), the previous widget is destroyed
                csv = st.file_uploader("Upload CSV", type=['csv'], key="csv_diode")
                if csv:
                    df = pd.read_csv(csv)
                    cols = df.columns.tolist() # column mapping logic
                    v_idx = 0
                    v_col = st.selectbox("Voltage Column", cols, index=v_idx)
                    
                    if fit_mode == "C-V Curve":
                        c_idx = 1 if len(cols) > 1 else 0
                        c_col = st.selectbox("Capacitance Column", cols, index=c_idx)
                        rename_dict = {v_col: 'V', c_col: 'C'}
                    else:
                        i_idx = 1 if len(cols) > 1 else 0
                        i_col = st.selectbox("Current Column", cols, index=i_idx)
                        rename_dict = {v_col: 'V', i_col: 'I'}
                    
                    if fit_mode == "Multi-Temperature I-V":
                        t_idx = 2 if len(cols) > 2 else 0
                        t_col = st.selectbox("Temperature Column", cols, index=t_idx)
                        rename_dict[t_col] = 'T'
                    
                    df = df.rename(columns=rename_dict)
                    
        if df is not None: # only show if data exists
            with col1:
                st.divider()
                st.markdown("**Initial Guesses**") # input widget for initial param guesses
                
                def update_guess(params):
                    st.session_state['guess_Is'] = float(params['I_s'])
                    st.session_state['guess_n'] = float(params['n'])
                    st.session_state['guess_Rs'] = float(params['R_s'])
                
                if fit_mode == 'C-V Curve':
                    if 'guess_Cj' not in st.session_state: st.session_state['guess_Cj'] = 1e-12
                    if 'guess_Vbi' not in st.session_state: st.session_state['guess_Vbi'] = 0.7
                    if 'guess_m' not in st.session_state: st.session_state['guess_m'] = 0.5
                    
                    if source != "Synthetic":
                        def run_diode_cv_guess():
                            temp_model = DiodeModel()
                            temp_ext = ModelExtractor(temp_model)
                            pred = temp_ext._get_diode_cv_guess(df['V'].values, df['C'].values)
                            if pred:
                                st.session_state['guess_Cj'] = float(pred['C_j'])
                                st.session_state['guess_Vbi'] = float(pred['V_bi'])
                                st.session_state['guess_m'] = float(pred['m'])
                                st.toast("ML Prediction Applied")
                            else:
                                st.toast("ML Prediction Failed")
                                
                        st.button("Auto-Guess Parameters (ML)", on_click=run_diode_cv_guess)
                        
                    g_Cj = st.number_input("Guess $C_j$", value=1e-12, format="%.2e", key='guess_Cj')
                    g_Vbi = st.number_input("Guess $V_{bi}$", value=0.7, key='guess_Vbi')
                    g_m = st.number_input("Guess $m$", value=0.5, key='guess_m')
                else: 
                    if 'guess_Is' not in st.session_state: st.session_state['guess_Is'] = 1e-12
                    if 'guess_n' not in st.session_state: st.session_state['guess_n'] = 1.0
                    if 'guess_Rs' not in st.session_state: st.session_state['guess_Rs'] = 0.1
                    
                    if source != "Synthetic":
                        def run_ml_guess():
                            temp_model = DiodeModel()
                            temp_ext = ModelExtractor(temp_model)
                            pred = temp_ext._get_diode_ml_guess(df['V'].values, df['I'].values)
                            if pred:
                                st.session_state['guess_Is'] = float(pred['I_s'])
                                st.session_state['guess_n'] = float(pred['n'])
                                st.session_state['guess_Rs'] = float(pred['R_s'])
                                st.toast("ML Prediction Applied")
                            else:
                                st.toast("ML Prediction Failed")
                            
                        st.button("Auto-Guess Parameters (ML)", on_click=run_ml_guess)
                    
                    g_Is = st.number_input("Guess $I_{s}$", format="%.2e", key='guess_Is')
                    g_n = st.number_input("Guess n", key='guess_n')
                    g_Rs = st.number_input("Guess $R_{s}$ (Ω)", key='guess_Rs')
                    
                    if fit_mode == "Multi-Temperature I-V": # allow for initial Eg guess if using multi-temp diode data
                        g_Eg = st.number_input("Guess $E_{g}$ (eV)", value=1.1)
                    else:
                        g_Eg = 1.12 # prevents valueerrors
                    
                run_btn = st.button("Run Extraction", type="primary")
                
            if run_btn: # extraction logic
                model = DiodeModel()
                extractor = ModelExtractor(model)
                
                if fit_mode == 'C-V Curve':
                    initial = {'C_j': g_Cj, 'V_bi': g_Vbi, 'm': g_m}
                    report = extractor.diode_cv_fit(df['V'].values, df['C'].values, initial_params=initial)
                    st.session_state['diode_result'] = {
                        'type': 'cv',
                        'report': report,
                        'df': df,
                        'model': model
                    }
                elif fit_mode == "Multi-Temperature I-V":
                    datasets = []
                    unique_temps = sorted(df['T'].unique())
                    for T in unique_temps:
                        sub = df[df['T'] == T].sort_values('V')
                        datasets.append((sub['V'].values, sub['I'].values, float(T)))
                    
                    initial = {'I_s': g_Is, 'n': g_n, 'R_s': g_Rs, 'Eg': g_Eg}
                    report = extractor.diode_temp_fit(datasets, initial_params=initial)
                    
                    st.session_state['diode_result'] = {
                        'type': 'multi',
                        'report': report,
                        'datasets': datasets,
                        'model': model
                    }
                    
                else:
                    initial = {'I_s': g_Is, 'n': g_n, 'R_s': g_Rs, 'Eg': g_Eg}
                    report = extractor.diode_fit(df['V'].values, df['I'].values, initial_params=initial)
                    
                    st.session_state['diode_result'] = {
                        'type': 'single',
                        'report': report,
                        'df': df,
                        'model': model
                    }
                    
        if 'diode_result' in st.session_state:
            result = st.session_state['diode_result']
            report = result['report']
            model = result['model']
            
            st.success("Extraction Converged")
            
            if result['type'] != 'cv':
                c1, c2, c3, c4 = st.columns(4)
                fit_params = report['parameters']
                c1.metric("$I_s$ (A)", f"{fit_params['I_s']:.2e}")
                c2.metric("Ideality $n$", f"{fit_params['n']:.4f}")
                c3.metric("Series $R_s$ ($\Omega$)", f"{fit_params['R_s']:.4f}")
                
                if result['type'] ==  'multi':
                    c4.metric("Bandgap $E_g$ (eV)", f"{fit_params['Eg']:.4f}")
            
            st.divider()
            st.subheader("SPICE Model")
            spice_str = generate_spice_model(report['parameters'], "diode", model_name="Diode_Model")
            st.code(spice_str, language='spice')
            
            st.download_button(
                label="Download Model File",
                data=spice_str,
                file_name="diode_model.lib",
                mime="text/plain"
            )
            
            if result['type'] == 'cv':
                df_res = result['df']
                st.pyplot(plot_diode_cv(df_res['V'].values, df_res['C'].values, model, report['parameters']))
                
                st.divider()
                st.subheader("Diode C-V Physics Analysis")
                
                tab1, tab2 = st.tabs(["Extracted Depletion Width", "Junction State Visualization"])
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Depletion Width Extraction")
                        area = st.slider("Device Area ($cm^2$)", min_value=1e-5, max_value=1e-2, value=7.096e-4, step=1e-5, format="%.2e")
                    with col2:
                        C_fit = model.compute_capacitance(df_res['V'].values, report['parameters'])
                        w = diode_dep_width_plot(df_res['V'].values, C_fit, area)
                        st.info("Depletion width $w$ increases with reverse bias voltage. $C = \\epsilon A / W$")
                        st.pyplot(w)
                    
                with tab2:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Junction State Visualization")
                        v_min = float(df_res['V'].min())
                        v_max = float(df_res['V'].max())
                        vis_v = st.slider("Bias Voltage ($V$)", min_value=v_min, max_value=v_max, value=v_min, format="%.2f", key="cv_vis")
                    with col2:
                        fig_struct, ax_struct = plt.subplots(figsize=(5, 3))
                        draw_diode_cross(ax_struct, report['parameters'], v_bias=vis_v)
                        st.pyplot(fig_struct)

            elif result['type'] == 'multi':
                datasets = result['datasets']
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.plasma(np.linspace(0, 1, len(datasets)))
                T_ref = 300.0
                fit_params = report['parameters']
                
                for idx, (v_data, i_data, T) in enumerate(datasets):
                    c = colors[idx]
                    ax.semilogy(v_data, i_data, 'o', alpha=0.4, color=c, label=f'{T}K Data')
                    Is_T = fit_params['I_s'] * (T / T_ref)**3 * np.exp(((fit_params['Eg'] * q_e) / k_B) * (1/T_ref - 1/T))
                    p_local = {'I_s': Is_T, 'n': fit_params['n'], 'R_s': fit_params['R_s']}
                    i_fit = model.compute_current(v_data, p_local, T=T)
                    ax.semilogy(v_data, i_fit, '-', color=c, label=f'{T}K Fit')
                
                ax.set_xlabel("Voltage [V]")
                ax.set_ylabel("Current [A]")
                ax.set_title("Temperature-Dependent Diode Fit")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig)
            
            else:
                df_res = result['df']
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.semilogy(df_res['V'], df_res['I'], 'o', alpha=0.5, label='Data')
                I_fit = model.compute_current(df_res['V'].values, report['parameters'])
                ax.semilogy(df_res['V'], I_fit, 'r-', label='Fit')
                ax.set_xlabel("Voltage [V]")
                ax.set_ylabel("Current [A]")
                ax.legend()
                st.pyplot(fig)
                
            if result['type'] != 'cv':
                st.divider()
                st.subheader("Diode I-V Physics Analysis")
                
                tab1, tab2 = st.tabs(["Physical Junction State", "3D Characteristics Surface"])
                
                with tab1:
                    st.markdown("### Physical Junction State")
                    v_min = 0.0
                    v_max = 1.0
                    
                    if result['type'] == 'single':
                        v_min = float(result['df']['V'].min())
                        v_max = float(result['df']['V'].max())
                    elif result['type'] == 'multi':
                        all_v = np.concatenate([d[0] for d in result['datasets']])
                        v_min = float(all_v.min())
                        v_max = float(all_v.max())
                    
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        vis_v = st.slider("Bias Voltage ($V$)", min_value=(v_min - 0.5), max_value=(v_max + 0.5), value=v_min, format="%.2f")
                        
                    with v_col2:
                        fig_struct, ax_struct = plt.subplots(figsize=(5, 3))
                        draw_diode_cross(ax_struct, report['parameters'], v_bias=vis_v)
                        st.pyplot(fig_struct)
                        
                with tab2:
                    if result['type'] == 'single':
                        v_max = float(result['df']['V'].max())
                        t_min, t_max = 280, 340
                    else:
                        v = np.concatenate([d[0] for d in result['datasets']])
                        v_max = float(v.max())
                        t = [d[2] for d in result['datasets']]
                        t_min, t_max = float(min(t)), float(max(t))
                        
                    fig = plot_3d_diode(model, report['parameters'], v_max, t_min, t_max)
                    st.plotly_chart(fig, width='stretch')
    elif app_mode == "Physics Explorer":
        st.header("Diode Physics Explorer")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Parameters")
            na = st.slider("Acceptor Doping ($N_A$) [log cm$^{-3}$]", 14.0, 18.0, 16.0, 0.1)
            nd = st.slider("Donor Doping ($N_D$) [log cm$^{-3}$]", 14.0, 18.0, 16.0, 0.1)
            t = st.slider("Temperature ($T$) [K]", 200, 400, 300, 10)
            v_bias = st.slider("Applied Bias ($V$) [V]", -2.0, 1.0, 0.0, 0.05)
            
            Na = 10**na
            Nd = 10**nd
            phys = DiodePhysics(Na, Nd, t)
            v_bi = phys.get_bi_potential()
            w, xp, xn = phys.get_dep_width(v_bias)
            
            st.divider()
            st.subheader("Key Metrics")
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Built-in Potential ($V_{bi}$)", f"{v_bi:.3f} V")
                st.latex(r"V_{bi} = V_T \ln\left(\frac{N_A N_D}{n_i^2}\right)")
                st.metric("Thermal Voltage ($V_T$)", f"{phys.Vt*1e3:.1f} mV")
                st.latex(r"V_T = \frac{k_B T}{q}")
                st.metric("Energy Bandgap ($E_g$)", f"{phys.Eg:.3f} eV")
                st.latex(r"E_g(T) = 1.17 - \frac{4.73 \cdot 10^{-4} T^2}{T + 636}")
            
            with c2:   
                st.metric("Depletion Width ($W$)", f"{w*1e4:.3f} μm")
                st.latex(r"W = \sqrt{\frac{2 \epsilon_s}{q} \left(\frac{1}{N_A} + \frac{1}{N_D}\right) (V_{bi} - V_{bias})}")
                st.metric("Intrinsic Concentration ($n_i$)", f"{phys.ni:.2e} cm$^{{-3}}$")
                st.latex(r"n_i = \sqrt{N_c N_v} e^{-E_g / 2k_B T}")
            
        with col2:
            st.subheader("Energy Band Diagram")
            
            if v_bias > 0:
                st.info("**Forward Bias**: the applied voltage opposes the built-in potential, lowering the energy barrier. This allows for majority carriers to diffuse across the junction, resulting in significant current flow ($I \propto e^{V/V_T}$).")
            elif v_bias < 0:
                st.warning("**Reverse Bias**: the applied voltage adds to the built-in potential, raising the energy barrier. The depletion region widens and only a small leakage current flows due to drift of minority carriers.")
            else:
                st.success("**Equilibrium**: no net current flows, the Fermi level ($E_F$) is constant throughout the device.")
            
            limit = max(w * 2, 1e-4)
            x_grid = np.linspace(-limit, limit, 500)
            band = phys.compute_energy_bands(v_bias, x_grid)
            fig = plot_diode_bands(band)
            st.pyplot(fig)

elif device_type == "MOSFET": # MOSFET logic
    if app_mode == "Extraction":
        st.header("MOSFET Extraction")
        col1, col2 = st.columns([1, 2])
        with col1:
            source = st.radio("Data Source", ["Synthetic", "Upload CSV"])
            df = None
            
            if source == "Synthetic":
                true_Vth = st.number_input("True $V_{th}$ (V)", value=0.7)
                true_kn = st.number_input("True $k_{n}$", value=1e-3, format="%.2e")
                true_lam = st.number_input("True $\lambda$", value=0.02)
                synth_type = st.radio("Synthetic Data Type", ["Transfer Curve ($I_{d}-V_{gs}$)", "Output Family ($I_{d}-V_{ds}$)"])
                
                if synth_type == "Transfer Curve ($I_{d}-V_{gs}$)":
                    df, model = generate_synthetic_mosfet(true_Vth, true_kn, true_lam)
                else:
                    vgs_input = st.text_input("Family $V_{gs}$ values (comma-separated)", value="1.5, 2.0, 2.5, 3.0, 3.5, 4.0")
                    try:
                        vgs_list = [float(x.strip()) for x in vgs_input.split(',')]
                    except ValueError:
                        st.error("Invalid Vgs list format")
                        vgs_list = [1.5, 2.0, 2.5, 3.0]
                    
                    df, model = generate_synthetic_mosfet_family(true_Vth, true_kn, true_lam, V_gs=vgs_list)
                
            else:
                csv = st.file_uploader("Upload CSV", type=['csv'], key="csv_mosfet")
                if csv:
                    df = pd.read_csv(csv)
                    cols = df.columns.tolist()
                    
                    vg_idx = 0
                    vd_idx = 1 if len(cols) > 1 else 0
                    id_idx = 2 if len(cols) > 2 else 0
                    
                    vg_col = st.selectbox("$V_{gs}$ Column", cols, index=vg_idx)
                    vd_col = st.selectbox("$V_{ds}$ Column", cols, index=vd_idx)
                    id_col = st.selectbox("$I_{d}$ Column", cols, index=id_idx)
                    
                    df = df.rename(columns={vg_col: 'V_gs', id_col: 'I_d', vd_col: 'V_ds'})
                    
        if df is not None:
            st.subheader("Configuration")
            fit_mode = st.radio("Fit Mode", ["Single Curve", "Multi-Curve"])
            
            with col1:
                st.divider()
                st.markdown("**Initial Guesses**")
                
                if 'guess_Vth' not in st.session_state: st.session_state['guess_Vth'] = 0.5
                if 'guess_kn' not in st.session_state: st.session_state['guess_kn'] = 1e-4
                
                if source != "Synthetic":
                    def run_mos_ml_guess():
                        temp_model = MOSFETModel()
                        temp_ext = ModelExtractor(temp_model)
                        pred = None
                        
                        if fit_mode == 'Multi-Curve':
                            try:
                                unique_vgs = sorted(df['V_gs'].unique())
                                preds_vth, preds_kn, preds_lam = [], [], []
                                
                                for vgs in unique_vgs:
                                    sub = df[df['V_gs'] == vgs].sort_values('V_ds')
                                    if len(sub) < 10: continue
                                    p = temp_ext._get_mosfet_output_ml_guess(sub['V_ds'].values, sub['I_d'].values, float(vgs))
                                    
                                    if p:
                                        preds_vth.append(p['V_th'])
                                        preds_kn.append(p['k_n'])
                                        preds_lam.append(p['lam'])
                                
                                if preds_vth:
                                    pred = {
                                        'V_th': np.median(preds_vth),
                                        'k_n': np.median(preds_kn),
                                        'lam': np.median(preds_lam)
                                    }
                            except Exception as e:
                                print(f"Multi-curve guess error: {e}")
                            
                        else:
                            if sweep_type == "$I_{d}-V_{gs}$ (Transfer)":
                                sub = df[df['V_ds'] == sel_param].sort_values('V_gs')
                                pred = temp_ext._get_mosfet_transfer_ml_guess(sub['V_gs'].values, sub['I_d'].values, float(sel_param))
                            else:
                                sub = df[df['V_gs'] == sel_param].sort_values('V_ds')
                                pred = temp_ext._get_mosfet_output_ml_guess(sub['V_ds'].values, sub['I_d'].values, float(sel_param))
                                
                        if pred:
                            st.session_state['guess_Vth'] = float(pred['V_th'])
                            st.session_state['guess_kn'] = float(pred['k_n'])
                            if 'lam' in pred:
                                st.session_state['guess_lam'] = float(pred['lam'])
        
                            st.toast("ML Prediction Applied")
                        else:
                            st.toast("ML Prediction Failed")
                            
                    st.button("Auto-Guess Parameters (ML)", on_click=run_mos_ml_guess)
                    
                g_Vth = st.number_input("Guess $V_{th}$ (V)", key='guess_Vth')
                g_kn = st.number_input("Guess $k_{n}$", format="%.2e", key='guess_kn')
                g_lam = st.number_input("Guess $\lambda$", format="%.4f", step=0.01, key='guess_lam')

            if fit_mode == "Multi-Curve": # detect how many unique family curves exist within CSV
                st.info(f"Detected {df['V_gs'].nunique()} unique $V_{{gs}}$ curves for global fit.")
                
                if st.button("Run Global Extraction", type="primary"):
                    model = MOSFETModel()
                    extractor = ModelExtractor(model)
                    initial = {'V_th': st.session_state['guess_Vth'], 'k_n': st.session_state['guess_kn'], 'lam': 0.0}
                    
                    datasets = []
                    unique_vgs = sorted(df['V_gs'].unique())
                    for vgs in unique_vgs:
                        sub = df[df['V_gs'] == vgs].sort_values('V_ds')
                        datasets.append((sub['V_ds'].values, sub['I_d'].values, float(vgs)))
                    
                    report = extractor.multi_mosfet_fit(datasets, initial_params=initial)
                    
                    st.session_state['mosfet_result'] = {
                        'type': 'multi',
                        'report': report,
                        'datasets': datasets,
                        'model': model
                    }
            else: # Single Curve Mode
                sweep_type = st.radio("Sweep Type", ["$I_{d}-V_{gs}$ (Transfer)", "$I_{d}-V_{ds}$ (Output)"])
                
                subset = None
                sel_param = 0.0
                
                if sweep_type == "$I_{d}-V_{gs}$ (Transfer)":
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
                    
                    if sweep_type == "$I_{d}-V_{gs}$ (Transfer)":
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
                    
                    st.session_state['mosfet_result'] = {
                        'type': 'single',
                        'report': report,
                        'subset': subset,
                        'sweep_type': sweep_type,
                        'v_ds_arg': v_ds_arg,
                        'v_gs_arg': v_gs_arg,
                        'sel_param': sel_param,
                        'model': model
                    }

        # Render results from session state
        if 'mosfet_result' in st.session_state:
            result = st.session_state['mosfet_result']
            report = result['report']
            model = MOSFETModel()

            st.success("Extraction Converged")
            m1, m2, m3 = st.columns(3)
            m1.metric('V_th', f"{report['parameters']['V_th']:.4f} V")
            m2.metric('k_n', f"{report['parameters']['k_n']:.2e}")
            m3.metric('lambda', f"{report['parameters']['lam']:.4f}")
            
            st.divider()
            st.subheader("SPICE Model")
            spice_str = generate_spice_model(report['parameters'], "MOSFET", model_name="MOS_Model")
            st.code(spice_str, language='spice')
            
            st.download_button(
                label="Download Model File",
                data=spice_str,
                file_name="mosfet_model.lib",
                mime="text/plain"
            )
            
            if result['type'] == 'multi':
                fig, ax = plt.subplots(figsize=(10, 6))
                datasets = result['datasets']
                colors = plt.cm.jet(np.linspace(0, 1, len(datasets)))
                
                for idx, (vds_data, id_data, vgs_val) in enumerate(datasets):
                    c = colors[idx]
                    ax.plot(vds_data, id_data, 'o', alpha=0.4, color=c, label=f'Data {vgs_val}V')
                    
                    p = report['parameters'].copy()
                    p['V_ds'] = vds_data
                    vgs_arr = np.full_like(vds_data, vgs_val)
                    I_fit = model.compute_current(vgs_arr, p)
                    
                    ax.plot(vds_data, I_fit, '-', color=c, label=f'Fit {vgs_val}V')
                
                ax.set_xlabel("$V_{ds}$ [V]")
                ax.set_ylabel("$I_{d}$ [A]")
                ax.set_title("Global Fit: Output Characteristics")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig)
                
            elif result['type'] == 'single':
                fig, ax = plt.subplots(figsize=(10, 5))
                subset = result['subset']
                sweep_type = result['sweep_type']
                v_ds_arg = result['v_ds_arg']
                v_gs_arg = result['v_gs_arg']
                
                if sweep_type == "$I_{d}-V_{gs}$ (Transfer)":
                    x_data = subset['V_gs']
                    x_label = "$V_{gs}$ [V]"
                else:
                    x_data = subset['V_ds']
                    x_label = "$V_{ds}$ [V]"
                    
                ax.plot(x_data, subset['I_d'], 'o', alpha=0.5, label='Data')
                
                fit_params = report['parameters']
                fit_params['V_ds'] = v_ds_arg
                I_fit = model.compute_current(v_gs_arg, fit_params)
                
                ax.plot(x_data, I_fit, 'r-', label='Fit')
                ax.set_xlabel(x_label)
                ax.set_ylabel('$I_{d}$ [A]')
                ax.legend()
                st.pyplot(fig)

            st.divider()
            st.subheader("MOSFET Physics & Characteristics")
            
            tab1, tab2 = st.tabs(["Physical Channel State", "3D Characteristics Surface"])
            
            with tab1:
                st.markdown("### Physical Channel State")
                
                if df is not None:
                    vgs_min = float(df['V_gs'].min())
                    vgs_max = float(df['V_gs'].max())
                    vds_min = float(df['V_ds'].min())
                    vds_max = float(df['V_ds'].max())
                else:
                    vgs_min = 0.0
                    vgs_max = 5.0
                    vds_min = 0.0
                    vds_max = 5.0
                
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    vis_vgs = st.slider("Gate-to-Source Voltage ($V_{gs}$)", min_value=(vgs_min - 0.5), max_value=(vgs_max + 0.5), value=vgs_min, format="%.2f")
                    vis_vds = st.slider("Drain-to-Source Voltage ($V_{ds}$)", min_value=(vds_min - 0.5), max_value=(vds_max + 0.5), value=vds_min, format="%.2f")
                    
                with v_col2:
                    fig_struct, ax_struct = plt.subplots(figsize=(5, 3))
                    draw_mosfet_cross(ax_struct, report['parameters'], vgs=vis_vgs, vds=vis_vds)
                    st.pyplot(fig_struct)
                    
            with tab2:
                vgs_max = df['V_gs'].max() if df is not None else 5.0
                vds_max = df['V_ds'].max() if df is not None else 5.0
                fig = plot_3d_fet_surface(model, report['parameters'], vgs_max, vds_max)
                st.plotly_chart(fig, width='stretch')
    elif app_mode == "Physics Explorer":
        st.header("MOSFET Physics Explorer")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Parameters")
            na = st.slider("Substrate Doping ($N_A$) [log cm$^{-3}$]", 14.0, 18.0, 16.0, 0.1)
            tox_nm = st.slider("Oxide Thickness ($t_{ox}$) [nm]", 1.0, 100.0, 10.0, 0.5)
            t = st.slider("Temperature ($T$) [K]", 200, 400, 300, 10)
            vgs = st.slider("Gate Voltage ($V_{gs}$) [V]", -2.0, 5.0, 0.0, 0.05)
            
            Na = 10**na
            tox = tox_nm * 1e-7
            phys = MOSFETPhysics(Na, tox, t)
            phi_s = phys.solve_surface_potential(vgs)
            
            st.divider()
            st.subheader("Key Metrics")
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.metric("Threshold Voltage ($V_{th}$)", f"{phys.Vth:.3f} V")
                st.latex(r"V_{th} = V_{fb} + 2\phi_f + \gamma\sqrt{2\phi_f}\\V_{fb, Si} = -0.9V")
                st.metric("Surface Potential ($\phi_s$)", f"{phi_s:.3f} V")
                st.latex(r"\phi_s(Inv) \approx 2\phi_f")
                st.metric("Oxide Capacitance ($C_{ox}$)", f"{phys.cox*1e6:.3f} μF/cm²")
                st.latex(r"C_{ox} = \frac{\epsilon_{ox}}{t_{ox}}")
                
            with c2:
                st.metric("Fermi Potential ($\phi_f$)", f"{phys.phi_f:.3f} V")
                st.latex(r"\phi_f = V_T \ln\left(\frac{N_A}{n_i}\right)")
                st.metric("Body Effect ($\gamma$)", f"{phys.gamma:.3f} sqrt(V)")
                st.latex(r"\gamma = \frac{\sqrt{2 q \epsilon_{si} N_A}}{C_{ox}}")
                
        with col2:
            st.subheader("MOS Band Diagram")
            
            if vgs < -0.9:
                st.info("**Accumulation**: when $V_{gs} < V_{fb}$, majority carriers are attracted to the oxide-semiconductor interface, causing the valence bend to bend upward closer to the Fermi level.")
            elif abs(phi_s) < 0.01:
                st.info("**Flatband**: The gate voltage compensates for the work function difference, leading to the energy bands being flats and there being no net charge in the semiconductor.")
            elif phi_s < 2 * phys.phi_f:
                st.warning("**Depletion**: $V_{gs} > V_{fb}$, so majority carriers are repeled from the surface and leave behind immobile negative acceptor ions. The bands bend downward, depleting the surface of mobile carriers. Even though eventually an electron inversion layer appears, we are still not in an inversion state yet and instead are in 'weak inversion'.")
            else:
                st.success("**Inversion**: $V_{gs} > V_{th}$, so the bands bend significantly downward such that the intrinsic level $E_{i}$ crosses the Fermi level, causing the minority carriers to gather at the surface and form a conductive n-channel.")
                
            x_grid = np.linspace(0, 1e-4, 500)
            band = phys.compute_band_diagrams(vgs, x_grid)
            fig = plot_mos_bands(band)
            st.pyplot(fig)