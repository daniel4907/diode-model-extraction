import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
from scipy.constants import k as k_B, e as q_e

def plot_diode_fit(V_data, I_data, model, fitted_params, filename=None, temps=None):
    """
    Generates a comparison of the I-V data and the I-V curve from the fitted parameters

    Args:
        V_data (scalar/numpy array): voltage from I-V data
        I_data (scalar/numpy array): current from I-V data
        model: instance of device model class
        fitted_params (dict): dict containing fitted parameters from least squares algorithm
        filename (optional): filename for saving the error plot locally, defaults to None
        temps (optional): array of temperatures for multi-temp diode curves, defaults to None
    """ 
    if temps is None:
        plt.semilogy(V_data, I_data, label='Original data')
        I_fit = model.compute_current(V_data, fitted_params)
        plt.semilogy(V_data, I_fit, label='Fitted data')
        plt.legend()
        plt.xlabel("Voltage [V]")
        plt.ylabel("Current [A]")
    
        if filename is not None:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.show()
    else:
        for i, T in enumerate(temps):
            local_params = fitted_params.copy()
            if 'Eg' in fitted_params:
                local_params['I_s'] = model.compute_sat_current(fitted_params['I_s'], fitted_params['Eg'], T)
            V_i = V_data[i] if isinstance(V_data, list) else V_data
            I_i = I_data[i] if isinstance(I_data, list) else I_data
            I_fit = model.compute_current(V_i, local_params, T=T)
            plt.figure()
            plt.semilogy(V_i, I_i, label=f"Data {T}K")
            plt.semilogy(V_i, I_fit, label=f"Fit {T}K")
            plt.title(f"Diode I-V curve at {T} K")
            plt.legend()
            
            if filename is not None:
                plt.savefig(f"{filename}_{T}K", dpi=300, bbox_inches='tight')
            
        plt.show()
    
def diode_error_plot(V_data, I_data, model, fitted_params, filename=None, temps=None):
    """
    Generates a relative error plot for each voltage point using the I-V data and the fitted parameters

    Args:
        V_data (scalar/numpy array): voltage from I-V data
        I_data (scalar/numpy array): current from I-V data
        model: instance of device model class
        fitted_params (dict): dict containing fitted parameters from least squares algorithm
        filename (optional): filename for saving the error plot locally, defaults to None
        temps (optional): array of temperatures for multi-temp diode curves, defaults to None
    """
    if temps is None:
        I_fit = model.compute_current(V_data, fitted_params)
        err = (I_fit - I_data) / np.maximum(np.abs(I_data), 1e-15)
        plt.figure()
        plt.plot(V_data, err, label='Relative error')
        plt.legend()
        plt.xlabel("Voltage [V]")
        plt.ylabel("Relative error")
    
        if filename is not None:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.show()
    else:
        for i, T in enumerate(temps):
            local_params = fitted_params.copy()
            if 'Eg' in fitted_params:
                local_params['I_s'] = model.compute_sat_current(fitted_params['I_s'], fitted_params['Eg'], T)
            V_i = V_data[i] if isinstance(V_data, list) else V_data
            I_i = I_data[i] if isinstance(I_data, list) else I_data
            I_fit = model.compute_current(V_i, local_params, T=T)
            err = (I_fit - I_i) / np.maximum(np.abs(I_i), 1e-15)
            
            plt.figure()
            plt.plot(V_i, err, label=f'Relative error {T}K')
            plt.legend()
            plt.xlabel("Voltage [V]")
            plt.ylabel("Relative error")
            plt.title(f"Diode fit relative error at {T} K")
            
            if filename is not None:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
        
            plt.show()
            
def diode_sat_current_plot(temps, model, fitted_params, filename=None):
    """
    Plots the saturation current as a function of temperature

    Args:
        temps (list): temperature list
        model: instance of device model class
        fitted_params (dict): dict containing fitted parameters from least squares algorithm
        filename (optional): filename for saving the error plot locally, defaults to None
    """
    Is = fitted_params['I_s']
    Eg = fitted_params['Eg']
    Is_vals = [model.compute_sat_current(Is, Eg, T) for T in temps]
    
    plt.figure()
    plt.semilogy(temps, Is_vals, label='Fitted $I_s(T)$')
    plt.xlabel("Temperature [K]")
    plt.ylabel("Saturation current [A]")
    plt.legend()
    
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    plt.show()
    
def draw_diode_cross(ax, params, v_bias=0.0):
    """
    Draw cross-section of diode
    """
    ax.clear()
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-1.0, 2.0)
    ax.axis('off')
    ax.set_title(f"PN Junction State ($V_{{bias}}={v_bias}$)")
    
    v_bi = 0.7
    if v_bias >= v_bi:
        w_dep = 0.05
    else:
        w_dep = 0.3 * np.sqrt(max(0, v_bi - v_bias))
        
    p_width = 1.5 - w_dep
    n_start = 1.5 + w_dep
    n_width = 1.5 - w_dep
    p_region = patches.Rectangle((0, -0.5), p_width, 1.0, edgecolor='black', facecolor='#ff9999')
    n_region = patches.Rectangle((n_start, -0.5), n_width, 1.0, edgecolor='black', facecolor='#99ccff')
    dep_region = patches.Rectangle((p_width, -0.5), 2 * w_dep, 1.0, edgecolor='black', facecolor='gray', hatch='///')
    anode = patches.Rectangle((-0.1, -0.3), 0.1, 0.6, edgecolor='black', facecolor='gray')
    cathode = patches.Rectangle((3.0, -0.3), 0.1, 0.6, edgecolor='black', facecolor='gray')
    ax.add_patch(p_region)
    ax.add_patch(n_region)
    ax.add_patch(dep_region)
    ax.add_patch(anode)
    ax.add_patch(cathode)
    ax.text(0.75, 0, 'P-Type\n(Anode)', ha='center', fontsize=8, fontweight='bold')
    ax.text(2.25, 0, 'N-Type\n(Cathode)', ha='center', fontsize=8, fontweight='bold')
    ax.text(1.5, 0.8, 'Depletion Region', ha='center', fontsize=8, fontweight='bold')
    
    T_ref = 300
    Vt = (k_B * T_ref) / q_e
        
    if v_bias > 5 * Vt:
        status = "Forward Bias"
        color = 'green'
    elif v_bias < -5 * Vt:
        status = "Reverse Bias"
        color = 'red'
    else:
        status = "Equilibrium"
        color = 'blue'
        
    ax.text(1.5, -1.2, status, ha='center', color=color, fontsize=10, fontweight='bold')
    
def plot_3d_diode(model, params, v_max=1.0, t_min=280, t_max=340):
    """
    Plots 3D surface plot of diode current against voltage and temperature
    """
    fig = plt.figure(figsize=(10, 7))
    v = np.linspace(0, v_max, 30)
    t = np.linspace(t_min, t_max, 30)
    V, T = np.meshgrid(v, t)
    I = np.zeros_like(V)
    
    for i in range(V.shape[0]):
        for j in range(T.shape[0]):
            local_params = params.copy()
            
            if 'Eg' in params:
                local_params['I_s'] = model.compute_sat_current(params['I_s'], params['Eg'], T[i, j])
            
            I[i, j] = model.compute_current([V[i, j]], local_params, T=T[i,j])[0]

    fig = go.Figure(data=[go.Surface(z=I, x=v, y=t, colorscale='Viridis')])
    fig.update_layout(
        title='3D Characteristics Surface',
        scene=dict(
            xaxis_title='Voltage [V]',
            yaxis_title='Temperature [K]',
            zaxis_title='Current [A]'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def plot_mosfet_fit(V_gs, I_data, model, fitted_params, filename=None, yscale='linear'):
    """
    Generates a comparison of the Id-Vgs data and the Id-Vgs curve from the fitted parameters

    Args:
        V_gs (scalar/numpy array): gate-to-source voltage
        I_data (scalar/numpy array): curret from Id-Vgs data
        model: instance of device model class
        fitted_params (dict): dict containing fitted parameters from least squares algorithm
        filename (str, optional): filename for saving the error plot locally, defaults to None
        yscale (str, optional): 'linear' or 'log', defaults to 'linear'
    """
    I_fit = model.compute_current(V_gs, fitted_params)
    
    if yscale == 'log':
        plt.semilogy(V_gs, I_data, label='Original data')
        plt.semilogy(V_gs, I_fit, label='Fitted data')
        plt.title('$I_{d}-V_{gs}$ curve on semilog scale')
    else:
        plt.plot(V_gs, I_data, label='Original_data')
        plt.plot(V_gs, I_fit, label='Fitted data')
        plt.title('$I_{d}-V_{gs}$ curve on linear scale')
    
    plt.xlabel("$V_{gs}$ [V]")
    plt.ylabel("$I_{d}$ [A]")
    plt.legend()
    
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    plt.show()
    
def plot_mosfet_multi(V_ds, I_data, model, fitted_params, V_gs_label=None, filename=None):
    """
    Generates a comparison of the Id-Vds data and the Id-Vds curves from the fitted parameters

    Args:
        V_ds (scalar/numpy array): drain-to-source voltage
        I_data (scalar/numpy array): curret from Id-Vgs data
        model: instance of device model class
        fitted_params (dict): dict containing fitted parameters from least squares algorithm
        V_gs_label (str, optional): label for title of plots
        filename (str, optional): filename for saving the error plot locally, defaults to None
    """
    I_fit = model.compute_current(fitted_params['vgs_array'], {**fitted_params, 'V_ds': V_ds})
    
    plt.semilogy(V_ds, I_data, label='Original data')
    plt.semilogy(V_ds, I_fit, label='Fitted data')
    
    if V_gs_label is not None:
        plt.title(f"$I_{{d}}$-$V_{{ds}}$ at $V_{{gs}}$ = {V_gs_label} V")
    
    plt.xlabel("$V_{ds}$ [V]")
    plt.ylabel("$I_{d}$ [A]")
    plt.legend()
    
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    plt.show()
    
def draw_mosfet_cross(ax, params, vgs=0, vds=0):
    """
    Draw cross-section of MOSFET
    """
    ax.clear()
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-1.0, 2.0)
    ax.axis('off')
    ax.set_title(f"NMOS State ($V_{{gs}}$={vgs} V, $V_{{ds}}$={vds} V)") # double curly brackets so python doesn't get confused with f-strings
    
    body = patches.Rectangle((0, -0.8), 3, 0.8, linewidth=1, edgecolor='black', facecolor='#e0e0e0')
    source = patches.Rectangle((0.2, -0.2), 0.6, 0.4, edgecolor='black', facecolor='#4da6ff')
    drain = patches.Rectangle((2.2, -0.2), 0.6, 0.4, edgecolor='black', facecolor='#4da6ff')
    oxide = patches.Rectangle((0.8, 0), 1.4, 0.1, edgecolor='black', facecolor='#ffff99')
    gate = patches.Rectangle((0.8, 0.1), 1.4, 0.3, edgecolor='black', facecolor='#ff6666')
    ax.add_patch(body)
    ax.add_patch(source)
    ax.add_patch(drain)
    ax.add_patch(oxide)
    ax.add_patch(gate)
    ax.text(1.5, -0.4, 'p-Si Substrate', ha='center', fontsize=8)
    ax.text(0.5, -0.1, 'Source', ha='center', color='white', fontsize=8, fontweight='bold')
    ax.text(2.5, -0.1, 'Drain', ha='center', color='white', fontsize=8, fontweight='bold')
    ax.text(1.5, 0.25, 'Gate', ha='center', color='white', fontsize=8, fontweight='bold')
    
    vth = params.get('V_th', 0.7)
    if vgs > vth: # if there is a channel
        channel_color = '#4da6ff'
        depth_source = 0.15
        
        if vds >= (vgs - vth): # saturation, find pinch off
            depth_drain = 0.01
            channel_poly = patches.Polygon([(0.8, 0), (2.2, 0), (2.2, -depth_drain), (0.8, -depth_source)], facecolor=channel_color, alpha=0.8)
            ax.text(1.5, -1.2, 'Saturation', ha='center', color='green', fontsize=10)
        else: # continuous channel
            r = 1 - (vds / (vgs - vth + 0.1))
            depth_drain = max(0.05, depth_source * r)
            channel_poly = patches.Polygon([(0.8, 0), (2.2, 0), (2.2, -depth_drain), (0.8, -depth_source)], facecolor=channel_color, alpha = 0.8)
            ax.text(1.5, -1.2, 'Triode/Linear', ha='center', color='green', fontsize=10)
            
        ax.add_patch(channel_poly)
    else:
        ax.text(1.5, -1.2, 'Cutoff', ha='center', color='green', fontsize=10)
        
def plot_3d_fet_surface(model, params, vgs_max=5.0, vds_max=5.0):
    """
    Plots 3D surface plot of drain current against gate-to-source and drain-to-source current
    """
    fig = plt.figure(figsize=(10, 7))
    vgs = np.linspace(0, vgs_max, 30)
    vds = np.linspace(0, vds_max, 30)
    Vgs, Vds = np.meshgrid(vgs, vds)
    Id = np.zeros_like(Vgs)
    
    for i in range(Vgs.shape[0]):
        for j in range(Vds.shape[0]):
            local_params = params.copy()
            local_params['V_ds'] = Vds[i, j]
            Id[i, j] = model.compute_current([Vgs[i, j]], local_params)[0]
    
    fig = go.Figure(data=[go.Surface(z=Id, x=vgs, y=vds, colorscale='Viridis')])
    fig.update_layout(
        title='3D Characteristics Surface',
        scene=dict(
            xaxis_title='V<sub>gs</sub> [V]', # need to use html formatting for plotly
            yaxis_title='V<sub>ds</sub> [V]',
            zaxis_title='I<sub>d</sub> [A]'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig
