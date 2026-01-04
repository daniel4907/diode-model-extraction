import numpy as np
import matplotlib.pyplot as plt

def plot_fit(V_data, I_data, model, fitted_params, filename=None):
    """
    Generates a comparison of the I-V data and the I-V curve from the fitted parameters

    Args:
        V_data (scalar/numpy array): voltage from I-V data
        I_data (scalar/numpy array): current from I-V data
        model: instance of device model class
        fitted_params (dict): dict containing fitted parameters from least squares algorithm
        filename (optional): filename for saving the error plot locally, defaults to None
    """
    plt.semilogy(V_data, I_data, label='Original data')
    I_fit = model.compute_current(V_data, fitted_params)
    plt.semilogy(V_data, I_fit, label='Fitted data')
    plt.legend()
    plt.xlabel("Voltage [V]")
    plt.ylabel("Current [A]")
    
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    plt.show()
    
def error_plot(V_data, I_data, model, fitted_params, filename=None):
    """
    Generates a relative error plot for each voltage point using the I-V data and the fitted parameters

    Args:
        V_data (scalar/numpy array): voltage from I-V data
        I_data (scalar/numpy array): current from I-V data
        model: instance of device model class
        fitted_params (dict): dict containing fitted parameters from least squares algorithm
        filename (optional): filename for saving the error plot locally, defaults to None
    """
    I_fit = model.compute_current(V_data, fitted_params)
    err = (I_fit - I_data) / np.maximum(np.abs(I_data), 1e-15)
    plt.plot(V_data, err, label='Relative error')
    plt.legend()
    plt.xlabel("Voltage [V]")
    plt.ylabel("Relative error")
    
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
    plt.show()