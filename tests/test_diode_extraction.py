import numpy as np
from src.models import DiodeModel
from src.extraction import ModelExtractor
from src.visualization import plot_fit, error_plot

model = DiodeModel()
extractor = ModelExtractor(model)
params = {'I_s': 1e-10, 'n': 1.5}
V_data = np.linspace(0, 0.8, 50)
I_data_true = model.compute_current(V_data, params)

I_data = I_data_true + (I_data_true * 0.05) # add noise
initial_guess = {'I_s': 1e-6, 'n': 1.67}
ls = extractor.diode_fit(V_data, I_data)

assert(np.abs((ls['parameters']['I_s'] - params['I_s']) / params['I_s']) <= 0.05)
assert(np.abs((ls['parameters']['n'] - params['n']) / params['n']) <= 0.05)
assert((ls['success'] is True))

plot_fit(V_data, I_data, model, ls['parameters'], filename="diode_fit.png")
error_plot(V_data, I_data, model, ls['parameters'], filename="error_plot.png")

print("True params:", params)
print("Fit params:", ls['parameters'])
print("RMS error:", ls['rms_err'])
print("Max error:", ls['max_err'])