import numpy as np
from scipy.constants import k as k_B, e as q_e, epsilon_0

class DiodePhysics:
    def __init__(self, Na, Nd, T):
        """
        DiodePhysics class constructor
        """
        self.Na = Na
        self.Nd = Nd
        self.T = T
        
        self.Nc = 2.8e19 * (T / 300)**1.5
        self.Nv = 1.04e19 * (T / 300)**1.5
        self.Eg = 1.17 - (4.73e-4 * T**2)/(636 + T)
        self.ni = np.sqrt(self.Nc * self.Nv) * np.exp((-self.Eg * q_e) / (2 * k_B * T))
        self.eps_si = 11.7 * (epsilon_0 / 100)
        self.Vt = (k_B * T) / q_e
        
    def get_bi_potential(self):
        """
        Calculates the built in potential of the device
        """
        v_bi = self.Vt * np.log((self.Na * self.Nd) / self.ni**2)
        return v_bi
    
    def get_dep_width(self, v_bias):
        """
        Calculates the total depletion width as well as the n and p-side widths
        """
        term = np.maximum(0, self.get_bi_potential() - v_bias)
        w = np.sqrt((2 * self.eps_si / q_e) * (1/self.Na + 1/self.Nd) * (term))
        xp = w * (self.Nd / (self.Na + self.Nd))
        xn = w * (self.Na / (self.Na + self.Nd))
        return w, xp, xn
    
    def compute_junction_cap(self, v_bias, area=7.096e-4): # 7.096e-4 cm^2 is the ECE444 wafer area
        """
        Calculates the junction capacitance of the device
        """
        w, _, _ = self.get_dep_width(v_bias)
        w_eff = np.maximum(w, 1e-12)
        c = self.eps_si * area / w_eff
        return c
        
    def compute_energy_bands(self, v_bias, x_grid):
        """
        Calculates the Ec and Ev energy band levels and the quasi-Fermi levels across a bunch of points

        Args:
            v_bias (float): bias voltage applied to device
            x_grid (numpy array): array of spatial coordinates 

        Returns:
            dict: dictionary containing values for Ec and Ev
        """
        w, xp, xn = self.get_dep_width(v_bias)
        v_bi = self.get_bi_potential()
        Ec = np.zeros_like(x_grid)
        Ev = np.zeros_like(x_grid)
        K = q_e / (2 * self.eps_si)
        Ev_bulk = -self.Vt * np.log(self.Nv / self.Na)
        Ec_bulk = Ev_bulk + self.Eg
        Efp = np.zeros_like(x_grid) # use as reference level
        Efn = np.zeros_like(x_grid)
        L_diff = 2 * w
        
        for i, x in enumerate(x_grid):
            if x < -xp:
                phi = 0
            elif -xp <= x < 0:
                phi = K * self.Na * (x + xp)**2
            elif 0 <= x < xn:
                phi = (v_bi - v_bias) - K * self.Nd * (xn - x)**2
            elif x >= xn:
                phi = v_bi - v_bias
                
            Ec[i] = Ec_bulk - phi
            Ev[i] = Ec[i] - self.Eg
            
            if x < -xp:
                Efp[i] = 0.0
                dist = x - (-xp)
                Efn[i] = v_bias * np.exp(dist / L_diff)
            elif x > xn:
                Efn[i] = v_bias
                dist = x - xn
                Efp[i] = v_bias * (1 - np.exp(-dist / L_diff))
            else:
                Efp[i] = 0.0
                Efn[i] = v_bias
            
        Ei = (Ec + Ev) / 2
        
        return {
            'x': x_grid,
            'Ec': Ec,
            'Ev': Ev,
            'Ei': Ei,
            'Efp': Efp,
            'Efn': Efn,
            'v_bias': v_bias
        }

class MOSFETPhysics:
    def __init__(self, Na, tox, T):
        """
        MOSFETPhysics class constructor
        """
        self.Na = Na
        self.tox = tox
        self.T = T
        
        self.Nc = 2.8e19 * (T / 300)**1.5
        self.Nv = 1.04e19 * (T / 300)**1.5
        self.Eg = 1.17 - (4.73e-4 * T**2)/(636 + T)
        self.ni = np.sqrt(self.Nc * self.Nv) * np.exp((-self.Eg * q_e) / (2 * k_B * T))
        self.cox = (epsilon_0 / 100) * 3.9 / tox
        self.Vt = (k_B * T) / q_e
        self.gamma = np.sqrt(2 * q_e * 11.7 * (epsilon_0 / 100) * Na) / self.cox
        self.phi_f = self.Vt * np.log(self.Na / self.ni)
        self.Vth = -0.9 + 2*self.phi_f + self.gamma * np.sqrt(2 * self.phi_f)
        
    def solve_surface_potential(self, Vgs):
        """
        Calculates the surface potential of the device
        """
        V_eff = Vgs + 0.9
        phi_s = 0.0
        
        if V_eff < 0: # accumulation
            phi_s = V_eff
        elif V_eff < (self.Vth + 0.9):
            phi_s = ((-self.gamma + np.sqrt(self.gamma**2 + 4 * V_eff))/2)**2
            if phi_s > 2 * self.phi_f:
                phi_s = 2 * self.phi_f
        else: # strong inversion
            phi_s = 2 * self.phi_f + (V_eff - self.Vth) * 0.05
            
        return phi_s
    
    def compute_band_diagrams(self, Vgs, x_grid):
        """
        Computes the energy band diagram for a MOS capacitor

        Args:
            Vgs (scalar): gate-to-source voltage
            x_grid (numpy array): array of spatial coordinates 

        Returns:
            dict: dictionary containing values for Ec and Ev alongside other values
        """
        phi_s = self.solve_surface_potential(Vgs)
        
        if phi_s > 0:
            w = np.sqrt(2 * 11.7 * (epsilon_0 / 100) * phi_s / (q_e * self.Na))
        else:
            w = 0.0
        
        L_debye = np.sqrt(11.7 * (epsilon_0 / 100) * self.Vt / (q_e * self.Na))
        phi = np.zeros_like(x_grid)
        
        for i, x in enumerate(x_grid):
            if x < 0:
                phi[i] = phi_s
            elif phi_s < 0: # accumulation
                phi[i] = phi_s * np.exp(-x / L_debye)
            elif x < w: # depletion
                phi[i] = phi_s * (1 - x/w)**2
            else:
                phi[i] = 0.0
                
        Ef = np.zeros_like(x_grid)
        Ei_bulk = self.phi_f
        Ei = Ei_bulk - phi
        Ec = Ei + self.Eg/2
        Ev = Ei - self.Eg/2
        
        return {
            'x': x_grid,
            'Ec': Ec,
            'Ev': Ev,
            'Ei': Ei,
            'Ef': Ef,
            'phi_s': phi_s,
            'w': w
        }
                
                
            