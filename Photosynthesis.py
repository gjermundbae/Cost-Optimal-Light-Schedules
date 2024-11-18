import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
class Photosynthesis:
    def __init__(self, temp, co2):
        self.temp = temp
        self.co2 = co2
        self.lai = 3
        self.g_stm = 0.005
        self.g_bnd = 0.007
        self.inty_to_par = 0.217
        self.c_w = 1.83e-3
        self.c_Gamma = 71.2
        self.c_car_1 = -1.32e-5
        self.c_car_2 = 5.94e-4
        self.c_car_3 = 2.64e-3
        self.c_epsilon = 17e-6
        self.x_init = ca.vertcat(0, 0)
        self.precompute_constants()

    def precompute_constants(self):
        self.g_car = self.c_car_1 * self.temp**2 + self.c_car_2 * self.temp - self.c_car_3
        self.g_CO2 = 1 / (1 / self.g_bnd + 1 / self.g_stm + 1 / self.g_car)
        self.Gamma = self.c_Gamma * 2 ** ((self.temp - 20) / 10)
        self.epsilon_biomass = self.c_epsilon * (self.co2 - self.Gamma) / (self.co2 + 2 * self.Gamma)
        self.A_sat = self.g_CO2 * self.c_w * (self.co2 - self.Gamma)
        self.k_slope = self.epsilon_biomass
        self.curve = 0.9
        self.slope = 0.927 * self.k_slope

    def photosynthesis_model(self):
        # Returns explicit rates and control variables
        # This model does not consider leaf area index or crop area cover
        phot = ca.MX.sym('phot')
        light_integral = ca.MX.sym('light_integral')
        x = ca.vertcat(phot, light_integral)
    
        u_light = ca.MX.sym('u_light') # light intensity
        u = ca.vertcat(u_light)

        term1 = self.A_sat + self.slope * u_light * self.inty_to_par
        term2 = ca.sqrt(1e-9 + term1**2 - 4 * self.curve * self.slope * u_light * self.inty_to_par * self.A_sat)
        phot_rate = (term1 - term2) / (2 * self.curve)
        light_rate = u_light
        f_expl = ca.vertcat(phot_rate, light_rate)
        return x, u, f_expl
    
    def casadi_function(self, ts=3600):
        states, controls, expl = self.photosynthesis_model()
        f = ca.Function('f', [states, controls], [expl], ['x', 'u'], ['ode'])
        intg_options = {}
        ode = {
            'x': states,
            'p': controls,
            'ode': f(states,controls)
        }
        intg = ca.integrator('intg', 'rk', ode, 0, ts, intg_options)
        res = intg(x0=states, p=controls)
        x_next = res['xf']
        F = ca.Function('F', [states, controls], [x_next], ['x', 'u_control'], ['x_next'])

        nx = states.shape[0]
        nu = controls.shape[0]

        return F, nx, nu
        

    def evaluate_photosynthesis_model(self, u_light_values):
        x, u, f = self.photosynthesis_model()
        phot_rate_fun = ca.Function('phot_rate_fun', [u], [f[0]])
        light_rate_fun = ca.Function('light_rate_fun', [u], [f[1]])

        phot_rate_values = phot_rate_fun(u_light_values)
        light_rate_values = light_rate_fun(u_light_values)

        return phot_rate_values.full().flatten(), light_rate_values.full().flatten()

    def plot_photosynthesis_rate(self):
        u_light_values = ca.DM(np.linspace(0, 500, 400))
        phot_rate_values, light_rate_values = self.evaluate_photosynthesis_model(u_light_values)

        plt.plot(light_rate_values, phot_rate_values)
        plt.xlabel('Light Intensity (u_light)')
        plt.ylabel('Photosynthesis Rate')
        plt.title('Photosynthesis Rate vs Light Intensity')
        plt.show()