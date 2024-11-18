import casadi as ca
import json
from opt_utils import *
import Photosynthesis
ENV_TEMP = 24
ENV_HUMID = None # TODO
ENV_CO2 = 400
BASELINE_PHOTOPERIOD = 16
BASELINE_INTY = 200
MAX_INTY = 250
HARDWARE_MAX_INTY = 250
PCT_SLACK = 10

DARK_HOURS = 2 # TODO implement

DAYS_HORIZON = 4
DAYS_SIMULATION = 20
DAYS_PADDING = 3
N_HORIZON = DAYS_HORIZON * 24
N_SIM = DAYS_SIMULATION * 24
START_DATE = "2024-01-01 Kl. 00-01"
BIDDING_ZONE = "NO3"

def main():
    """
    x consists of the total photosynthesis obtained and the light integral
    u consists of the light intensity
    """

    energy_prices_full = fetch_electricity_prices('Spotprices_norway.csv', length=DAYS_SIMULATION*24, start_datetime=START_DATE, days_padding=DAYS_PADDING, column=BIDDING_ZONE)
    true_energy_price = energy_prices_full[24*DAYS_PADDING:]
    photosynthesis = Photosynthesis.Photosynthesis(temp=ENV_TEMP, co2=ENV_CO2)
    F, nx, nu = photosynthesis.casadi_function(ts=3600)
    photosynthesis.plot_photosynthesis_rate()
    
    
    baseline_dict = run_baseline(N_SIM, BASELINE_PHOTOPERIOD, BASELINE_INTY, F, nx, true_energy_price, PCT_SLACK, MAX_INTY, plot=False)
    plot_simulation_results(states_baseline=baseline_dict['states_for_plot'], u_baseline=baseline_dict['u'],price=true_energy_price, N = N_SIM)

    min_phot_init = baseline_dict['states'][0, N_HORIZON]
    opti, obj, opti_x, opti_u, opti_x0, opti_prices = setup_optimizer(F,nx,nu, N_HORIZON, MAX_INTY, 0, baseline_dict['min_dli'], baseline_dict['max_dli'], min_phot_init)
    current_state = ca.vertcat(0,0)
    states = np.zeros((nx, N_SIM+1))
    states[:,0] = current_state.T

    control_actions = np.zeros((N_SIM))
    
    for i in range(0, DAYS_SIMULATION):
        # iteration i
        time_to_end = min(N_HORIZON, N_SIM-(i*24))
        min_phot = baseline_dict['states'][0, min(i*24 + N_HORIZON, N_SIM)]

        temp_prices, past_average = generate_temp_prices(energy_prices_full, DAYS_PADDING, N_HORIZON, i)
        
        opti, sol, x_opt, u_opt, u_apply, x_opt_plot = solve_iter(i, opti, opti_x, opti_u, opti_x0, opti_prices, current_state, temp_prices, N_SIM, N_HORIZON, time_to_end, F, MAX_INTY, baseline_dict['min_dli'], baseline_dict['max_dli'], min_phot)
        #plot_simulation_results(x_opt_plot, u_opt, temp_prices, N_HORIZON, baseline_dict['min_dli'], baseline_dict['max_dli'], min_phot, time_to_end)
        states, control_actions, current_state = simulate_system(i, F, states, control_actions, current_state, u_apply)
    
    optimal_dict = store_optimal_results(states, N_SIM, MAX_INTY, control_actions, true_energy_price)
    
    
    with open('scaled_optimal_intensities.json', 'w') as f:
        json.dump(optimal_dict['scaled_u'].tolist(), f)
    with open('scaled_baseline_intensities.json', 'w') as f:
        json.dump(baseline_dict['scaled_u'].tolist(), f)
    # Calculating the cost if we choose the cheapest days
    tot_cost_cheapest_days = compute_cheapest_costs(true_energy_price=true_energy_price, N_SIM=N_SIM, photoperiod=BASELINE_PHOTOPERIOD, baseline_intensity=BASELINE_INTY)
    

    print_savings(baseline_dict, optimal_dict, tot_cost_cheapest_days)
    plot_simulation_results(states, control_actions, true_energy_price, N_SIM, block=False)
    
    compare_results(baseline_dict, optimal_dict, true_energy_price, N_SIM, reset_li=True, block=False)
    compare_results(baseline_dict, optimal_dict, true_energy_price, N_SIM, reset_li=False)

if __name__ == "__main__":
    main()