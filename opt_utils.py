import casadi as ca
import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def fetch_electricity_prices(file_name, length, start_datetime='2022-02-01 Kl. 01-02', days_padding=0, column='NO1'):
    """
    length: number of hours in the whole growth cycle
    days_padding: no. of prior days used to estimate the electricity price beyond the day-ahead
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)

    data = pd.read_csv(file_path, sep=';', index_col='Dato/klokkeslett')

    # Finn starttime in the dataset
    if start_datetime not in data.index:
        print(start_datetime)
        raise ValueError("Start-date does not exist in the dataset (2016-01-01 01-02 is the earliest and 2024-04-15 23-00 is the latest).")

    # Find indeks for startdate
    start_index = data.index.get_loc(start_datetime)
    if start_index - days_padding*24 < 0:
        raise ValueError("Not enough space for padding")
    
    # Kontroller at det er nok data tilgjengelig fra startpunktet
    if start_index + length > len(data):
        raise ValueError("Not enough available length from start to end date.")

    # Hent data fra den angitte kolonnen og lengden
    prices = data.loc[data.index[start_index-days_padding*24:start_index + length], column].values
    return np.array(prices, dtype=float)
   
def load_config(file_name) -> dict:
    """Load multiple configuration files and combine them into one dictionary."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)

    with open(file_path, 'r') as file:
        config = json.load(file)
    print(config)

    return config

def setup_optimizer(F, nx, nu, N_horizon, u_max, u_min, min_DLI, max_DLI, min_phot):
    opti = ca.Opti()

    x = opti.variable(nx, N_horizon+1)
    u = opti.variable(nu, N_horizon)

    x0 = opti.parameter(nx, 1)
    prices = opti.parameter(1, N_horizon)

    obj = ca.dot(u, prices)
    opti.minimize(obj)
    opti = set_constraints(opti=opti, constraint_horizon=N_horizon, F=F, x=x, u=u, x0=x0, u_max=u_max, u_min=u_min, min_DLI=min_DLI, max_DLI=max_DLI, min_phot=min_phot, N_horizon=N_horizon)
    opts = {}
    opti.solver('ipopt', opts)
    return opti, obj, x, u, x0, prices
def set_constraints(opti, constraint_horizon, F, x, u, x0, u_max, u_min, min_DLI, max_DLI, min_phot, N_horizon):
    opti.subject_to()
    for k in range(N_horizon):
        opti.subject_to(x[:,k+1] == F(x[:,k], u[:,k]))
        opti.subject_to([u[k] <= u_max, u[k] >= u_min])
    timestamps = []  
    iters = []
    for k in range(constraint_horizon+1):  
        
        if k % 24 == 0 and k > 0:
            
            # DLI constraint
            timestamps.append(k)
            opti.subject_to(x[1, k] - x[1,k-24] >= min_DLI)
            opti.subject_to(x[1,k] - x[1, k-24] <= max_DLI)
    print(timestamps)

    opti.subject_to(x[0, constraint_horizon] >= min_phot)
    opti.subject_to(x[:,0] == x0)

    return opti
def update_optimizer(opti, x,u,x0, prices, current_state, current_prices, time_to_harvest, N_horizon, F, u_min, u_max, min_DLI, max_DLI, min_phot):
    constraint_horizon = np.min([N_horizon, time_to_harvest])
    obj = ca.dot(u[0,:constraint_horizon], prices[:constraint_horizon])
    opti.minimize(obj)
    
    print('-'*30)
    print('Constraint horizon: ')
    print(constraint_horizon)
    print('-'*30)
    opti = set_constraints(opti=opti, constraint_horizon=constraint_horizon, F=F, x=x, u=u, x0=x0, u_max=u_max, u_min=u_min, min_DLI=min_DLI, max_DLI=max_DLI, min_phot=min_phot, N_horizon=N_horizon)
    opti.set_value(x0, current_state)
    opti.set_value(prices, current_prices)
    return opti
def solve_mpc(opti, u_symbol, x_symbol, energy_values):
    sol = opti.solve()
    u_opt = sol.value(u_symbol)
    x_opt = sol.value(x_symbol)
    tot_cost_dynamic = ca.dot(u_opt.T, energy_values)*27/(1000*0.8)
    tot_energy_dynamic = x_opt[1, -1].copy()
    
    return sol, u_opt, x_opt, tot_cost_dynamic, tot_energy_dynamic
def solve_iter(i, opti, opti_x, opti_u, opti_x0, opti_prices, current_state, temp_prices, N_SIM, N_HORIZON, time_to_end, F, MAX_INTY, min_dli, max_dli, min_phot):
    opti.set_value(opti_prices, temp_prices)
    opti.set_value(opti_x0, current_state)
    if i > 0:
        opti = update_optimizer(opti, opti_x, opti_u, opti_x0, opti_prices, current_state, temp_prices, N_SIM-i*24, N_HORIZON, F, 0, MAX_INTY, min_dli, max_dli, min_phot)
    sol, u_opt, x_opt, cost_iter, energy_iter = solve_mpc(opti, opti_u, opti_x, temp_prices)
    x_opt_plot = x_opt.copy()
    for m in range(0, time_to_end, 24):
        # Resetting the light integral every day so that it instead represents daily light integral
        x_opt_plot[1, m+1:] -= x_opt_plot[1, m]
    x_opt_plot[1, 0] = 0
    #plot_simulation_results(x_opt_plot, u_opt, temp_prices, N_HORIZON, baseline_dict['min_dli'], baseline_dict['max_dli'], min_phot, time_to_end)
    u_apply = u_opt[:24]
    return opti, sol, x_opt, u_opt, u_apply, x_opt_plot
def simulate_system(i, F, states, control_actions, current_state, u_apply):
    # Simulate the system dynamics for each control input applied
    for j in range(24):
        # Assuming F is a function that takes current_state and control input and returns next_state
        next_state = F(current_state, u_apply[j])
        states[:, i*24+j+1] = next_state.T
        control_actions[i*24 + j] = u_apply[j]
        current_state = next_state  # Update current state
        
    return states, control_actions, current_state
def store_optimal_results(states, N_SIM, max_intensity, control_actions, true_energy_price):
    optimal_dict = {}
    states_for_plot = states.copy()
    for m in range(0, N_SIM, 24):
            # Resetting the light integral every day so that it instead represents daily light integral
            states_for_plot[1, m+1:] -= states_for_plot[1, m]
    tot_cost = np.dot(control_actions, true_energy_price)*27/(1000*0.8) # Total cost in NOK 
    tot_energy = states[1, -1] # The total energy consumption for the whole simulation
    optimal_dict['states'] = states
    optimal_dict['states_for_plot'] = states_for_plot
    optimal_dict['u'] = control_actions
    optimal_dict['tot_cost'] = tot_cost
    optimal_dict['tot_energy'] = tot_energy
    optimal_dict['scaled_u'] =  np.round(optimal_dict['u']/max_intensity * 100)

    return optimal_dict
def plot_simulation_results(states_baseline, u_baseline, price, N, min_dli=None, max_dli=None, min_phot=None, end_time=None, block=True):
    """
    Plots the simulation results: total photosynthesis, light integral, light intensity, and electricity price over time.
    
    Parameters:
    states_baseline (numpy.ndarray): Array containing the simulation states over time.
                                     Shape: (nx, N_SIM+1)
    u_baseline (numpy.ndarray): Array containing the light intensity input over time.
                                Shape: (N_SIM,)
    price (numpy.ndarray): Array containing the electricity price over time
                                Shape: (N_SIM,)
    """
    time = np.arange(N+1)
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Plot total photosynthesis
    axs[0].plot(time, states_baseline[0, :], label='Total Photosynthesis')
    axs[0].set_ylabel('Total Photosynthesis')
    if end_time is not None:
        axs[0].scatter(time[end_time], min_phot, marker='x', s=6)
    axs[0].legend()

    # Plot light integral
    axs[1].plot(time, states_baseline[1, :], label='Light Integral')
    if min_dli is not None:
        axs[1].hlines([min_dli, max_dli], time[0], time[-2], linestyles='dashed', colors=['g', 'b'], alpha=0.7)
    axs[1].set_ylabel('Light Integral')
    axs[1].legend()

    # Plot light intensity (input) and electricity price with separate y-axes
    ax3 = axs[2]
    ax4 = ax3.twinx()
    ax3.step(time[:-1], u_baseline, label='Light Intensity', color='tab:blue')
    ax4.step(time[:-1], price, label='Electricity Price', color='tab:orange', alpha=0.6)

    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Light Intensity', color='tab:blue')
    ax4.set_ylabel('Electricity Price', color='tab:orange')

    ax3.legend(loc='upper left')
    ax4.legend(loc='upper right')

    plt.tight_layout()
    plt.show(block=block)

def compare_results(baseline, optimal, price, N, reset_li=True, block=True):
    """
    Plots the baseline and the optimal solutions in the same plots: total photosynthesis, light integral, light intensity, and electricity price over time.
    
    Parameters:
    baseline (dict): Dictionary containing the baseline states and control inputs over time.
                     Keys should include 'states' and 'u'.
    optimal (dict): Dictionary containing the optimal states and control input over time.
                    Keys should include 'states' and 'u'.
    price (numpy.ndarray): Array containing the electricity price over time
                           Shape: (N_SIM,)
    N (int): Number of simulation steps.
    """
    time = np.arange(N+1)
    fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    # Plot total photosynthesis
    axs[0].plot(time, baseline['states'][0, :], label='Baseline Total Photosynthesis', color='tab:blue')
    axs[0].plot(time, optimal['states'][0, :], label='Optimal Total Photosynthesis', color='tab:orange')
    axs[0].set_ylabel('Total Photosynthesis')
    axs[0].legend()

    # Plot light integral
    if reset_li:
        axs[1].plot(time, baseline['states_for_plot'][1, :], label='Baseline Light Integral', color='tab:blue')
        axs[1].plot(time, optimal['states_for_plot'][1, :], label='Optimal Light Integral', color='tab:orange')
        axs[1].set_ylabel('Light Integral')
        axs[1].legend()
    else:
        axs[1].plot(time, baseline['states'][1, :], label='Baseline Light Integral', color='tab:blue')
        axs[1].plot(time, optimal['states'][1, :], label='Optimal Light Integral', color='tab:orange')
        axs[1].set_ylabel('Light Integral')
        axs[1].legend()
    # Plot light intensity (input) and electricity price with separate y-axes
    ax3 = axs[2]
    ax4 = ax3.twinx()
    ax3.step(time[:-1], baseline['u'], label='Baseline Light Intensity', color='tab:blue')
    ax4.step(time[:-1], price, label='Electricity Price', color='tab:orange', alpha=0.6)

    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Light Intensity', color='tab:blue')
    ax4.set_ylabel('Electricity Price', color='tab:orange')

    ax3.legend(loc='upper left')
    ax4.legend(loc='upper right')

    # Plot light intensity (input) and electricity price with separate y-axes
    ax4 = axs[3]
    ax5 = ax4.twinx()
    ax4.step(time[:-1], optimal['u'], label='Optimal Light Intensity', color='tab:green')
    ax5.step(time[:-1], price, label='Electricity Price', color='tab:orange', alpha=0.6)

    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Light Intensity', color='tab:blue')
    ax5.set_ylabel('Electricity Price', color='tab:orange')

    ax3.legend(loc='upper left')
    ax4.legend(loc='upper right')

    plt.tight_layout()
    plt.show(block=block)

def compute_cheapest_costs(true_energy_price, N_SIM, photoperiod, baseline_intensity):
    """
    Computes the total cost of the cheapest k hours of energy prices per day over the simulation period.
    
    Parameters:
    true_energy_price (numpy.ndarray): Array containing the true energy prices over time
                                       Shape: (N_SIM,)
    N_SIM (int): Total number of simulation steps (hours).
    
    Returns:
    float: Total cost of the cheapest k hours of energy prices per day.
    """
    tot_cost_cheapest_days = 0
    for p in range(0, N_SIM, 24):
        current_day_prices = true_energy_price[p:p+24]
        cheapest_prices = np.partition(current_day_prices, photoperiod)[:photoperiod]  # Get the 18 smallest prices
        tot_cost_cheapest_days += np.sum(cheapest_prices)
        
    return tot_cost_cheapest_days*baseline_intensity*27/(1000*0.8)

def print_savings(baseline_dict, optimal_dict, tot_cost_cheapest_days):

        print('-'*40)
        print('Total cost of baseline experiment: ', baseline_dict['tot_cost'])
        print('Total cost of optimal experiment: ', optimal_dict['tot_cost'])
        print('Saving a percent of: ', (1- optimal_dict['tot_cost']/(baseline_dict['tot_cost'])))
        print('-'*10)
        print('Total energy of baseline experiment: ', baseline_dict['tot_energy'])
        print('Total energy of optimal experiment: ', optimal_dict['tot_energy'])
        print('Saving a percent of: ', (1- optimal_dict['tot_energy']/(baseline_dict['tot_energy'])))
        print('-'*40)
        print('Now  the x cheapest hours approach:')

        print('Total cost of baseline experiment: ', baseline_dict['tot_cost'])
        print('Total cost of k cheapest experiment: ', tot_cost_cheapest_days)
        print('Saving a percent of: ', (1- tot_cost_cheapest_days/(baseline_dict['tot_cost'])))
        print('-'*40)


def generate_temp_prices(energy_prices, padding, length, iteration):
    past_average = np.average(energy_prices[iteration*24 : iteration*24 + padding*24].copy())

    temp_prices = np.zeros((length))
    for k in range(len(temp_prices)):
        if k < 24:
            temp_prices[k] = energy_prices[padding*24 + iteration*24 + k].copy()
        else:
            temp_prices[k] = past_average
        
    return temp_prices, past_average

def run_baseline(simulation_length, photoperiod, intensity, fun, nx, prices, pct_slack, max_intensity, plot):
        states_baseline = np.zeros((nx, simulation_length+1))
        u_baseline = np.zeros((simulation_length))
        for i in range(simulation_length):
            if i % 24 < photoperiod:
                u_baseline[i] = intensity
        for i in range(simulation_length):
            # Assuming F is a function that takes current_state and control input and returns next_state
            next_state = fun(states_baseline[:,i], u_baseline[i])
            states_baseline[:, i+1] = next_state.T

        tot_cost_baseline = np.dot(u_baseline, prices)*27/(1000*0.8)
        tot_energy_baseline = states_baseline[1,-1]
        dli_baseline = states_baseline[1, 24]
        max_dli = dli_baseline + dli_baseline*pct_slack/100
        min_dli = dli_baseline - dli_baseline*pct_slack/100
        print(dli_baseline)
        print(max_dli)
        print(min_dli)

        baseline_states_for_plot = states_baseline.copy()
        print('Baseline: Total cost: ', tot_cost_baseline, ' total energy consumption: ', tot_energy_baseline)
        for i in range(0, simulation_length, 24):
            # Resetting the light integral every day so that it instead represents daily light integral
            baseline_states_for_plot[1, i+1:] -= baseline_states_for_plot[1, i]
        if plot:
            plot_simulation_results(states_baseline=baseline_states_for_plot, u_baseline=u_baseline,price=prices, N = simulation_length)
        baseline_dict = {
            'states': states_baseline,
            'u': u_baseline,
            'tot_cost': tot_cost_baseline,
            'tot_energy': tot_energy_baseline, 
            'dli': dli_baseline,
            'max_dli': max_dli,
            'min_dli': min_dli,
            'states_for_plot': baseline_states_for_plot,
            'scaled_u': np.round(u_baseline/max_intensity * 100)
        }
        return baseline_dict