import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json
from tqdm import tqdm
import pandas as pd

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

# Import models
from SOH_model import SOHModel
from the_P_load import PowerLoadModel
try:
    from main import BatteryModel, T_Model, load_parameters
except ImportError:
    sys.path.append(os.path.join(parent_dir, 'Model'))
    from main import BatteryModel, T_Model, load_parameters

class FastLoadModel:
    """
    Optimized load model for simulation
    """
    def __init__(self, original_model, seed):
        self.model = original_model
        self.rng = np.random.RandomState(seed)
        self.current_state = 0 # standby
        self.current_t = 0.0
        self.next_transition_t = 0.0
        self.current_P = 0.0
        self.current_state_name = 'standby'
        self._update_next_transition()
        self._update_power()
        
    def _update_next_transition(self):
        rate_out = -self.model.Q[self.current_state, self.current_state]
        tau_hours = self.rng.exponential(1.0 / rate_out)
        self.next_transition_t = self.current_t + tau_hours * 3600
        
    def _transition(self):
        probs = self.model.Q[self.current_state, :].copy()
        probs[self.current_state] = 0
        probs = np.maximum(probs, 0)
        if probs.sum() > 0:
            probs = probs / probs.sum()
            next_state = self.rng.choice(5, p=probs)
        else:
            next_state = self.current_state
        self.current_state = next_state
        self.current_state_name = self.model.states[self.current_state]
        self._update_next_transition()
        self._update_power()
        
    def _update_power(self):
        state_name = self.current_state_name
        mean_params = self.model.params_mean[state_name]
        params = []
        for i, mean_val in enumerate(mean_params):
            noise = 0.2 * mean_val * self.rng.randn()
            val = 1.0 if i == 3 and mean_val > 0.5 else max(mean_val + noise, 0.05)
            params.append(val)
        L, cpu_usage, data_rate, gps_active = params
        P = 1.5*(L**2.2) + (0.1 + 1.9*(cpu_usage**1.3)) + (0.2 + 0.5*data_rate/0.4) + (0.3 if gps_active > 0.5 else 0.0) + 0.15
        self.current_P = P
        
    def step(self, dt):
        self.current_t += dt
        while self.current_t >= self.next_transition_t:
             self._transition()
        return self.current_P, self.current_state_name

def simulate_for_age(age_months, num_runs=20):
    """
    Run Monte Carlo simulations for a specific battery age
    Returns average energy consumption per state (Wh)
    """
    param_path = os.path.join(current_dir, 'parameters.json')
    all_params = load_parameters(param_path)
    sim_conf = all_params['simulation']
    
    # Calculate SOH for this specific age
    soh_model = SOHModel(all_params.get('soh_model', {}))
    current_soh_val = soh_model.calculate_SOH(age_months)
    
    base_load_model = PowerLoadModel(all_params.get('load_model'))
    states = base_load_model.states
    total_energies = {s: 0.0 for s in states}
    
    # Run simulations
    for _ in range(num_runs):
        seed = np.random.randint(0, 1000000)
        battery = BatteryModel(all_params['electrochemical'], current_soh=current_soh_val)
        thermal = T_Model(all_params['thermal'], T_init=sim_conf['initial_temp'])
        load_gen = FastLoadModel(base_load_model, seed)
        
        dt = sim_conf['dt']
        T_amb = sim_conf['T_amb']
        t = 0.0
        
        run_energies = {s: 0.0 for s in states}
        
        while True:
            T_curr = thermal.T
            R0_curr = battery.get_current_R0(T_curr)
            P_load, state_name = load_gen.step(dt)
            
            # Electrical Check
            I_curr, is_collapsed = battery.solve_current_from_power(P_load, R0_curr)
            
            if is_collapsed:
                break
            
            # Accumulate Energy (Joules)
            run_energies[state_name] += P_load * dt
            
            # Update States
            battery.update_state(dt, I_curr, T_curr)
            thermal.update_temperature(dt, I_curr, battery.get_current_R0, T_amb)
            
            # Check cutoffs
            if battery.state[0] <= 0.03: # Empty
                break
            if t > 3600 * 48: # Safety timeout
                break
                
            t += dt
            
        for s in states:
            total_energies[s] += run_energies[s]
            
    # Calculate Average Wh
    avg_energies_wh = {k: (v / num_runs) / 3600.0 for k, v in total_energies.items()}
    return avg_energies_wh

def generate_high_res_chart():
    # Rigorous Simulation Settings
    step_size = 0.5
    months_points = np.arange(0, 60 + step_size, step_size)
    num_mc_runs = 20
    
    # Data collection
    data = {
        'standby': [],
        'social': [],
        'game': [],
        'video': [],
        'nav': []
    }
    
    csv_rows = []
    
    print(f"Starting RIGOROUS simulation for ages: 0 to 60 months, step={step_size}")
    print(f"Total points: {len(months_points)}")
    print(f"Runs per point: {num_mc_runs}")
    print("This will take some time...")
    
    for m in tqdm(months_points):
        results = simulate_for_age(m, num_runs=num_mc_runs)
        
        # Prepare plotting data
        for s in data.keys():
            data[s].append(results.get(s, 0))
            
        # Prepare CSV data
        row = {'Age_Months': m}
        row.update(results)
        # Add total energy
        row['Total_Energy_Wh'] = sum(results.values())
        csv_rows.append(row)
            
    # Save Data to CSV
    df_results = pd.DataFrame(csv_rows)
    csv_path = os.path.join(current_dir, 'aging_simulation_results_high_res.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"Experiment data saved to {csv_path}")

    # Plotting
    plt.figure(figsize=(14, 8))
    plt.style.use('seaborn-v0_8-white')
    
    # Data for stackplot
    x = months_points
    y = [data['standby'], data['social'], data['game'], data['video'], data['nav']]
    labels = ['Standby', 'Social', 'Game', 'Video', 'Nav']
    
    # Revised Colors (No Green at Top)
    colors = [
        '#CFD8DC', # Standby - Light Blue Grey
        '#90CAF9', # Social - Light Blue
        '#FFCC80', # Game - Orange
        '#EF9A9A', # Video - Light Red
        '#CE93D8'  # Nav - Light Purple (Top)
    ]
    
    plt.stackplot(x, y, labels=labels, colors=colors, alpha=0.9)
    
    plt.title('Energy Consumption Composition vs. Battery Aging (High Resolution)', fontsize=16, pad=20, weight='bold')
    plt.xlabel('Battery Age (Months)', fontsize=12)
    plt.ylabel('Total Energy Consumed per Cycle (Wh)', fontsize=12)
    plt.xlim(0, 60)
    
    # Legend
    plt.legend(loc='upper right', frameon=True, fontsize=10, bbox_to_anchor=(1.12, 1))
    
    # Grid and Spines
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Annotations for total capacity drop
    total_start = sum(data[s][0] for s in data)
    total_end = sum(data[s][-1] for s in data)
    drop_pct = (total_start - total_end) / total_start * 100
    
    plt.text(62, total_end, f"-{drop_pct:.1f}% Capacity", 
             verticalalignment='center', color='red', fontsize=12, weight='bold')

    plt.tight_layout()
    output_path = os.path.join(current_dir, 'aging_energy_stacked_area_high_res.png')
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved to {output_path}")

if __name__ == "__main__":
    generate_high_res_chart()
