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
    # Fallback if running from root
    sys.path.append(os.path.join(parent_dir, 'Model'))
    from main import BatteryModel, T_Model, load_parameters

class FastLoadModel:
    def __init__(self, original_model, seed):
        self.model = original_model
        self.rng = np.random.RandomState(seed)
        self.current_state = 0 # standby
        self.current_t = 0.0
        self.next_transition_t = 0.0
        self.current_P = 0.0
        self.current_state_name = 'standby'
        
        # Initial setup
        self._update_next_transition()
        self._update_power()
        
    def _update_next_transition(self):
        rate_out = -self.model.Q[self.current_state, self.current_state]
        # tau in hours
        tau_hours = self.rng.exponential(1.0 / rate_out)
        tau_seconds = tau_hours * 3600
        # self.next_transition_t is absolute simulation time
        self.next_transition_t = self.current_t + tau_seconds
        
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
            if i == 3:  # GPS
                val = 1.0 if mean_val > 0.5 else 0.0
            else:
                val = max(mean_val + noise, 0.05)
            params.append(val)

        L, cpu_usage, data_rate, gps_active = params

        # Power calculation formulas from PowerLoadModel
        P_screen = 1.5 * (L ** 2.2)
        P_cpu = 0.1 + 1.9 * (cpu_usage ** 1.3)
        P_com = 0.2 + 0.5 * data_rate / 0.4
        P_gps = 0.3 if gps_active > 0.5 else 0.0
        P_background = 0.15

        P_foreground = P_screen + P_cpu + P_com + P_gps
        self.current_P = P_background + P_foreground
        
    def step(self, dt):
        self.current_t += dt
        # Handle transitions
        while self.current_t >= self.next_transition_t:
             self._transition()
        
        return self.current_P, self.current_state_name

def run_monte_carlo():
    # Load parameters
    param_path = os.path.join(current_dir, 'parameters.json')
    all_params = load_parameters(param_path)
    sim_conf = all_params['simulation']
    
    # Calculate SOH once (fixed age)
    soh_model = SOHModel(all_params.get('soh_model', {}))
    battery_age_months = sim_conf.get('battery_age_months', 0.0)
    current_soh_val = soh_model.calculate_SOH(battery_age_months)
    
    # Template load model to get constants
    base_load_model = PowerLoadModel(all_params.get('load_model'))
    
    num_runs = 1000
    results = []
    
    print(f"Starting {num_runs} Monte Carlo simulations...")
    
    for i in tqdm(range(num_runs)):
        # New seed for each run
        seed = np.random.randint(0, 1000000)
        
        # Reset models
        battery = BatteryModel(all_params['electrochemical'], current_soh=current_soh_val)
        thermal = T_Model(all_params['thermal'], T_init=sim_conf['initial_temp'])
        load_gen = FastLoadModel(base_load_model, seed)
        
        dt = sim_conf['dt']
        T_amb = sim_conf['T_amb']
        t = 0.0
        
        # Track energy (Joules)
        energy_map = {s: 0.0 for s in base_load_model.states}
        
        while True:
            # 1. Get thermal state
            T_curr = thermal.T
            R0_curr = battery.get_current_R0(T_curr)
            
            # 2. Get Load
            P_load, state_name = load_gen.step(dt)
            
            # 3. Electrical solve
            I_curr, is_collapsed = battery.solve_current_from_power(P_load, R0_curr)
            
            if is_collapsed:
                # Voltage collapse
                break
            
            v_term = battery.get_U_L(I_curr, R0_curr)
            
            # 4. Accumulate Energy
            # Energy = Power * dt
            energy_map[state_name] += P_load * dt
            
            # 5. Check Cutoff
            if v_term < 3.0: # Explicit cutoff voltage usually 3.0 or 2.5
                # But main.py checks `v_term < 0` which is extremely low, 
                # however it also has `battery.state[0] <= 0.03`.
                pass # main.py just marks cutoff time but continues until 0.03 SOC
            
            if battery.state[0] <= 0.03:
                break
                
            # 6. Update States
            battery.update_state(dt, I_curr, T_curr)
            thermal.update_temperature(dt, I_curr, battery.get_current_R0, T_amb)
            
            t += dt
            
            # Safety break for infinite loops
            if t > 3600 * 48: # 48 hours
                break
        
        # Convert Joules to Wh
        energy_wh = {k: v / 3600.0 for k, v in energy_map.items()}
        results.append(energy_wh)
        
    return results

def plot_results(results):
    df = pd.DataFrame(results)
    
    # Reorder columns to match logical order if needed
    order = ['standby', 'social', 'game', 'video', 'nav']
    df = df[order]
    
    plt.figure(figsize=(10, 6))
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create boxplot
    # boxprops to set facecolor
    box = plt.boxplot([df[col] for col in df.columns], 
                      labels=df.columns, 
                      patch_artist=True,
                      notch=True,
                      medianprops={'color': 'gold', 'linewidth': 1.5},
                      flierprops={'marker': 'o', 'markerfacecolor': 'gray', 'markersize': 4})
    
    # Colors for boxes
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#CC99FF']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    plt.title('Energy Consumption Distribution by State (100 Monte Carlo Runs)', fontsize=14, pad=15)
    plt.xlabel('Device State', fontsize=12)
    plt.ylabel('Energy Consumption (Wh)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    output_path = os.path.join(current_dir, 'energy_boxplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    data = run_monte_carlo()
    plot_results(data)
