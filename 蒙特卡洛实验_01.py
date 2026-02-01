import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

# Ensure we can import from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import models from main.py
try:
    from main import BatteryModel, T_Model, load_parameters
except ImportError:
    # Fallback if main.py is not importable directly (e.g. if running from parent dir)
    sys.path.append(os.path.join(current_dir, 'Model'))
    from main import BatteryModel, T_Model, load_parameters

# Optimized Load Model for Streaming Simulation (replicating PowerLoadModel logic)
class StreamPowerLoadModel:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        
        self.states = ['idle', 'social', 'game', 'video', 'nav']
        # Transition Matrix Q
        self.Q = np.array([
            [-2.0, 1.0, 0.3, 0.5, 0.2],  # idle
            [0.8, -3.0, 0.5, 1.2, 0.5],  # social
            [0.2, 0.3, -1.5, 0.5, 0.5],  # game
            [0.5, 0.8, 0.4, -2.5, 0.4],  # video
            [0.3, 0.4, 0.3, 0.5, -3.5]   # nav
        ])
        
        # Mean parameters for each state
        self.params_mean = {
            'idle': [0.15, 0.10, 0.05, 0.05],
            'social': [0.40, 0.30, 0.30, 0.10],
            'game': [0.85, 0.75, 0.50, 0.01],
            'video': [0.70, 0.40, 2.00, 0.05],
            'nav': [0.80, 0.50, 0.20, 1.00]
        }
        
        # Initial state setup
        self.current_state_idx = 0 # Start at idle
        self.time_in_state = 0.0
        self.holding_time = self._sample_holding_time(0)
        self.current_P_load = self._calculate_power(0)
        
    def _sample_holding_time(self, state_idx):
        rate_out = -self.Q[state_idx, state_idx]
        tau_hours = self.rng.exponential(1.0 / rate_out)
        return tau_hours * 3600 # Convert to seconds

    def _calculate_power(self, state_idx):
        state_name = self.states[state_idx]
        mean_params = self.params_mean[state_name]
        
        params = []
        for i, mean_val in enumerate(mean_params):
            # Noise logic matching original model
            noise = 0.2 * mean_val * self.rng.standard_normal()
            if i == 3:  # GPS
                val = 1.0 if mean_val > 0.5 else 0.0
            else:
                val = max(mean_val + noise, 0.05)
            params.append(val)

        L, cpu_usage, data_rate, gps_active = params

        # Power calculation
        P_screen = 1.5 * (L ** 2.2)
        P_cpu = 0.1 + 1.9 * (cpu_usage ** 1.3)
        P_com = 0.2 + 0.5 * data_rate / 0.4
        P_gps = 0.3 if gps_active > 0.5 else 0.0
        P_background = 0.15

        P_foreground = P_screen + P_cpu + P_com + P_gps
        P_load = P_background + P_foreground
        return P_load

    def _transition(self):
        probs = self.Q[self.current_state_idx, :].copy()
        probs[self.current_state_idx] = 0
        probs = np.maximum(probs, 0)
        
        if probs.sum() > 0:
            probs = probs / probs.sum()
            next_state = self.rng.choice(5, p=probs)
        else:
            next_state = self.current_state_idx
            
        self.current_state_idx = next_state
        self.holding_time = self._sample_holding_time(next_state)
        # Recalculate power parameters for the new state visit
        self.current_P_load = self._calculate_power(next_state)

    def step(self, dt):
        """Advances time by dt and returns (current_state_name, P_load)."""
        self.time_in_state += dt
        
        # Handle transitions
        while self.time_in_state >= self.holding_time:
            dt_remaining = self.time_in_state - self.holding_time
            self._transition()
            self.time_in_state = dt_remaining
            
        return self.states[self.current_state_idx], self.current_P_load

def run_simulation(sim_id, params):
    # Extract Simulation Settings
    sim_conf = params['simulation']
    dt = sim_conf.get('dt', 1.0)
    T_amb = 25.0 # Fixed as per requirement
    initial_soc = 1.0 # Fixed as per requirement
    
    # Initialize Models
    battery = BatteryModel(params['electrochemical'])
    battery.state[0] = initial_soc
    
    thermal = T_Model(params['thermal'], T_init=25.0)
    
    # Use optimized stream load model with unique seed
    load_model = StreamPowerLoadModel(seed=sim_id)
    
    # Trackers
    # Using dictionary for energy and time per state
    state_energy = {s: 0.0 for s in load_model.states}
    state_time = {s: 0.0 for s in load_model.states}
    
    t = 0.0
    active = True
    
    while active:
        # 1. Get Load
        current_state_name, P_load = load_model.step(dt)
        
        # 2. Get Battery State
        T_curr = thermal.T
        R0_curr = battery.get_current_R0(T_curr)
        
        # 3. Calculate Current
        I_curr, is_collapsed = battery.solve_current_from_power(P_load, R0_curr)
        
        # 4. Collapse Check
        if is_collapsed:
            break
            
        # 5. Voltage Check
        v_term = battery.get_U_L(I_curr, R0_curr)
        
        # Accumulate Data (Energy in Joules = Power * Time)
        state_energy[current_state_name] += P_load * dt
        state_time[current_state_name] += dt
        
        # 6. Cutoff Conditions
        # Voltage cutoff (using 3.0V as reasonable cutoff, or following main.py logic)
        # main.py checks < 0 and SOC < 0.03. 
        # Ideally, phones turn off around 3.3V - 3.4V.
        # Let's use 3.0V to be safe or SOC < 0.01
        if v_term < 3.0 or battery.state[0] <= 0.01:
            break
            
        # 7. Update Models
        battery.update_state(dt, I_curr, T_curr)
        thermal.update_temperature(dt, I_curr, battery.get_current_R0, T_amb)
        
        t += dt
        
        # Safety break (e.g. if > 24 hours)
        if t > 86400: 
            break
            
    return state_energy, state_time

def main():
    # 1. Load Parameters
    try:
        all_params = load_parameters()
    except Exception:
        # If running from different context, try to load directly
        with open('parameters.json', 'r', encoding='utf-8') as f:
            all_params = json.load(f)

    # 2. Monte Carlo Settings
    N_SIMULATIONS = 1000
    print(f"Starting {N_SIMULATIONS} Monte Carlo Simulations...")
    print("Conditions: T_amb=25°C, Initial SoC=1.0")
    
    # Data Storage
    # List of dictionaries, each dict contains result for one simulation
    energy_results = [] # [{'idle': E1, ...}, ...]
    time_results = []   # [{'idle': T1, ...}, ...]
    
    for i in range(N_SIMULATIONS):
        # Use a seed based on index to ensure reproducibility across runs if needed
        # but variability between runs.
        seed = i + 1000
        e_dict, t_dict = run_simulation(seed, all_params)
        
        energy_results.append(e_dict)
        time_results.append(t_dict)
        
        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{N_SIMULATIONS}")
            
    # 3. Data Processing
    states = ['idle', 'social', 'game', 'video', 'nav']
    
    # Calculate Energy Percentages per run
    # For each run, calculate % of energy for each state
    energy_pct_means = {s: 0.0 for s in states}
    
    for e_dict in energy_results:
        total_e = sum(e_dict.values())
        if total_e > 0:
            for s in states:
                energy_pct_means[s] += (e_dict[s] / total_e) * 100.0
    
    for s in states:
        energy_pct_means[s] /= N_SIMULATIONS
        
    # Calculate Time Percentages per run
    time_pct_means = {s: 0.0 for s in states}
    for t_dict in time_results:
        total_t = sum(t_dict.values())
        if total_t > 0:
            for s in states:
                time_pct_means[s] += (t_dict[s] / total_t) * 100.0
                
    for s in states:
        time_pct_means[s] /= N_SIMULATIONS
        
    print("\nResults Computed.")
    print("Mean Energy %:", energy_pct_means)
    print("Mean Time %:", time_pct_means)
    
    # 4. Plotting
    # Chart 1: Bar Chart of Mean Energy Consumption Percentage
    plt.figure(figsize=(10, 6))
    
    # Data for plotting
    states_labels = [s.capitalize() for s in states]
    energy_values = [energy_pct_means[s] for s in states]
    
    bars = plt.bar(states_labels, energy_values, color=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f'])
    
    plt.title('Average Energy Consumption by State (Monte Carlo n=1000)', fontsize=14)
    plt.xlabel('State', fontsize=12)
    plt.ylabel('Energy Consumption (%)', fontsize=12)
    plt.ylim(0, max(energy_values) * 1.2) # Add some headroom
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom')
                 
    plt.tight_layout()
    plt.savefig('Energy_Consumption_Distribution.png')
    print("Saved Energy_Consumption_Distribution.png")
    
    # Chart 2: Pie Chart of Mean Time Distribution
    plt.figure(figsize=(8, 8))
    
    time_values = [time_pct_means[s] for s in states]
    
    plt.pie(time_values, labels=states_labels, autopct='%1.1f%%', startangle=140,
            colors=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f'],
            pctdistance=0.85, explode=[0.05]*5)
            
    plt.title('Average Time Spent in Each State', fontsize=14)
    
    # Draw circle for donut chart style (optional, but looks nice)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.tight_layout()
    plt.savefig('Time_Distribution_Pie.png')
    print("Saved Time_Distribution_Pie.png")

if __name__ == "__main__":
    main()
