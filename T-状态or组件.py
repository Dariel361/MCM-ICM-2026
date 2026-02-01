import numpy as np
import json
import sys
import os
import csv
from collections import defaultdict

# Add parent directory to path to allow imports in main.py to work (e.g. SOH_model)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import from local modules
try:
    from main import BatteryModel, T_Model, load_parameters
    from the_P_load import PowerLoadModel
except ImportError:
    # If running from different context, ensure current dir is in path
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from main import BatteryModel, T_Model, load_parameters
    from the_P_load import PowerLoadModel

def run_simulation_for_temp(T_amb):
    """
    Run simulation for a specific ambient temperature T_amb.
    Returns a dictionary with stats.
    """
    # Load parameters
    all_params = load_parameters()
    
    # Disable aging as requested
    current_soh = 1.0 
    
    # Fix seed for consistent load profile across different temperatures
    if 'load_model' not in all_params or all_params['load_model'] is None:
        all_params['load_model'] = {}
    all_params['load_model']['seed'] = 42
    
    # Initialize models
    # We initialize battery and thermal models
    # T_init is set to T_amb
    battery = BatteryModel(all_params['electrochemical'], current_soh=current_soh)
    thermal = T_Model(all_params['thermal'], T_init=T_amb)
    load_model = PowerLoadModel(all_params.get('load_model'))
    
    dt = all_params['simulation']['dt']
    
    t = 0.0
    
    # Accumulators for Energy (Joules)
    # Components: Screen, CPU, Com, GPS, Background
    comp_energy = defaultdict(float)
    # States: standby, social, game, video, nav
    state_energy = defaultdict(float)
    
    while True:
        # 1. Update Thermal & R0
        T_curr = thermal.T
        R0_curr = battery.get_current_R0(T_curr)
        
        # 2. Calculate Load
        P_load, state_name, components = load_model.calculate_P_load(t)
        
        # 3. Solve Current
        I_curr, is_collapsed = battery.solve_current_from_power(P_load, R0_curr)
        
        # Stop if voltage collapsed
        if is_collapsed:
            break
            
        # 4. Check Voltage
        v_term = battery.get_U_L(I_curr, R0_curr)
        
        # Stop conditions
        if v_term < 0: # Cutoff voltage
            break
        if battery.state[0] <= 0.03: # SOC cutoff
            break
            
        # Accumulate Energy
        # Energy = Power * dt
        for c_name, p_val in components.items():
            comp_energy[c_name] += p_val * dt
            
        state_energy[state_name] += P_load * dt
        
        # 5. Update State
        battery.update_state(dt, I_curr, T_curr)
        thermal.update_temperature(dt, I_curr, battery.get_current_R0, T_amb)
        
        t += dt
        
        # Safety break (e.g., 50 hours)
        if t > 3600 * 50:
            break
            
    # Calculate statistics
    duration = t if t > 0 else 1e-6
    total_comp_energy = sum(comp_energy.values())
    total_state_energy = sum(state_energy.values())
    
    # If total energy is 0 (immediate break), handle gracefully
    if total_comp_energy == 0: total_comp_energy = 1e-9
    if total_state_energy == 0: total_state_energy = 1e-9
    
    result = {
        "Temperature": round(T_amb, 2),
        "Duration_s": round(duration, 2)
    }
    
    # Components Stats
    comp_list = ['Screen', 'CPU', 'Com', 'GPS', 'Background']
    for c in comp_list:
        e = comp_energy.get(c, 0.0)
        p_avg = e / duration
        pct = (e / total_comp_energy) * 100
        result[f"{c}_Power_W"] = round(p_avg, 4)
        result[f"{c}_Pct"] = round(pct, 2)
        
    # States Stats
    state_list = ['standby', 'social', 'game', 'video', 'nav']
    for s in state_list:
        e = state_energy.get(s, 0.0)
        p_avg = e / duration
        pct = (e / total_state_energy) * 100
        result[f"{s}_Power_W"] = round(p_avg, 4)
        result[f"{s}_Pct"] = round(pct, 2)
        
    return result

def main():
    # Define Temperature range: -40 to 60, step 0.1
    # Using integer loop to avoid float issues
    # -400 to 600, divide by 10
    temps = [x / 10.0 for x in range(-400, 601)]
    
    print(f"Starting simulation for {len(temps)} temperatures (-40 to 60, step 0.1)...")
    
    # Define CSV headers
    headers = ["Temperature", "Duration_s"]
    comp_list = ['Screen', 'CPU', 'Com', 'GPS', 'Background']
    state_list = ['standby', 'social', 'game', 'video', 'nav']
    
    for c in comp_list:
        headers.extend([f"{c}_Power_W", f"{c}_Pct"])
    for s in state_list:
        headers.extend([f"{s}_Power_W", f"{s}_Pct"])
        
    output_filename = "component_state_power_analysis.csv"
    
    # Write to CSV
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            for i, T in enumerate(temps):
                res = run_simulation_for_temp(T)
                writer.writerow(res)
                
                if (i + 1) % 50 == 0:
                    print(f"Progress: {i + 1}/{len(temps)} completed (Current T={T}°C)")
                    
        print(f"Simulation completed. Results saved to {output_filename}")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user. Partial results saved.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
