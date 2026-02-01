import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import csv
import time

# Add parent directory to path to import SOH_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SOH_model import SOHModel
from the_P_load import PowerLoadModel
from main import BatteryModel, T_Model, load_parameters

def run_simulation(age_months, T_amb=25.0):
    # Load parameters
    all_params = load_parameters()
    sim_conf = all_params['simulation']
    
    # 1. Setup SOH and calculate current SOH value
    soh_params = all_params.get('soh_model', {})
    soh_model = SOHModel(soh_params)
    current_soh_val = soh_model.calculate_SOH(age_months)
    
    # 2. Initialize models
    # Note: We pass current_soh_val to BatteryModel
    battery = BatteryModel(all_params['electrochemical'], current_soh=current_soh_val)
    thermal = T_Model(all_params['thermal'], T_init=T_amb) # Initialize at ambient temperature
    load_model = PowerLoadModel(all_params.get('load_model'))
    
    dt = sim_conf['dt']
    t = 0.0
    
    # Simulation Loop
    while True:
        # A. Thermal & R0
        T_curr = thermal.T
        R0_curr = battery.get_current_R0(T_curr)

        # B. Power Load
        P_load, _, _ = load_model.calculate_P_load(t)
        
        # C. Current Calculation
        I_curr, is_collapsed = battery.solve_current_from_power(P_load, R0_curr)
        
        if is_collapsed:
            # Voltage collapse (battery cannot support the load)
            break
            
        # D. Voltage
        v_term = battery.get_U_L(I_curr, R0_curr)
        
        # E. Cutoff conditions
        # 1. SOC cutoff (Primary shutdown condition as per main.py)
        if battery.state[0] <= 0.03:
            break
            
        # 2. Voltage cutoff (Safety fallback, usually around 3.0V-3.4V)
        # Using 3.0V as a hard cutoff for shutdown
        if v_term < 3.0: 
            break
        
        # F. Update State
        battery.update_state(dt, I_curr, T_curr)
        thermal.update_temperature(dt, I_curr, battery.get_current_R0, T_amb)
        
        t += dt
        
    return t / 60.0 # Return runtime in minutes

def main():
    # Setup experiment: 0 to 60 months, step 0.5
    ages = np.arange(0, 60.5, 0.5) 
    
    csv_file = 'aging_simulation_results.csv'
    
    print(f"=== Battery Aging Analysis Simulation ===")
    print(f"Temperature: 25°C")
    print(f"Ages to simulate: {len(ages)} points from 0 to 60 months (step 0.5)")
    print(f"Saving data incrementally to {csv_file}")
    print("-" * 40)
    
    # Prepare CSV file
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Age_Months', 'Runtime_Minutes'])
    
    runtimes = []
    
    for age in ages:
        start_time = time.time()
        print(f"Simulating Age: {age:.1f} months... ", end="", flush=True)
        
        runtime = run_simulation(age, T_amb=25.0)
        runtimes.append(runtime)
        
        # Save result immediately
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([age, runtime])
            
        elapsed = time.time() - start_time
        print(f"Runtime: {runtime:.2f} min (Sim took {elapsed:.2f}s)")
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(ages, runtimes, linestyle='-', color='royalblue', linewidth=1.5)
    
    plt.title('Battery Runtime vs. Aging Time', fontsize=14)
    plt.xlabel('Aging Time (months)', fontsize=12)
    plt.ylabel('Total Runtime (minutes)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Simplify x-axis ticks
    plt.xticks(np.arange(0, 61, 5))
    
    output_path = 'battery_aging_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("-" * 40)
    print(f"Simulation complete. Plot saved to {os.path.abspath(output_path)}")
    print(f"Data saved to {os.path.abspath(csv_file)}")

if __name__ == "__main__":
    main()
