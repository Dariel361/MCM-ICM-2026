import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import random
import time

import multiprocessing
from multiprocessing import Pool, cpu_count

# Ensure we can import modules from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from main import BatteryModel, T_Model, load_parameters
from the_P_load import PowerLoadModel
from xiguanP_t import PCalculator

# Load parameters once globally to avoid repeated I/O in worker processes
# This works because on macOS/Linux fork/spawn will handle this, 
# or it will be re-executed in spawned processes.
try:
    PARAMS_PATH = os.path.join(current_dir, 'parameters.json')
    GLOBAL_PARAMS = load_parameters(PARAMS_PATH)
except Exception:
    GLOBAL_PARAMS = None

def run_single_simulation(seed):
    """
    Run a single simulation with a specific random seed.
    Returns:
        seed (int): The seed used.
        P (float): The calculated P value.
        total_time_minutes (float): The total simulation time in minutes.
    """
    # Use global parameters if available, otherwise load them
    if GLOBAL_PARAMS:
        all_params = GLOBAL_PARAMS
    else:
        params_path = os.path.join(current_dir, 'parameters.json')
        all_params = load_parameters(params_path)
        
    sim_conf = all_params['simulation']
    
    # Initialize models
    battery = BatteryModel(all_params['electrochemical'])
    thermal = T_Model(all_params['thermal'], T_init=sim_conf['initial_temp'])
    
    # Configure load model with specific seed
    load_params = all_params.get('load_model', {}).copy()
    load_params['seed'] = seed
    load_model = PowerLoadModel(load_params)
    
    # Initialize P Calculator
    p_calc = PCalculator()
    
    dt = sim_conf['dt']
    T_amb = sim_conf['T_amb']
    
    t = 0.0
    
    # Safety breakout
    MAX_STEPS = 48 * 3600 # Max 48 hours simulated time
    
    while t < MAX_STEPS:
        # A. Get thermal state and internal resistance
        T_curr = thermal.T
        R0_curr = battery.get_current_R0(T_curr)

        # A+. Calculate current load power and state
        P_load, state_name, _ = load_model.calculate_P_load(t)
        
        # Track state duration
        p_calc.update(state_name, dt)
        
        # B. Calculate current from power
        I_curr, is_collapsed = battery.solve_current_from_power(P_load, R0_curr)
        
        # C. Check for voltage collapse
        if is_collapsed:
            break
            
        # D. Calculate terminal voltage
        v_term = battery.get_U_L(I_curr, R0_curr)
        
        # E. Cutoff conditions
        if v_term < 0:
            break
        
        if battery.state[0] <= 0.03:
            break
            
        # F. Update states
        battery.update_state(dt, I_curr, T_curr)
        thermal.update_temperature(dt, I_curr, battery.get_current_R0, T_amb)
        
        t += dt
        
    return seed, p_calc.calculate_P(), t / 60.0

def main():
    target_simulations = 10000  # Reduced to 10,000
    csv_filename = 'simulation_results_10k.csv'
    csv_path = os.path.join(current_dir, csv_filename)
    
    # 1. Load existing results to support resume
    completed_seeds = set()
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r') as f:
                header = f.readline()
                for line in f:
                    if line.strip():
                        parts = line.strip().split(',')
                        if len(parts) >= 1:
                            completed_seeds.add(int(parts[0]))
            print(f"Found {len(completed_seeds)} existing records in {csv_filename}. Resuming...")
        except Exception as e:
            print(f"Error reading existing CSV: {e}. Starting fresh/appending safely.")

    # 2. Generate remaining seeds
    # We generate all target seeds, then filter out completed ones
    all_seeds = [(int(time.time() * 1000) + i * 997) % (2**32 - 1) for i in range(target_simulations)]
    # Use a list comprehension to keep order, though not strictly necessary
    remaining_seeds = [s for s in all_seeds if s not in completed_seeds]
    
    if not remaining_seeds:
        print("All target simulations completed!")
    else:
        print(f"Remaining simulations to run: {len(remaining_seeds)}")
        
        # Use multiprocessing
        num_processes = max(1, cpu_count() - 1)
        print(f"Starting simulations using {num_processes} parallel processes...")
        
        start_time = time.time()
        
        # Open CSV in append mode
        # If file doesn't exist or is empty, write header
        is_new_file = not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0
        
        with open(csv_path, 'a', buffering=1) as f: # buffering=1 for line buffering
            if is_new_file:
                f.write("Seed,P_value,Time_Minutes\n")
            
            with Pool(processes=num_processes) as pool:
                # Use imap_unordered for real-time results processing
                chunksize = 20
                counter = 0
                total = len(remaining_seeds)
                
                for res in pool.imap_unordered(run_single_simulation, remaining_seeds, chunksize=chunksize):
                    seed, p_val, t_val = res
                    # Write to CSV immediately
                    f.write(f"{seed},{p_val},{t_val}\n")
                    
                    counter += 1
                    if counter % 100 == 0 or counter == total:
                        elapsed = time.time() - start_time
                        rate = counter / elapsed
                        eta = (total - counter) / rate / 60.0 if rate > 0 else 0
                        print(f"Progress: {counter}/{total} ({counter/total:.1%}) - Rate: {rate:.1f} sim/s - ETA: {eta:.1f} min")

        print(f"Batch completed in {time.time() - start_time:.2f} seconds.")

    # 3. Plotting from CSV
    print("Generating plot from all data...")
    # Re-read all data for plotting
    data_P = []
    data_time = []
    
    with open(csv_path, 'r') as f:
        header = f.readline()
        for line in f:
            if line.strip():
                try:
                    parts = line.strip().split(',')
                    data_P.append(float(parts[1]))
                    data_time.append(float(parts[2]))
                except ValueError:
                    continue
    
    if not data_P:
        print("No data found to plot.")
        return

    plt.figure(figsize=(12, 8))
    
    # Hexbin plot
    hb = plt.hexbin(data_P, data_time, gridsize=50, cmap='Blues', mincnt=1)
    cb = plt.colorbar(hb, label='Count')
    
    # Trend line
    if len(data_P) > 1:
        z = np.polyfit(data_P, data_time, 2)
        p = np.poly1d(z)
        x_range = np.linspace(min(data_P), max(data_P), 100)
        plt.plot(x_range, p(x_range), "r--", linewidth=3, label='Trend Curve (Poly fit)')

    plt.title(f'Relationship between P-value and Battery Life (n={len(data_P)})', fontsize=14)
    plt.xlabel('P (Weighted Usage Ratio)', fontsize=12)
    plt.ylabel('Total Time (minutes)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    output_filename = 'P_vs_Time_Curve_HighRes.png'
    output_path = os.path.join(current_dir, output_filename)
    plt.savefig(output_path, dpi=300)
    print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    main()
