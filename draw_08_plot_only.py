import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can import modules from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    csv_filename = 'simulation_results_10k.csv'
    csv_path = os.path.join(current_dir, csv_filename)
    
    print(f"Reading data from {csv_path}...")
    
    data_P = []
    data_time = []
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return

    with open(csv_path, 'r') as f:
        header = f.readline()
        count = 0
        for line in f:
            if line.strip():
                try:
                    parts = line.strip().split(',')
                    # Format: Seed,P_value,Time_Minutes
                    if len(parts) >= 3:
                        p_val = float(parts[1])
                        t_val = float(parts[2])
                        data_P.append(p_val)
                        data_time.append(t_val)
                        count += 1
                except ValueError:
                    continue
    
    print(f"Loaded {len(data_P)} data points.")
    
    if not data_P:
        print("No data found to plot.")
        return

    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Hexbin plot for density
    hb = plt.hexbin(data_P, data_time, gridsize=50, cmap='Blues', mincnt=1)
    cb = plt.colorbar(hb, label='Count')
    
    # Trend line (Polynomial degree 2)
    if len(data_P) > 1:
        try:
            z = np.polyfit(data_P, data_time, 2)
            p = np.poly1d(z)
            x_range = np.linspace(min(data_P), max(data_P), 100)
            plt.plot(x_range, p(x_range), "r--", linewidth=3, label='Trend Curve (Poly fit)')
            
            # Linear fit for comparison
            z1 = np.polyfit(data_P, data_time, 1)
            p1 = np.poly1d(z1)
            plt.plot(x_range, p1(x_range), "g-.", linewidth=2, label='Linear Trend')
        except Exception as e:
            print(f"Error calculating trend line: {e}")

    plt.title(f'Relationship between P-value and Battery Life (n={len(data_P)})', fontsize=14)
    plt.xlabel('P (Weighted Usage Ratio)', fontsize=12)
    plt.ylabel('Total Time (minutes)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    output_filename = 'P_vs_Time_Curve_HighRes_Final.png'
    output_path = os.path.join(current_dir, output_filename)
    plt.savefig(output_path, dpi=300)
    print(f"Graph saved to {output_path}")
    
    # Show plot if environment supports it
    # plt.show()

if __name__ == "__main__":
    main()
