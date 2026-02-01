import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    csv_file = 'aging_simulation_results.csv'
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    # Read data
    df = pd.read_csv(csv_file)
    ages = df['Age_Months']
    runtimes_min = df['Runtime_Minutes']
    
    # Convert runtime to hours
    runtimes_hour = runtimes_min / 60.0

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Using the same style as before: royalblue, line
    plt.plot(ages, runtimes_hour, linestyle='-', color='royalblue', linewidth=2)
    
    # Academic Title and Labels
    plt.title('Battery Operational Duration under Aging Effects', fontsize=14)
    plt.xlabel('Aging Time (months)', fontsize=12)
    plt.ylabel('Operational Duration (hours)', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Set Y-axis limits to add some headroom
    y_min, y_max = runtimes_hour.min(), runtimes_hour.max()
    plt.ylim(y_min - 0.5, y_max + 0.5)
    
    # X-axis ticks every 5 months
    plt.xticks(range(0, 61, 5))
    
    # Add a few key data points as annotations (start, mid, end)
    key_indices = [0, len(ages)//2, len(ages)-1]
    for i in key_indices:
        x = ages.iloc[i]
        y = runtimes_hour.iloc[i]
        # Adjust vertical offset based on position to avoid overlap
        offset = (0, -15) if i == 0 else (0, 10) 
        plt.annotate(f'{y:.2f}h', (x, y), textcoords="offset points", xytext=offset, ha='center', fontsize=10, fontweight='bold')

    output_path = 'battery_aging_analysis_hours.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print("-" * 40)
    print(f"Plot saved to {os.path.abspath(output_path)}")
    print(f"Start Duration: {runtimes_hour.iloc[0]:.2f} h")
    print(f"End Duration:   {runtimes_hour.iloc[-1]:.2f} h")

if __name__ == "__main__":
    main()
