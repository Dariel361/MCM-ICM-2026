import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import json
from copy import deepcopy

# Add current directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from main import BatteryModel, load_parameters
except ImportError:
    # Fallback if running from a different directory
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Model'))
    from main import BatteryModel, load_parameters

# Set plotting style for better visualization
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    pass

class SensitivityAnalyzer:
    def __init__(self, params_file='parameters.json'):
        self.base_params = load_parameters(params_file)
        self.elec_params = self.base_params['electrochemical']
        self.sim_params = self.base_params['simulation']
        
        # Baseline settings
        self.dt = 1.0
        self.T_fixed = 25.0  # Fixed temperature as requested
        self.cutoff_voltage = 3.0
        self.nominal_capacity_Ah = self.elec_params['Cn'] / 3600.0
        self.discharge_current_1C = self.nominal_capacity_Ah * 1.0 # 1C current (Amps)

    def run_discharge_simulation(self, battery_model, current_A=None, current_noise_std=0.0):
        """
        Runs a constant current discharge simulation.
        Returns: time_series, voltage_series, soc_series, eod_time
        """
        if current_A is None:
            current_A = self.discharge_current_1C
            
        t = 0.0
        times = []
        voltages = []
        socs = []
        
        # Reset state just in case
        battery_model.state = np.array([1.0, 0.0, 0.0, 0.0]) # SOC=1.0, others=0
        
        max_time = 7200 # Safety limit (2 hours for 1C is plenty)
        
        while t < max_time:
            # Apply current noise if requested (Robustness test)
            I_actual = current_A
            if current_noise_std > 0:
                I_actual += np.random.normal(0, current_noise_std)
            
            # 1. Get R0 at fixed T
            R0 = battery_model.get_current_R0(self.T_fixed)
            
            # 2. Calculate Voltage
            V_term = battery_model.get_U_L(I_actual, R0)
            
            times.append(t)
            voltages.append(V_term)
            socs.append(battery_model.state[0])
            
            # Cutoff conditions
            if V_term < self.cutoff_voltage:
                # Linear interpolation for exact time
                if len(voltages) >= 2:
                    v_prev = voltages[-2]
                    t_prev = times[-2]
                    # v_prev > cutoff > V_term
                    ratio = (v_prev - self.cutoff_voltage) / (v_prev - V_term)
                    t_exact = t_prev + ratio * self.dt
                    return np.array(times), np.array(voltages), np.array(socs), t_exact
                else:
                    return np.array(times), np.array(voltages), np.array(socs), t
            
            if battery_model.state[0] <= 0.0:
                 # Linear interpolation for SOC
                if len(socs) >= 2:
                    s_prev = socs[-2]
                    t_prev = times[-2]
                    ratio = (s_prev - 0.0) / (s_prev - battery_model.state[0])
                    t_exact = t_prev + ratio * self.dt
                    return np.array(times), np.array(voltages), np.array(socs), t_exact
                else:
                    return np.array(times), np.array(voltages), np.array(socs), t
                
            # 3. Update State
            battery_model.update_state(self.dt, I_actual, self.T_fixed)
            t += self.dt
            
        return np.array(times), np.array(voltages), np.array(socs), t

    def run_oat_sensitivity(self, param_names, variations=[0.8, 0.9, 1.0, 1.1, 1.2]):
        """
        Runs One-at-a-Time sensitivity analysis.
        param_names: list of parameter keys to test (e.g., ['Cn', 'R1', ...])
        variations: multipliers to apply to the base value
        """
        results = []
        
        print(f"--- Starting OAT Sensitivity Analysis ---")
        print(f"Parameters: {param_names}")
        print(f"Variations: {variations}")
        
        # Baseline run
        base_battery = BatteryModel(self.elec_params)
        _, base_v, _, base_eod = self.run_discharge_simulation(base_battery)
        base_avg_v = np.mean(base_v)
        
        print(f"Baseline: EOD={base_eod:.4f}s, AvgV={base_avg_v:.4f}V")

        for param in param_names:
            for mult in variations:
                if mult == 1.0:
                    continue
                    
                # Create modified parameters
                mod_params = deepcopy(self.elec_params)
                
                # Special handling for R0 which is composed of a1, c1
                if param == 'R0_factor':
                    mod_params['a1'] *= mult
                    mod_params['c1'] *= mult
                else:
                    # Robust key matching (case insensitive attempt if exact key fails)
                    if param not in mod_params:
                        # Try to find case-insensitive match
                        keys = list(mod_params.keys())
                        found = False
                        for k in keys:
                            if k.lower() == param.lower():
                                mod_params[k] *= mult
                                found = True
                                break
                        if not found:
                            print(f"Warning: Parameter {param} not found in model parameters!")
                    else:
                        mod_params[param] *= mult
                
                # Run Simulation
                model = BatteryModel(mod_params)
                _, v_series, _, eod_time = self.run_discharge_simulation(model)
                avg_v = np.mean(v_series)
                
                # Calculate metrics relative to baseline
                eod_change_pct = (eod_time - base_eod) / base_eod * 100
                v_avg_change_pct = (avg_v - base_avg_v) / base_avg_v * 100
                
                results.append({
                    'Parameter': param,
                    'Multiplier': mult,
                    'EOD_Time': eod_time,
                    'Avg_Voltage': avg_v,
                    'EOD_Change_Pct': eod_change_pct,
                    'V_Avg_Change_Pct': v_avg_change_pct
                })
                
        return pd.DataFrame(results)

    def run_structural_robustness(self):
        """
        Tests robustness against OCV curve perturbations.
        """
        print("\n--- Starting Structural Robustness Test (OCV Perturbation) ---")
        
        class PerturbedOCVBattery(BatteryModel):
            def __init__(self, params, perturbation_amp=0.0):
                super().__init__(params)
                self.perturbation_amp = perturbation_amp
                
            def get_U_OC(self, soc):
                base_ocv = super().get_U_OC(soc)
                # Add sinusoidal perturbation
                noise = self.perturbation_amp * np.sin(20 * np.pi * soc)
                return base_ocv + noise

        perturbations = [0.0, 0.01, 0.05] # Volts
        results = {}
        
        plt.figure(figsize=(10, 6))
        
        for amp in perturbations:
            model = PerturbedOCVBattery(self.elec_params, perturbation_amp=amp)
            times, voltages, socs, eod = self.run_discharge_simulation(model)
            
            label = f"Perturbation +/-{amp}V"
            plt.plot(times, voltages, label=label, alpha=0.8 if amp > 0 else 1.0, linewidth=1.5)
            
            results[amp] = {
                'EOD_Time': eod,
                'Voltage_Std': np.std(np.diff(voltages)) # Measure smoothness
            }
            print(f"Perturbation {amp}V: EOD={eod:.2f}s")
            
        plt.title("Robustness: Discharge Curves with OCV Perturbation")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.legend()
        plt.grid(True)
        plt.savefig("robustness_ocv_perturbation.png")
        print("Saved plot to robustness_ocv_perturbation.png")
        return results

    def run_input_noise_robustness(self):
        """
        Tests robustness against current sensor noise.
        """
        print("\n--- Starting Input Noise Robustness Test ---")
        
        noise_levels = [0.0, 0.1, 0.5] # Amps (Standard Deviation)
        
        plt.figure(figsize=(10, 6))
        
        for sigma in noise_levels:
            model = BatteryModel(self.elec_params)
            times, voltages, socs, eod = self.run_discharge_simulation(model, current_noise_std=sigma)
            
            # Smooth plot for high noise to see trend
            alpha = 1.0 if sigma == 0 else 0.7
            plt.plot(times, voltages, label=f"Noise Sigma={sigma}A", alpha=alpha, linewidth=1)
            
            print(f"Current Noise {sigma}A: EOD={eod:.2f}s")
            
        plt.title("Robustness: Discharge Curves with Current Noise")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.legend()
        plt.grid(True)
        plt.savefig("robustness_current_noise.png")
        print("Saved plot to robustness_current_noise.png")

    def run_extreme_cases(self):
        """
        Tests corner cases to ensure no crashes.
        """
        print("\n--- Starting Extreme Case Testing ---")
        
        cases = [
            {'name': 'High Impedance', 'mod': {'R0_factor': 5.0, 'R1': 5.0}},
            {'name': 'Low Capacity', 'mod': {'Cn': 0.5}},
            {'name': 'Fast Dynamics', 'mod': {'C1': 0.1, 'C2': 0.1}},
            {'name': 'Slow Dynamics', 'mod': {'C1': 10.0, 'C2': 10.0}}
        ]
        
        for case in cases:
            mod_params = deepcopy(self.elec_params)
            mods = case['mod']
            
            # Apply mods
            if 'R0_factor' in mods:
                factor = mods.pop('R0_factor')
                mod_params['a1'] *= factor
                mod_params['c1'] *= factor
            
            for k, v in mods.items():
                if k in mod_params:
                    mod_params[k] *= v
                
            model = BatteryModel(mod_params)
            try:
                _, _, _, eod = self.run_discharge_simulation(model)
                print(f"Case [{case['name']}]: PASSED (EOD={eod:.2f}s)")
            except Exception as e:
                print(f"Case [{case['name']}]: FAILED - {str(e)}")

    def plot_sensitivity_tornado(self, df_results):
        """
        Generates a Tornado chart for sensitivity.
        """
        # Focus on +/- 20% (Multiplier 0.8 and 1.2)
        subset = df_results[df_results['Multiplier'].isin([0.8, 1.2])].copy()
        
        if subset.empty:
            print("No data for Tornado chart (checking 0.8/1.2 multipliers)")
            return

        # Pivot to get Low (0.8) and High (1.2) impact for EOD
        pivot = subset.pivot(index='Parameter', columns='Multiplier', values='EOD_Change_Pct')
        
        # Calculate range
        pivot['Range'] = abs(pivot[1.2] - pivot[0.8])
        pivot = pivot.sort_values('Range', ascending=True)
        
        plt.figure(figsize=(10, 6))
        
        y = np.arange(len(pivot))
        width = 0.4
        
        # Plot bars
        plt.barh(y, pivot[1.2], height=width, label='+20% Param', color='skyblue')
        plt.barh(y, pivot[0.8], height=width, label='-20% Param', color='salmon')
        
        plt.yticks(y, pivot.index)
        plt.axvline(0, color='black', linewidth=0.8)
        plt.xlabel("Change in EOD Time (%)")
        plt.title("Sensitivity Tornado Chart: Impact on Discharge Time")
        plt.legend()
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("sensitivity_tornado_eod.png")
        print("Saved plot to sensitivity_tornado_eod.png")

    def plot_sensitivity_heatmap(self, df_results):
        """
        Generates a Heatmap for sensitivity analysis.
        Shows the percentage change in EOD time for different parameter multipliers.
        """
        print("\nGenerating Sensitivity Heatmap...")
        try:
            import seaborn as sns
        except ImportError:
            print("Seaborn not installed, skipping heatmap.")
            return

        # Pivot data: Rows=Parameters, Cols=Multipliers, Values=EOD_Change_Pct
        pivot_table = df_results.pivot(index='Parameter', columns='Multiplier', values='EOD_Change_Pct')
        
        # Sort parameters by maximum impact (range)
        ranges = pivot_table.max(axis=1) - pivot_table.min(axis=1)
        pivot_table = pivot_table.loc[ranges.sort_values(ascending=False).index]

        plt.figure(figsize=(10, 8))
        
        # Draw Heatmap
        # RdBu colormap: Red for negative change (worse battery life), Blue for positive (better)
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="RdBu", center=0,
                    linewidths=.5, cbar_kws={'label': 'Change in Runtime (%)'})
        
        plt.title("Parameter Sensitivity Heatmap: Impact on Battery Runtime", fontsize=14, pad=15)
        plt.xlabel("Parameter Multiplier (Variation)", fontsize=12)
        plt.ylabel("Parameters", fontsize=12)
        plt.tight_layout()
        
        output_file = "sensitivity_heatmap.png"
        plt.savefig(output_file, dpi=300)
        print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    analyzer = SensitivityAnalyzer()
    
    # 1. OAT Sensitivity
    # Expanded parameter list to include R2, C2, kd, etc.
    params_to_test = ['Cn', 'R0_factor', 'R1', 'C1', 'R2', 'C2', 'tau_d', 'kd']
    df_res = analyzer.run_oat_sensitivity(params_to_test)
    print("\nSensitivity Results (Top 5):")
    print(df_res.head())
    df_res.to_csv("sensitivity_results.csv", index=False)
    
    # Generate Visualizations
    analyzer.plot_sensitivity_tornado(df_res)
    analyzer.plot_sensitivity_heatmap(df_res)
    
    # 2. Structural Robustness
    analyzer.run_structural_robustness()
    
    # 3. Input Noise Robustness
    analyzer.run_input_noise_robustness()
    
    # 4. Extreme Cases
    analyzer.run_extreme_cases()
    
    print("\n=== All Analysis Completed Successfully ===")
