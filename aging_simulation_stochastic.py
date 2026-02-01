import numpy as np
import json
import sys
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # 用于显示进度条

# 确保可以导入同级目录的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from main import BatteryModel, T_Model, load_parameters
    from the_P_load import PowerLoadModel
    from SOH_model import SOHModel
except ImportError:
    print("Error: Could not import necessary modules (main, the_P_load, SOH_model).")
    sys.exit(1)

# === 性能优化：有状态的负载模型 ===
class FastPowerLoadModel:
    """
    一个优化的、有状态的负载模型。
    避免了 PowerLoadModel.calculate_P_load 每次从 t=0 开始推演的 O(t) 复杂度。
    """
    def __init__(self, base_model, seed):
        self.base_model = base_model
        # 使用独立的 RandomState 确保结果可复现且不干扰全局
        self.rng = np.random.RandomState(seed)
        
        self.current_state = 0  # 初始状态: standby
        self.current_time = 0.0
        self.next_jump_time = 0.0
        
        # 初始化第一次跳变
        self._schedule_next_jump()

    def _schedule_next_jump(self):
        # 计算在当前状态停留的时间
        rate_out = -self.base_model.Q[self.current_state, self.current_state]
        tau_hours = self.rng.exponential(1.0 / rate_out)
        tau_seconds = tau_hours * 3600
        self.next_jump_time = self.current_time + tau_seconds

    def get_power_at(self, t):
        # 推进状态直到覆盖时间 t
        while self.next_jump_time <= t:
            self.current_time = self.next_jump_time
            
            # 状态转移
            probs = self.base_model.Q[self.current_state, :].copy()
            probs[self.current_state] = 0
            probs = np.maximum(probs, 0)

            if probs.sum() > 0:
                probs = probs / probs.sum()
                next_state = self.rng.choice(5, p=probs)
            else:
                next_state = self.current_state
            
            self.current_state = next_state
            self._schedule_next_jump()

        # 计算当前时刻的功率 (逻辑复刻自 PowerLoadModel)
        state_name = self.base_model.states[self.current_state]
        mean_params = self.base_model.params_mean[state_name]

        params = []
        for i, mean_val in enumerate(mean_params):
            # 注意：原模型中是在每次调用 calculate_P_load 时随机生成噪声
            # 这里我们也保持每一帧都有随机噪声
            noise = 0.2 * mean_val * self.rng.randn()
            if i == 3:  # GPS
                val = 1.0 if mean_val > 0.5 else 0.0
            else:
                val = max(mean_val + noise, 0.05)
            params.append(val)

        L, cpu_usage, data_rate, gps_active = params

        P_screen = 1.5 * (L ** 2.2)
        P_cpu = 0.1 + 1.9 * (cpu_usage ** 1.3)
        P_com = 0.2 + 0.5 * data_rate / 0.4
        P_gps = 0.3 if gps_active > 0.5 else 0.0
        P_background = 0.15

        P_foreground = P_screen + P_cpu + P_com + P_gps
        P_load = P_background + P_foreground
        
        return P_load

def run_single_day_simulation(age_months, seed, all_params):
    """
    运行单次单日仿真，返回续航时间（分钟）
    """
    # 1. 设置电池老化状态
    soh_params = all_params.get('soh_model', {})
    soh_model = SOHModel(soh_params)
    current_soh = soh_model.calculate_SOH(age_months)
    
    # 2. 初始化模型
    battery = BatteryModel(all_params['electrochemical'], current_soh=current_soh)
    thermal = T_Model(all_params['thermal'], T_init=all_params['simulation']['initial_temp'])
    
    # 3. 初始化优化后的负载模型
    base_load_model = PowerLoadModel() # 获取参数配置用
    fast_load_model = FastPowerLoadModel(base_load_model, seed)
    
    # 4. 优化仿真步长
    # 电池充放电对于秒级变化不敏感，使用 10s 或 30s 步长可以极大加速
    dt = 10.0 
    T_amb = all_params['simulation']['T_amb']
    
    t = 0.0
    max_time = 24 * 3600 * 1.5 
    
    while t < max_time:
        # A. 获取热状态和电阻
        T_curr = thermal.T
        R0_curr = battery.get_current_R0(T_curr)

        # B. 计算负载 (使用快速模型)
        P_load = fast_load_model.get_power_at(t)
        
        # C. 计算电流
        I_curr, is_collapsed = battery.solve_current_from_power(P_load, R0_curr)
        
        # D. 截止条件判断
        if is_collapsed:
            break
            
        v_term = battery.get_U_L(I_curr, R0_curr)
        
        if v_term < 0: # 电压截止
            break
        
        if battery.state[0] <= 0.03: # SOC 截止
            break
            
        # E. 更新状态
        battery.update_state(dt, I_curr, T_curr)
        thermal.update_temperature(dt, I_curr, battery.get_current_R0, T_amb)
        
        t += dt
        
    return t / 60.0 # 返回分钟

def main():
    print("Starting Stochastic Aging Simulation (Monte Carlo)...")
    print("Optimization: Using FastPowerLoadModel and dt=10.0s")
    
    all_params = load_parameters()
    
    # 仿真设置 - 减少点数以加快演示速度
    age_start = 0
    age_end = 60
    age_step = 0.2  # 每2个月一个点
    simulations_per_point = 10  # 每个点取10次平均 (如果还慢可改为5)
    
    ages = np.arange(age_start, age_end + 0.1, age_step)
    
    # CSV 输出文件
    output_csv = "aging_simulation_stochastic.csv"
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Age_Months", "Avg_Runtime_Minutes", "Std_Dev_Minutes"])
        
        # 使用 tqdm 显示总体进度
        for age in tqdm(ages, desc="Simulating Aging", unit="point"):
            runtimes = []
            
            # 蒙特卡洛循环
            for i in range(simulations_per_point):
                day_seed = 1000 + i + int(age * 100) # 确保随机性与Age相关又不同
                runtime = run_single_day_simulation(age, day_seed, all_params)
                runtimes.append(runtime)
            
            # 计算统计量
            avg_runtime = np.mean(runtimes)
            std_dev = np.std(runtimes)
            
            writer.writerow([age, avg_runtime, std_dev])
            f.flush()

    print(f"\nSimulation completed. Data saved to {output_csv}")
    
    # 自动绘图
    plot_results(output_csv)

def plot_results(csv_file):
    print("Plotting results...")
    df = pd.read_csv(csv_file)
    
    # 分钟转小时
    df['Runtime_Hours'] = df['Avg_Runtime_Minutes'] / 60.0
    
    plt.figure(figsize=(10, 6))
    
    # 1. 绘制置信区间 (作为背景，淡一点)
    std_hours = df['Std_Dev_Minutes'] / 60.0
    plt.fill_between(df['Age_Months'], 
                     df['Runtime_Hours'] - std_hours, 
                     df['Runtime_Hours'] + std_hours, 
                     color='#88CCEE', alpha=0.15, label='Usage Variance (Std Dev)')
    
    # 2. 绘制散点 (变小变细，不连线)
    # s=15 控制点的大小，alpha=0.6 控制透明度
    plt.scatter(df['Age_Months'], df['Runtime_Hours'], 
                color='#4477AA', s=15, alpha=0.6, edgecolors='none', label='Simulated Data Points')
    
    # 3. 拟合趋势线 (使用3次多项式拟合出平滑曲线)
    if len(df) > 3:
        # 拟合
        z = np.polyfit(df['Age_Months'], df['Runtime_Hours'], 3)
        p = np.poly1d(z)
        
        # 生成平滑的x轴数据用于绘制曲线
        x_smooth = np.linspace(df['Age_Months'].min(), df['Age_Months'].max(), 200)
        y_smooth = p(x_smooth)
        
        plt.plot(x_smooth, y_smooth, color='#CC3311', linewidth=2.5, label='Fitted Trend Line')
    
    plt.title('Battery Life Degradation Analysis', fontsize=14)
    plt.xlabel('Aging Time (Months)', fontsize=12)
    plt.ylabel('Operational Duration (Hours)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    output_img = "aging_simulation_stochastic_plot.png"
    plt.savefig(output_img, dpi=300)
    print(f"Plot saved to {output_img}")
    # plt.show()

if __name__ == "__main__":
    main()
