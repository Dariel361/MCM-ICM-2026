import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 将当前目录添加到路径中以确保能导入同级模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from main import BatteryModel, T_Model, load_parameters
    from the_P_load import PowerLoadModel
except ImportError:
    # 兼容可能的文件名大小写问题
    from main import BatteryModel, T_Model, load_parameters
    from the_P_load import PowerLoadModel

def run_simulation(seed):
    """
    运行单次仿真并返回时间(分钟)和SOC(%)数据
    """
    # 加载参数
    all_params = load_parameters()
    sim_conf = all_params['simulation']
    
    # 配置负载模型（使用特定种子）
    load_params = all_params.get('load_model', {}).copy()
    load_params['seed'] = seed
    
    # 初始化各个模型
    battery = BatteryModel(all_params['electrochemical'])
    thermal = T_Model(all_params['thermal'], T_init=sim_conf['initial_temp'])
    load_model = PowerLoadModel(load_params)
    
    dt = sim_conf['dt']
    T_amb = sim_conf['T_amb']
    
    # 记录数据的列表
    times_min = []
    soc_percent = []
    
    t = 0.0
    
    while True:
        # 每隔60秒（1分钟）记录一次数据
        if int(t) % 60 == 0:
            times_min.append(t / 60.0)
            soc_percent.append(battery.state[0] * 100.0)
            
        # 1. 获取当前负载功率
        # 注意：calculate_P_load 内部会根据 t 重新模拟随机过程
        P_load, _, _ = load_model.calculate_P_load(t)
        
        # 2. 获取热状态和内阻
        T_curr = thermal.T
        R0_curr = battery.get_current_R0(T_curr)
        
        # 3. 计算电流
        I_curr, is_collapsed = battery.solve_current_from_power(P_load, R0_curr)
        
        # 4. 检查终止条件
        if is_collapsed: # 电压崩塌
            break
            
        if battery.state[0] <= 0.0: # 电量完全耗尽
            break
            
        # 5. 更新状态
        battery.update_state(dt, I_curr, T_curr)
        thermal.update_temperature(dt, I_curr, battery.get_current_R0, T_amb)
        
        t += dt
        
        # 安全终止（防止死循环，例如超过24小时）
        if t > 24 * 3600:
            break
            
    return times_min, soc_percent

def main():
    # 定义5个不同的随机种子
    seeds = [42, 101, 2024, 777, 9999]
    
    # 设置绘图风格和大小
    plt.figure(figsize=(12, 7))
    plt.rcParams['font.family'] = 'sans-serif'
    
    print("Starting simulations...")
    for seed in seeds:
        print(f"-> Simulating scenario with seed: {seed}")
        t_data, soc_data = run_simulation(seed)
        plt.plot(t_data, soc_data, label=f'Scenario (Seed {seed})', linewidth=2, alpha=0.8)
    
    # 设置英文标签和标题
    plt.title('Battery SOC Depletion Over Time (Different Usage Scenarios)', fontsize=16, pad=15)
    plt.xlabel('Time (Minutes)', fontsize=14)
    plt.ylabel('State of Charge (SOC) [%]', fontsize=14)
    
    # 设置坐标轴范围
    plt.xlim(left=0)
    plt.ylim(0, 100)
    
    # 添加图例和网格
    plt.legend(fontsize=12, loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 紧凑布局并保存
    plt.tight_layout()
    output_filename = 'SOC_Comparison_Curves.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Graph saved as {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()
