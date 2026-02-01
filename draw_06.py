import numpy as np
import matplotlib.pyplot as plt
from main import BatteryModel, T_Model, load_parameters

# 尝试导入 PowerLoadModel，兼容不同的文件名大小写情况
try:
    from the_P_load import PowerLoadModel
except ImportError:
    from the_P_load import PowerLoadModel

def run_simulation(T_ambient):
    """
    运行指定环境温度下的仿真，返回电池续航时间（分钟）
    截止条件：SOC <= 0.03 或 电压崩塌
    """
    # 加载参数
    all_params = load_parameters()
    sim_conf = all_params['simulation']
    
    # 初始化模型
    battery = BatteryModel(all_params['electrochemical'])
    # 初始 SOC 默认为 1.0
    
    # 初始化热模型，初始温度设为环境温度
    thermal = T_Model(all_params['thermal'], T_init=T_ambient)
    
    load_model = PowerLoadModel(all_params.get('load_model'))
    
    dt = sim_conf['dt']
    T_amb = T_ambient  # 使用循环变量作为环境温度
    
    t = 0.0
    
    while True:
        # 1. 获取热状态和电阻
        T_curr = thermal.T
        R0_curr = battery.get_current_R0(T_curr)

        # 2. 计算当前动态负载功率
        P_load, _, _ = load_model.calculate_P_load(t)
        
        # 3. 根据功率计算电流
        I_curr, is_collapsed = battery.solve_current_from_power(P_load, R0_curr)
        
        # 4. 截止条件判断
        # 电压崩塌
        if is_collapsed:
            break
            
        # 电量耗尽 (SOC <= 0.03)
        if battery.state[0] <= 0.03:
            break

        # 5. 更新状态
        battery.update_state(dt, I_curr, T_curr)
        thermal.update_temperature(dt, I_curr, battery.get_current_R0, T_amb)
        
        t += dt
        
        # 安全中止（防止死循环，虽然物理上电量终会耗尽）
        if t > 3600 * 100: 
            break
            
    return t / 60.0

if __name__ == "__main__":
    # 温度范围 -60 到 60，步长为 2
    temperatures = np.arange(-60, 61, 2)
    life_times = []
    
    print("Simulating battery life across temperature range...")
    
    for T in temperatures:
        life = run_simulation(T)
        life_times.append(life)
        print(f"T = {T}°C, Life = {life:.2f} min")
        
    # 绘图
    plt.figure(figsize=(10, 6))
    
    # 设置英文字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False 
    
    plt.plot(temperatures, life_times, marker='.', linestyle='-', linewidth=2, color='blue')
    
    plt.xlabel('Ambient Temperature (Celsius)')
    plt.ylabel('Battery Life (Minutes)')
    plt.title('Battery Life vs. Ambient Temperature')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
