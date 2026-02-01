import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from main import BatteryModel, T_Model, load_parameters
# 尝试导入 PowerLoadModel，兼容不同的文件名大小写情况
try:
    from the_P_load import PowerLoadModel
except ImportError:
    from the_P_load import PowerLoadModel

def run_simulation(T_ambient, initial_soc=1.0):
    """
    运行单次仿真并返回时间（分钟）、SOC列表和状态列表
    """
    # 加载参数
    all_params = load_parameters()
    sim_conf = all_params['simulation']
    
    # 初始化模型
    battery = BatteryModel(all_params['electrochemical'])
    battery.state[0] = initial_soc  # 设置初始SOC
    
    # 使用目标环境温度初始化电池温度和热模型
    thermal = T_Model(all_params['thermal'], T_init=T_ambient)
    load_model = PowerLoadModel(all_params.get('load_model'))
    
    dt = sim_conf['dt']
    T_amb = T_ambient # 使用设定的环境温度
    
    time_minutes = []
    soc_values = []
    state_history = []
    
    t = 0.0
    
    while True:
        # 1. 获取热状态和电阻
        T_curr = thermal.T
        R0_curr = battery.get_current_R0(T_curr)

        # 2. 计算当前动态负载功率 (获取状态)
        P_load, state_name, _ = load_model.calculate_P_load(t)
        
        # 每分钟记录一次数据，减少绘图点数
        if int(t) % 60 == 0:
            time_minutes.append(t / 60.0)
            soc_values.append(battery.state[0])
            state_history.append(state_name)
            
        # 3. 根据功率计算电流
        I_curr, is_collapsed = battery.solve_current_from_power(P_load, R0_curr)
        
        # 4. 终止条件检测
        if is_collapsed:
            break
            
        if battery.state[0] <= 0.03:
            # 记录最后一点 (如果尚未记录)
            if not time_minutes or abs(time_minutes[-1] - t / 60.0) > 1e-6:
                time_minutes.append(t / 60.0)
                soc_values.append(battery.state[0])
                state_history.append(state_name)
            break

        # 5. 更新状态
        battery.update_state(dt, I_curr, T_curr)
        thermal.update_temperature(dt, I_curr, battery.get_current_R0, T_amb)
        
        t += dt
        
    return time_minutes, soc_values, state_history

if __name__ == "__main__":
    temperatures = [-50, -15, 0, 10, 25, 50]
    initial_soc = 1.0
    
    plt.figure(figsize=(12, 7))
    
    # 设置英文字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False 

    # 定义状态颜色 (加深颜色以提高区分度)
    state_colors = {
        'standby': '#90a4ae',      # Blue Grey
        'social': '#29b6f6',    # Light Blue 
        'game': '#ff5252',      # Red Accent
        'video': '#fdd835',     # Yellow 
        'nav': '#66bb6a'        # Green 
    }

    longest_run = {'t': [], 'states': []}

    for T in temperatures:
        print(f"Simulating Temperature: {T}°C")
        t_hist, soc_hist, states = run_simulation(T_ambient=T, initial_soc=initial_soc)
        
        # 所有温度曲线都改为实线
        linestyle = '-'
        
        if T == 50:
            plt.plot(t_hist, soc_hist, label=f'T = {T} °C', linewidth=2, color='white', linestyle=linestyle)
        else:
            plt.plot(t_hist, soc_hist, label=f'T = {T} °C', linewidth=2, linestyle=linestyle)
        
        # 记录最长的一次仿真用于绘制背景
        if len(t_hist) > len(longest_run['t']):
            longest_run['t'] = t_hist
            longest_run['states'] = states
            
    # 添加 SOC=0.03 截止线
    plt.axhline(y=0.03, color='#555555', linestyle='--', linewidth=1.5, label='Cutoff SOC = 0.03')
        
    # 绘制背景状态区域
    t_arr = longest_run['t']
    s_arr = longest_run['states']
    
    if t_arr:
        start_idx = 0
        for i in range(1, len(t_arr)):
            if s_arr[i] != s_arr[start_idx]:
                end_idx = i
                # 绘制区间 (alpha 稍微调高)
                plt.axvspan(t_arr[start_idx], t_arr[end_idx], 
                            facecolor=state_colors.get(s_arr[start_idx], 'white'), 
                            alpha=0.4, edgecolor=None)
                start_idx = i
        # 绘制最后一段
        plt.axvspan(t_arr[start_idx], t_arr[-1], 
                    facecolor=state_colors.get(s_arr[start_idx], 'white'), 
                    alpha=0.4, edgecolor=None)

    plt.xlabel('Time (Minutes)')
    plt.ylabel('SOC')
    plt.title(f'SOC Curve at Different Ambient Temperatures (Initial SOC={initial_soc:.0%})')
    
    # 图例1：温度曲线 (固定在右上角)
    legend1 = plt.legend(loc='upper right', bbox_to_anchor=(1, 1), title="Temperature", framealpha=0.9)
    plt.gca().add_artist(legend1)
    
    # 图例2：负载状态 (移动到右上角，位于 温度 图例下方)
    state_patches = [Patch(facecolor=color, edgecolor='none', alpha=0.4, label=state)
                     for state, color in state_colors.items()]
    plt.legend(handles=state_patches, loc='upper right', bbox_to_anchor=(1, 0.73), title="Active Component", ncol=1, framealpha=0.9)

    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()
