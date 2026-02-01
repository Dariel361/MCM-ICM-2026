import matplotlib.pyplot as plt
import numpy as np
from main import BatteryModel, T_Model, load_parameters

def run_and_draw():
    # 1. 加载参数
    all_params = load_parameters()
    sim_conf = all_params['simulation']
    
    # 2. 初始化模型
    battery = BatteryModel(all_params['electrochemical'])
    thermal = T_Model(all_params['thermal'], T_init=sim_conf['initial_temp'])
    
    # 3. 实验设定
    P_load = 0.2  # 恒定功率
    dt = sim_conf['dt']
    T_amb = sim_conf['T_amb']
    
    # 数据记录容器
    time_list = []
    soc_list = []
    volt_list = []
    curr_list = []
    
    t = 0.0
    
    # 仿真循环
    while True:
        # A. 获取热状态和电阻
        T_curr = thermal.T
        R0_curr = battery.get_current_R0(T_curr)
        
        # B. 根据功率计算电流
        I_curr, is_collapsed = battery.solve_current_from_power(P_load, R0_curr)
        
        # 崩塌检查
        if is_collapsed:
            print(f"Voltage collapse at t={t}")
            break
            
        # C. 计算端电压
        v_term = battery.get_U_L(I_curr, R0_curr)
        
        # D. 记录数据
        time_list.append(t)
        soc_list.append(battery.state[0])
        volt_list.append(v_term)
        curr_list.append(I_curr)
        
        # E. 截止条件
        # 1. 电压截止 (V < 0)
        if v_term < 0:
            print(f"Cutoff voltage (< 0V) reached at t={t:.2f}s")
            break
        
        # 2. SOC 耗尽 (SOC <= 0.03)
        if battery.state[0] <= 0.03:
            print("SOC depleted (<= 0.03).")
            break
            
        # F. 更新状态
        battery.update_state(dt, I_curr, T_curr)
        thermal.update_temperature(dt, I_curr, battery.get_current_R0, T_amb)
        
        t += dt

    # 4. 数据处理与绘图
    times = np.array(time_list)
    socs = np.array(soc_list)
    volts = np.array(volt_list)
    currs = np.array(curr_list)
    
    # 将时间转换为天
    days = times / (24 * 3600.0)
    
    # 创建画布
    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(right=0.85) # 调整右边距以容纳第三个轴
    
    # --- 左轴 SOC ---
    color_soc = 'tab:blue'
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('SOC', color=color_soc, fontsize=12)
    ax1.plot(days, socs, color=color_soc, linewidth=2, label='SOC')

    # 添加 SOC=0.03 的虚线
    ax1.axhline(y=0.03, color=color_soc, linestyle='--', linewidth=1.5, label='Min SOC (0.03)')

    ax1.tick_params(axis='y', labelcolor=color_soc)
    ax1.set_ylim(0, 1.05)  # 限制SOC范围展示
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --- 右轴 Voltage ---
    ax2 = ax1.twinx()  # 共享x轴
    color_vol = 'tab:red'
    ax2.set_ylabel('Voltage (V)', color=color_vol, fontsize=12)
    ax2.plot(days, volts, color=color_vol, linewidth=2, label='Voltage')
    ax2.tick_params(axis='y', labelcolor=color_vol)
    
    # 让刻度更紧凑以突显变化
    v_min, v_max = np.min(volts), np.max(volts)
    padding = (v_max - v_min) * 0.1
    ax2.set_ylim(v_min - padding, v_max + padding)

    # --- 第三轴 Current ---
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1)) # 将轴向右偏移
    color_curr = 'tab:green'
    ax3.set_ylabel('Current (A)', color=color_curr, fontsize=12)
    ax3.plot(days, currs, color=color_curr, linewidth=2, linestyle='-.', label='Current')
    ax3.tick_params(axis='y', labelcolor=color_curr)
    
    # 图例设置 (合并三轴图例)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2 + lines_3, labels_1 + labels_2 + labels_3, loc='center left')

    plt.title(f'Battery Discharge Simulation', fontsize=14)
    # plt.tight_layout() # 与 subplots_adjust 冲突，故注释掉
    plt.show()

if __name__ == "__main__":
    run_and_draw()
