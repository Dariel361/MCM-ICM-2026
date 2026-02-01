import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def simulate_continuous_power(duration_hours=24.0, dt=1.0, seed=None):
    """
    生成连续、平滑的智能手机功率消耗轨迹
    duration_hours: 模拟时长(小时)
    dt: 采样时间步长(秒)
    """
    if seed is None:
        seed = 42
    np.random.seed(seed)
    
    # === 1. 模型参数定义 ===
    states = ['idle', 'social', 'game', 'video', 'nav']
    state_indices = {name: i for i, name in enumerate(states)}
    
    # 转移速率矩阵 Q (次/小时)
    Q = np.array([
        [-2.0, 1.0, 0.3, 0.5, 0.2],  # idle
        [0.8, -3.0, 0.5, 1.2, 0.5],  # social
        [0.2, 0.3, -1.5, 0.5, 0.5],  # game
        [0.5, 0.8, 0.4, -2.5, 0.8],  # video
        [0.3, 0.4, 0.3, 0.5, -1.5]   # nav
    ])
    
    # 各状态参数均值 [L, cpu, data, gps_prob]
    params_mean = np.array([
        [0.15, 0.10, 0.05, 0.05],  # idle
        [0.40, 0.30, 0.30, 0.10],  # social
        [0.85, 0.75, 0.50, 0.01],  # game
        [0.70, 0.40, 2.00, 0.05],  # video
        [0.80, 0.50, 0.20, 1.00]   # nav
    ])
    
    # === 2. 生成状态序列 (连续时间马尔可夫链) ===
    total_seconds = int(duration_hours * 3600)
    num_steps = int(total_seconds / dt)
    times = np.linspace(0, duration_hours, num_steps)
    
    # 预分配参数数组 (时间步数, 参数数量)
    # 参数顺序: 0:L, 1:CPU, 2:Data, 3:GPS_active
    param_traces = np.zeros((num_steps, 4))
    
    current_state = 0 # Start at idle
    current_time = 0.0
    
    # 记录每个时间点的状态索引
    state_trace = np.zeros(num_steps, dtype=int)
    
    # 快速生成状态段
    segments = []
    while current_time < duration_hours:
        rate_out = -Q[current_state, current_state]
        dwell_time = np.random.exponential(1.0 / rate_out) # hours
        
        segments.append((current_time, current_time + dwell_time, current_state))
        
        current_time += dwell_time
        
        # 转移
        probs = Q[current_state, :].copy()
        probs[current_state] = 0
        if probs.sum() > 0:
            probs /= probs.sum()
            current_state = np.random.choice(len(states), p=probs)
            
    # 将状态段映射到时间网格
    # 这是一个矢量化操作，比逐点循环快得多
    for start, end, state_idx in segments:
        # 找到对应的时间索引范围
        start_idx = int(start * 3600 / dt)
        end_idx = int(end * 3600 / dt)
        
        # 边界处理
        start_idx = max(0, start_idx)
        end_idx = min(num_steps, end_idx)
        
        if start_idx < end_idx:
            # 基础值
            base_vals = params_mean[state_idx]
            
            # === 3. 添加连续噪声 (关键步骤) ===
            # 为该段生成与长度匹配的随机噪声，但不是独立的，而是有一定相关性
            segment_len = end_idx - start_idx
            
            # 生成白噪声
            noise = np.random.randn(segment_len, 4) * 0.1 * base_vals # 噪声强度为均值的10%
            
            # 填充基础值 + 噪声
            param_traces[start_idx:end_idx, :] = base_vals + noise
            state_trace[start_idx:end_idx] = state_idx

    # GPS 特殊处理：它通常是二值的 (开/关)
    # 我们让 GPS 概率 > 0.5 的时候开启，但为了平滑，我们在后面会对功率进行平滑
    gps_prob_trace = param_traces[:, 3]
    gps_active_trace = (gps_prob_trace > 0.5).astype(float)
    
    # === 4. 全局平滑 (Make it continuous!) ===
    # 使用高斯滤波器对参数曲线进行平滑，模拟状态切换时的过渡
    # sigma 决定了平滑程度，数值越大越平滑
    sigma_sec = 300 # 300秒的平滑窗口
    sigma_points = sigma_sec / dt
    
    # 对前三个参数(L, CPU, Data)进行平滑
    for i in range(3):
        param_traces[:, i] = gaussian_filter1d(param_traces[:, i], sigma=sigma_points)
        
    # 对 GPS 信号也进行轻微平滑，使其看起来像是有“热身”过程
    gps_active_trace = gaussian_filter1d(gps_active_trace, sigma=sigma_points/2)
    
    # 确保参数非负
    param_traces = np.maximum(param_traces, 0.01)
    
    # === 5. 计算功率 (基于平滑后的参数) ===
    L = param_traces[:, 0]
    cpu_usage = param_traces[:, 1]
    data_rate = param_traces[:, 2]
    
    # 功率公式 (与原代码一致)
    P_screen = 1.5 * (L ** 2.2)
    P_cpu = 0.1 + 1.9 * (cpu_usage ** 1.3)
    P_com = 0.2 + 0.5 * (data_rate / 0.4)
    P_gps = 0.3 * gps_active_trace # GPS功率随激活程度变化
    P_bg = np.full(num_steps, 0.15) # Background constant
    
    # 再次对最终功率进行一次轻微平滑，消除计算非线性带来的毛刺
    final_smooth_sigma = 60 / dt
    P_screen = gaussian_filter1d(P_screen, final_smooth_sigma)
    P_cpu = gaussian_filter1d(P_cpu, final_smooth_sigma)
    P_com = gaussian_filter1d(P_com, final_smooth_sigma)
    P_gps = gaussian_filter1d(P_gps, final_smooth_sigma)
    
    return times, P_bg, P_screen, P_cpu, P_com, P_gps

def draw_beautiful_chart():
    # 运行模拟
    print("Generating continuous trajectory...")
    times, p_bg, p_screen, p_cpu, p_com, p_gps = simulate_continuous_power()
    
    # 准备绘图数据
    # 修改堆叠顺序：Background 移到最后（最上层）
    # 顺序：Screen -> CPU -> Communication -> GPS -> Background
    y_stack = np.vstack([p_screen, p_cpu, p_com, p_gps, p_bg])
    labels = ["Screen", "CPU", "Communication", "GPS", "Background"]
    
    # 修改配色方案
    # Screen(保留蓝), CPU(黄色), Communication(橙色), GPS(红色), Background(保留浅灰)
    colors = ['#4A90E2', '#FFD700', '#FF8C00', '#E53935', '#E0E0E0']
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制堆叠图
    # baseline='zero' 确保从0开始堆叠
    ax.stackplot(times, y_stack, labels=labels, colors=colors, alpha=0.9)
    
    # 绘制总功率曲线（黑色细线），增加轮廓感
    total_power = np.sum(y_stack, axis=0)
    ax.plot(times, total_power, color='black', linewidth=1.5, alpha=0.3, label='_nolegend_')
    
    # 美化图表元素
    ax.set_title('Smartphone Power Consumption Simulation (24 Hours)', fontsize=18, pad=20)
    ax.set_xlabel('Time (hours)', fontsize=14)
    ax.set_ylabel('Power (W)', fontsize=14)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, np.max(total_power) * 1.15)
    
    # 设置网格
    ax.grid(True, which='major', linestyle='--', alpha=0.4, color='gray')
    
    # 优化图例位置
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
              title="Components", frameon=False, fontsize=12, title_fontsize=14)
    
    # 调整布局以适应图例
    plt.tight_layout()
    
    print("Plotting...")
    plt.show()

if __name__ == "__main__":
    draw_beautiful_chart()