import numpy as np
import time

class PowerLoadModel:
    def __init__(self, params=None):
        """
        初始化负载模型
        params: 字典，可选配置参数 (如 seed)
        """
        self.params = params if params is not None else {}
        # 优先使用params中的seed，否则基于时间生成
        self.seed = self.params.get('seed', int(time.time() * 1000) % 1000000)

        # 定义5种用户状态：空闲、社交、游戏、视频、导航
        self.states = ['standby', 'social', 'game', 'video', 'nav']

        # 连续时间马尔可夫链(CTMC)的生成矩阵Q
        self.Q = np.array([
            [-2.0, 1.0, 0.3, 0.5, 0.2],  # standby
            [0.8, -3.0, 0.5, 1.2, 0.5],  # social
            [0.2, 0.3, -1.5, 0.5, 0.5],  # game
            [0.5, 0.8, 0.4, -2.5, 0.4],  # video
            [0.3, 0.4, 0.3, 0.5, -3.5]   # nav
        ])

        # 各状态下关键参数的均值
        self.params_mean = {
            'standby': [0.15, 0.10, 0.05, 0.05],
            'social': [0.40, 0.30, 0.30, 0.10],
            'game': [0.85, 0.75, 0.50, 0.01],
            'video': [0.70, 0.40, 2.00, 0.05],
            'nav': [0.80, 0.50, 0.20, 1.00]
        }

    def calculate_P_load(self, t, seed=None):
        """
        计算时刻t的智能手机负载功率P_load(t)
        """
        # 如果未提供随机数种子，使用实例默认种子
        current_seed = seed if seed is not None else self.seed
        np.random.seed(current_seed)

        # 初始化状态
        current_state = 0  # standby
        current_time = 0.0

        # 模拟状态演化
        while current_time < t:
            rate_out = -self.Q[current_state, current_state]
            tau_hours = np.random.exponential(1.0 / rate_out)
            tau_seconds = tau_hours * 3600

            probs = self.Q[current_state, :].copy()
            probs[current_state] = 0
            probs = np.maximum(probs, 0)

            if probs.sum() > 0:
                probs = probs / probs.sum()
                next_state = np.random.choice(5, p=probs)
            else:
                next_state = current_state

            if current_time + tau_seconds > t:
                break

            current_time += tau_seconds
            current_state = next_state

        # 确定最终状态及参数
        state_name = self.states[current_state]
        mean_params = self.params_mean[state_name]

        params = []
        for i, mean_val in enumerate(mean_params):
            noise = 0.2 * mean_val * np.random.randn()
            if i == 3:  # GPS
                val = 1.0 if mean_val > 0.5 else 0.0
            else:
                val = max(mean_val + noise, 0.05)
            params.append(val)

        L, cpu_usage, data_rate, gps_active = params

        # 计算功耗
        P_screen = 1.5 * (L ** 2.2)
        P_cpu = 0.1 + 1.9 * (cpu_usage ** 1.3)
        P_com = 0.2 + 0.5 * data_rate / 0.4
        P_gps = 0.3 if gps_active > 0.5 else 0.0
        P_background = 0.15

        P_foreground = P_screen + P_cpu + P_com + P_gps
        P_load = P_background + P_foreground

        components = {
            'Screen': P_screen,
            'CPU': P_cpu,
            'Com': P_com,
            'GPS': P_gps,
            'Background': P_background
        }

        return P_load, state_name, components

# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("智能手机负载功率（随机模拟）")
    print("=" * 100)
    
    # 实例化模型
    model = PowerLoadModel()

    time_points = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0]

    for t_hours in time_points:
        t_seconds = t_hours * 3600
        P_load, state, comps = model.calculate_P_load(t_seconds)

        print(f"t={t_hours:4.1f}h: P_load={P_load:5.3f}W, state={state:<6} | "
              f"Scn={comps['Screen']:.3f}W, CPU={comps['CPU']:.3f}W, "
              f"Com={comps['Com']:.3f}W, GPS={comps['GPS']:.3f}W, Bg={comps['Background']:.3f}W")

    print("=" * 100)