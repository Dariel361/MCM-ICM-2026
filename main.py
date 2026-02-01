import numpy as np
import json
import sys
import os

# 添加父目录到 sys.path 以导入 SOH_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SOH_model import SOHModel
from the_P_load import PowerLoadModel

# 电池模型类
class BatteryModel:
    def __init__(self, params, current_soh=1.0):
        """
        初始化电池模型参数
        params: 字典，包含 Cn, eta_ref, delta_eta, T_ref, R1, C1...
        current_soh: 当前电池健康状态 (0~1)
        """
        self.params = params
        self.soh = current_soh
        self.state = np.array([1.0, 0.0, 0.0, 0.0])  # 初始状态x(t)=[SOC, U1, U2, Ud]
        
    def get_current_eta(self, T_current):
        """
        根据当前温度计算放电效率 eta(T)
        """
        eta_ref = self.params['eta_ref']
        delta_eta = self.params['delta_eta']
        T_ref = self.params['T_ref']
        
        # 如果当前温度低于参考温度，效率衰减
        if T_current < T_ref:
            eta = eta_ref * (1.0 - delta_eta * (T_ref - T_current))
            
            # 物理约束：效率不应小于0（虽然在正常参数下很难发生，但为了数值安全）
            eta = max(eta, 0.01)
        else:
            # 高于参考温度时，维持参考效率
            eta = eta_ref
            
        return eta

    def get_U_OC(self, soc):
        """
        根据 SOC 计算开路电压 U_OC
        公式: U_OC = 9.37*SOC^8 - 128.51*SOC^7 + 419.45*SOC^6 - 617.86*SOC^5 
                     + 479.31*SOC^4 - 200.09*SOC^3 + 42.29*SOC^2 - 3.32*SOC + 3.54
        """
        s = np.clip(soc, 0.0, 1.0)
        return (9.37 * s**8 - 128.51 * s**7 + 419.45 * s**6 
                - 617.86 * s**5 + 479.31 * s**4 - 200.09 * s**3 
                + 42.29 * s**2 - 3.32 * s + 3.54)

    def get_current_R0(self, T):
        a1, b1 = self.params['a1'], self.params['b1']
        c1, d1 = self.params['c1'], self.params['d1']
        
        # SOH 影响参数
        gamma = self.params.get('gamma', 0.5)
        k_r0 = self.params.get('k_r0', 1.5)
    
        # 基础 Arrhenius 关系
        R0_base = a1 * np.exp(b1 * T) + c1 * np.exp(d1 * T)
        
        # 引入 SOH 影响: R0 = R0_base * (1 + gamma * (1 - SOH)^k_r0)
        # 注意: self.soh 是 0~1 的小数
        aging_factor = 1.0 + gamma * (1.0 - self.soh)**k_r0
        
        return R0_base * aging_factor

    def get_derived_params(self, t, state, I_current, T_current=25.0):
        """
        计算状态导数
        """
        z, u1, u2, ud = state
        
        Cn = self.params['Cn']
        R1, C1 = self.params['R1'], self.params['C1']
        R2, C2 = self.params['R2'], self.params['C2']
        tau_d, kd = self.params['tau_d'], self.params['kd']
        
        current_eta = self.get_current_eta(T_current)
        
        # 实际容量 = 标称容量 * SOH
        Cn_actual = Cn * self.soh
        
        # SOC 变化率: dSOC/dt = - (eta * I) / (Q_nom * SOH)
        dz_dt = - (current_eta * I_current) / Cn_actual 
         
        # 极化动力学 (2RC)
        du1_dt = -u1 / (R1 * C1) + I_current / C1
        du2_dt = -u2 / (R2 * C2) + I_current / C2     

        # 扩散动力学
        dud_dt = -ud / tau_d + kd * I_current
        
        return np.array([dz_dt, du1_dt, du2_dt, dud_dt])
    
    def get_U_L(self, I_current, R0_current):
        """
        计算端电压 UL
        """
        z, u1, u2, ud = self.state

        U_OC = self.get_U_OC(z)
        
        # U_L = U_OC - U_polarization - U_diffusion - I*R0
        U_L = U_OC - (u1 + u2 + ud) - I_current * R0_current
        return U_L
    
    def update_state(self, dt, current, T_current=25.0):
        """
        更新一步状态 (使用 RK4)
        """
        # 定义计算导数的 lambda 函数，固定 inputs
        def get_derivs(state_val):
            return self.get_derived_params(0, state_val, current, T_current)

        # RK4 步骤
        k1 = get_derivs(self.state)
        k2 = get_derivs(self.state + 0.5 * dt * k1)
        k3 = get_derivs(self.state + 0.5 * dt * k2)
        k4 = get_derivs(self.state + dt * k3)
        
        self.state = self.state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # 物理约束：SoC 不能小于 0 或大于 1
        self.state[0] = np.clip(self.state[0], 0, 1)
        

    def solve_current_from_power(self, P_load, R0_current):
        """
        根据恒功率负载 P_load 计算所需的放电电流 I(t)
        求解方程: R0 * I^2 - (U_oc - U_polar) * I + P_load = 0
        
        Returns:
            I_sol: 计算出的电流值
            is_collapsed: 布尔值，如果发生电压崩塌(无解)则为 True
        """
        # 1. 获取当前状态变量
        z, u1, u2, ud = self.state
        U_OC = self.get_U_OC(z)
        
        # 2. 计算极化电压总和 
        U_polar = u1 + u2 + ud
        
        # 3. 构建一元二次方程 ax^2 + bx + c = 0
        # 对应文中的方程: R0*I^2 - (Uoc - Upolar)*I + Pload = 0
        a = R0_current
        b = -(U_OC - U_polar)
        c = P_load
        
        # 4. 计算判别式 Delta 
        delta = b**2 - 4 * a * c
        
        # 5. 判断是否崩塌 
        if delta < 0:
            # 判别式小于0，方程无实根 -> 电压崩塌，电池无法提供所需功率
            return None, True
        
        # 6. 求解电流 
        # I = (-b ± sqrt(delta)) / 2a
        # 对于放电电路，我们取较小的那个根作为物理稳定解 (对应较高的电压)
        # 因为 I 越小，IR 压降越小，端电压越高，效率越高。
        # 另外：b是负数，所以 -b 是正数。
        # I = ((Uoc - Upolar) - sqrt(delta)) / (2*R0)
        I_sol = (-b - np.sqrt(delta)) / (2 * a)
        
        return I_sol, False


# 热模型类
class T_Model:
    def __init__(self, params, T_init):
        self.params = params
        self.T = T_init  # 当前温度状态 T(t)
        
    def get_dT_dt(self, T_state, I_current, R0_current, T_amb):
        """
        计算温度变化率 dT/dt
        """
        Cth = self.params['Cth'] # 热容
        Rth = self.params['Rth'] # 热阻
        
        # 1. 产热 (焦耳热) 
        P_heat = (I_current ** 2) * R0_current
        
        # 2. 散热 
        P_diss = (T_state - T_amb) / Rth
        
        # 3. 热平衡方程 
        dT_dt = (P_heat - P_diss) / Cth
        
        return dT_dt
        
    def update_temperature(self, dt, I_current, r0_func, T_amb):
        """
        使用 RK4 更新温度
        r0_func: 用于根据温度计算 R0 的函数句柄
        """
        T_0 = self.T

        # k1
        R0_1 = r0_func(T_0)
        k1 = self.get_dT_dt(T_0, I_current, R0_1, T_amb)
        
        # k2
        T_2 = T_0 + 0.5 * dt * k1
        R0_2 = r0_func(T_2)
        k2 = self.get_dT_dt(T_2, I_current, R0_2, T_amb)
        
        # k3
        T_3 = T_0 + 0.5 * dt * k2
        R0_3 = r0_func(T_3)
        k3 = self.get_dT_dt(T_3, I_current, R0_3, T_amb)
        
        # k4
        T_4 = T_0 + dt * k3
        R0_4 = r0_func(T_4)
        k4 = self.get_dT_dt(T_4, I_current, R0_4, T_amb)
        
        self.T = T_0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
     



# 实验部分
def load_parameters(filepath='parameters.json'):
    """读取参数文件并返回字典"""
    import os
    # 优先尝试当前目录，若不存在则尝试绝对路径
    if not os.path.exists(filepath):
        filepath = '/Users/daunt/归档/美赛/数学建模美赛/Model/parameters.json'
        
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    # 1. 加载参数
    all_params = load_parameters()
    sim_conf = all_params['simulation']
    
    # 1.1 计算当前 SOH
    soh_params = all_params.get('soh_model', {})
    soh_model = SOHModel(soh_params)
    battery_age_months = sim_conf.get('battery_age_months', 0.0)
    
    # SOHModel 返回的是 0~1 的小数 (例如 0.95)
    current_soh_val = soh_model.calculate_SOH(battery_age_months)
    print(f"当前电池状态: Age={battery_age_months} months, SOH={current_soh_val*100:.2f}%")
    
    # 2. 初始化模型
    battery = BatteryModel(all_params['electrochemical'], current_soh=current_soh_val)
    thermal = T_Model(all_params['thermal'], T_init=sim_conf['initial_temp'])
    load_model = PowerLoadModel(all_params.get('load_model'))
    
    # 3. 实验设定
  
    dt = sim_conf['dt']
    T_amb = sim_conf['T_amb']
    
    print(f"=== 开始动态负载放电仿真 (T_amb={T_amb}°C) ===")
    
    t = 0.0
    # 记录最后的状态以便输出
    final_I, final_V, final_R0, final_T = 0.0, 0.0, 0.0, 0.0
    
    # 记录截止时间 (达到截止电压的时间)
    t_cutoff = None
    
    while True:
        # A. 获取热状态和电阻
        T_curr = thermal.T
        R0_curr = battery.get_current_R0(T_curr)

        # A+. 计算当前动态负载功率
        P_load, state_name, _ = load_model.calculate_P_load(t)
        
        # B. 根据功率计算电流
        I_curr, is_collapsed = battery.solve_current_from_power(P_load, R0_curr)
        
        # C. 崩塌检测
        if is_collapsed:
            print(f"!!! 电压崩塌停止 t={t:.2f}s")
            print(f"电压崩塌时的 SOC: {battery.state[0]:.6f}")
            final_I, final_R0, final_T = 0.0, R0_curr, T_curr # 崩塌时电流无解
            final_V = 0.0
            break
            
        # D. 计算端电压
        v_term = battery.get_U_L(I_curr, R0_curr)
        
        # 保存当前有效状态
        final_I, final_V, final_R0, final_T = I_curr, v_term, R0_curr, T_curr
        
        # 记录/打印 (每1小时打印一次状态)
        if int(t) % 1800 == 0:
            print(f"t={t:.0f}s | SoC={battery.state[0]:.4f} | V={v_term:.3f}V | I={I_curr:.4f}A | T={T_curr:.2f}°C | P_load={P_load:.3f}W ({state_name})")
        
        # E. 截止条件判断
        # 1. 电压截止 (记录时间但不强制停止，为了寻找可能的崩塌点)
        if v_term < 0 and t_cutoff is None:
            print(f"达到截止电压 t={t:.2f}s ")
            t_cutoff = t
            break
        
        # 2. 电量完全耗尽 
        if battery.state[0] <= 0.03:
            print(f"SoC 耗尽 (SOC <= 0.03) 停止,端电压为{final_V:.4f},SOC={battery.state[0]:.6f}。")
            break
            
        

        # F. 更新状态 (RK4)
        battery.update_state(dt, I_curr, T_curr)
        thermal.update_temperature(dt, I_curr, battery.get_current_R0, T_amb)
        
        t += dt

    # 输出最终结果
    # 如果没有达到电压截止就崩塌或SOC耗尽，则耗尽时间为当前时间
    report_t = t_cutoff if t_cutoff is not None else t

    print("=" * 40)
    print(f"电池电量耗尽时间 t: {report_t:.2f} s ({report_t / (3600 * 24):.4f} days)")
    print(f"最终状态参数:")
    print(f"  电流 (I) : {final_I:.4f} A")
    print(f"  电压 (V) : {final_V:.4f} V")
    print(f"  电阻 (R0): {final_R0:.4f} Ω")
    print(f"  温度 (T) : {final_T:.4f} °C")
