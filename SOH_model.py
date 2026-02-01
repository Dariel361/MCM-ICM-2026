import numpy as np

class SOHModel:
    def __init__(self, params):
        """
        初始化 SOH 老化模型
        params: 包含 A 和 k 的字典
        """
        self.A = params.get('A', 25.0)
        self.k = params.get('k', -0.06)

    def calculate_SOH(self, t_months):
        """
        计算电池在 t_months 月后的 SOH
        注意: 内部会将月份转换为年份以匹配参数 k (通常基于年)
        """
        # 确保 t_months 非负
        t = np.maximum(t_months, 0.0)
        
        # 将月份转换为年份 (k=-0.06 是基于年的参数)
        t_years = t / 12.0
       
        # SOH = exp(k * t_years)
        # 5年 (60个月): exp(-0.06 * 5) = 0.74 (74%)
        soh = np.exp(self.k * t_years)
        
        return soh

if __name__ == "__main__":
    # 简单的测试
    params = {'A': 25.0, 'k': -0.06}
    model = SOHModel(params)
    
    months = [0, 6, 12, 36, 40, 24, 60, 120]
    print(f"SOH 模型测试 (A={params['A']}, k={params['k']}, Input in Months):")
    for m in months:
        print(f"Month {m}: SOH = {model.calculate_SOH(m):.2f}%")
