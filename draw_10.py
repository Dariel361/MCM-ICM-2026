import pandas as pd
import matplotlib.pyplot as plt
import os

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'aging_simulation_results.csv')

# 读取 CSV 数据
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit(1)
    
df = pd.read_csv(file_path)

# 将分钟转换为小时
df['Runtime_Hours'] = df['Runtime_Minutes'] / 60.0

# 设置绘图风格 (学术风格)
plt.style.use('seaborn-v0_8-paper')
plt.figure(figsize=(10, 6))

# 绘制散点图
# c='blue' 设置颜色
# s=30 设置点的大小
# alpha=0.7 设置透明度，防止重叠时看不清
# edgecolors='w' 设置点的边缘颜色为白色，增加对比度
plt.scatter(df['Age_Months'], df['Runtime_Hours'], c='#1f77b4', s=30, alpha=0.8, edgecolors='white', linewidth=0.5, label='Data Points')

# 设置标题和标签（英语，学术风格）
plt.title('Scatter Plot of Battery Operational Duration vs. Aging Time', fontsize=16, fontweight='normal', pad=15)
plt.xlabel('Aging Time (Months)', fontsize=14)
plt.ylabel('Operational Duration (Hours)', fontsize=14)

# 设置坐标轴刻度和网格
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5, color='gray')

# 添加图例（可选，只有一个系列数据时可省略）
# plt.legend()

# 调整布局并保存
plt.tight_layout()
output_file = 'aging_simulation_scatter_plot.png'
plt.savefig(output_file, dpi=300)
print(f"Scatter plot saved to {output_file}")
# plt.show()
