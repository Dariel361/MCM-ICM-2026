import pandas as pd
import matplotlib.pyplot as plt
import os

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 优先使用 stochastic 数据，因为那个更平滑且包含随机性，更像股票波动
file_path = os.path.join(current_dir, 'aging_simulation_stochastic.csv')

# 如果 stochastic 文件不存在，回退到普通结果
if not os.path.exists(file_path):
    print(f"Warning: {file_path} not found. Trying aging_simulation_results.csv")
    file_path = os.path.join(current_dir, 'aging_simulation_results.csv')

if not os.path.exists(file_path):
    print(f"Error: No data file found.")
    exit(1)
    
df = pd.read_csv(file_path)

# 处理列名差异
if 'Avg_Runtime_Minutes' in df.columns:
    y_col = 'Avg_Runtime_Minutes'
else:
    y_col = 'Runtime_Minutes'

# 将分钟转换为小时
df['Runtime_Hours'] = df[y_col] / 60.0

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 6))

# 颜色设置 (仿照样图的绿色系)
# 线条颜色: 深绿色
line_color = '#2E8B57'  # SeaGreen
# 填充颜色: 浅绿色
fill_color = '#98FB98'  # PaleGreen

# 绘制面积图
# alpha=0.2 保证填充色不会太深，不遮挡网格
plt.fill_between(df['Age_Months'], df['Runtime_Hours'], color=fill_color, alpha=0.3)
# 绘制折线 (透明度很高，几乎透明)
plt.plot(df['Age_Months'], df['Runtime_Hours'], color=line_color, linewidth=2, alpha=0.1)

# 设置标题和标签 (学术风格，英语)
plt.title('Time Series Analysis of Battery Operational Duration', fontsize=14, pad=15)
plt.xlabel('Aging Time (Months)', fontsize=12)
plt.ylabel('Operational Duration (Hours)', fontsize=12)

# 设置坐标轴范围
plt.xlim(df['Age_Months'].min(), df['Age_Months'].max())
# Y轴下限设为0或者稍微低于最小值，以展示填充效果
y_min = max(0, df['Runtime_Hours'].min() * 0.8)
y_max = df['Runtime_Hours'].max() * 1.05
plt.ylim(y_min, y_max)

# 设置网格 (浅灰色，细线，增加透明度)
plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.5, color='#E0E0E0')

# 去掉上边框和右边框，更符合现代学术图表风格
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 调整布局并保存
plt.tight_layout()
output_file = 'aging_simulation_time_series_green_plot.png'
plt.savefig(output_file, dpi=300)
print(f"Plot saved to {output_file}")
# plt.show()
