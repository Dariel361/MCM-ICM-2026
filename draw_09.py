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

# 设置绘图参数
plt.figure(figsize=(12, 5))  # 调整长宽比以接近样图

# 颜色设置 (参考样图的绿色风格)
line_color = '#2E8B57'   # 深绿色 SeaGreen
fill_color = '#98FB98'   # 浅绿色 PaleGreen

# 绘制面积图 (Area Chart)
# alpha 控制透明度，linewidth 控制线条粗细
plt.fill_between(df['Age_Months'], df['Runtime_Hours'], color=fill_color, alpha=0.3)
plt.plot(df['Age_Months'], df['Runtime_Hours'], color=line_color, linewidth=1.5)

# 设置标题和标签
# 标题设置为学术风格，不加粗 (fontweight='normal')
plt.title('Time Series Analysis of Battery Operational Duration', fontsize=14, fontweight='normal', pad=15)
plt.xlabel('Aging Time (Months)', fontsize=12)
plt.ylabel('Duration (Hours)', fontsize=12)

# 设置坐标轴范围
plt.xlim(df['Age_Months'].min(), df['Age_Months'].max())
# Y轴下限设为0或者稍微低于最小值，以展示填充效果，这里设为0更符合面积图逻辑
plt.ylim(0, df['Runtime_Hours'].max() * 1.1)

# 设置网格 (浅灰色，细线)
plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.5, color='lightgray')

# 调整刻度显示
plt.tick_params(axis='both', labelsize=10)

# 保存图片
plt.tight_layout()
output_file = 'aging_simulation_time_series_plot.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_file}")
