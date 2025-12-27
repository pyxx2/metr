import matplotlib.pyplot as plt
import numpy as np

# ==================== 1. 数据准备 (请替换为您的真实实验数据) ====================
# 横坐标：超参数的值 (例如：Embed Dim 或 Num Layers)
x_values = [2, 4, 6, 8, 10, 12] 
# 将其转换为字符串或是保持数值均可，如果想要刻度对齐，建议保持数值

# 左轴数据：MAE (柱状图)
mae_values = [19.49, 19.18, 18.97, 19.01, 19.16, 19.03]

# 右轴数据：MAPE (%) (折线图)
mape_values = [12.82, 12.61, 12.55, 12.54, 12.64, 12.50]

# ==================== 2. 创建画布与左轴 (MAE) ====================
# 设置字体，防止中文乱码 (如果是英文论文，可以注释掉下面这行)
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建图表对象，figsize=(宽, 高)
fig, ax1 = plt.subplots(figsize=(8, 5))

# --- 绘制柱状图 (MAE) ---
# color: 淡紫色 (#C8C8F3), alpha: 透明度, width: 柱子宽度
bar_color = '#CCCCF5' 
bars = ax1.bar(x_values, mae_values, color=bar_color, width=1.0, label='MAE', align='center')

# 设置左轴标签和刻度
ax1.set_ylabel('MAE', fontsize=14, color='black')
ax1.set_xlabel('Hyperparameter Value', fontsize=14) # 横坐标标签，请修改为您研究的参数名
ax1.tick_params(axis='y', labelsize=12)
ax1.tick_params(axis='x', labelsize=12)

# 设置左轴范围 (为了模仿原图效果，不从0开始，而是聚焦在数据变化区间)
# 您可以根据自己的数据范围手动调整下面的数值
ax1.set_ylim(18.5, 19.7) 
ax1.set_yticks(np.arange(18.5, 19.8, 0.3)) # 设置刻度间隔

# ==================== 3. 创建右轴 (MAPE) ====================
# twinx() 共享 x 轴
ax2 = ax1.twinx()

# --- 绘制折线图 (MAPE) ---
# color: 淡红色 (#FF9999), marker: 圆点, linewidth: 线宽
line_color = '#FF8884'
line = ax2.plot(x_values, mape_values, color=line_color, marker='o', 
                markersize=5, linewidth=1.5, label='MAPE')

# 设置右轴标签
ax2.set_ylabel('MAPE(%)', fontsize=14, color='black')
ax2.tick_params(axis='y', labelsize=12)

# 设置右轴范围 (同样聚焦数据区间)
ax2.set_ylim(12.2, 13.0)
ax2.set_yticks(np.arange(12.2, 13.1, 0.2))

# ==================== 4. 合并图例 (Legend) ====================
# 因为分属两个轴，需要手动收集图例句柄
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# 合并句柄和标签
final_handles = handles1 + handles2
final_labels = labels1 + labels2

# 绘制图例 (位置：upper right)
# framealpha=0.8 设置图例背景稍微透明一点
ax1.legend(final_handles, final_labels, loc='upper right', fontsize=12, framealpha=0.9)

# ==================== 5. 调整布局与保存 ====================
plt.title("Sensitivity Analysis", fontsize=16) # 标题
plt.tight_layout() # 自动调整布局防止重叠

# 保存图片 (dpi=300 为高清打印标准)
plt.savefig("hyperparameter_plot.png", dpi=300)
print("图表已保存为 hyperparameter_plot.png")

# 显示图表
plt.show()