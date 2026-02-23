import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties # 引入字体管理器

# ==================== 0. 字体配置 ====================
# 请确保同级目录下有一个名为 SimHei.ttf 的字体文件
font_path = "SimHei.ttf"
font_title = FontProperties(fname=font_path, size=16, weight='bold')
font_label = FontProperties(fname=font_path, size=14, weight='bold')
font_legend = FontProperties(fname=font_path, size=12)

# ==================== 1. 数据准备 (基于原图视觉估算还原) ====================
# 横坐标：编码器层数
dims = ['1', '2', '3']

# --- METR-LA 数据 ---
# Layer 1: 3.4460, 9.96%  (基准最佳值)
# Layer 2: 3.5000, 10.09% (看图 MAE 刚好在 3.50 线)
# Layer 3: 3.5060, 10.11% (看图略高于 Layer 2)
mae_metr = [3.4460, 3.5000, 3.5060]
mape_metr = [9.96, 10.09, 10.11]

# --- PEMS-BAY 数据 ---
# Layer 1: 2.0397, 4.81%  (基准最佳值)
# Layer 2: 2.0400, 4.91%  (看图 MAE 基本持平，MAPE 飙升过 4.90)
# Layer 3: 2.0688, 4.90%  (看图 MAE 接近 2.07，MAPE 微降)
mae_pems = [2.0397, 2.0400, 2.0688]
mape_pems = [4.81, 4.91, 4.90]

# ==================== 2. 全局设置 ====================
plt.rcParams['axes.unicode_minus'] = False

# 创建画布：1行2列
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 6))

# 配色方案 (保持统一)
color_bar = '#5D9CEC'  # 天青蓝 (MAE)
color_line = '#E9573F' # 珊瑚红 (MAPE)

# ==================== 3. 绘制子图 1: METR-LA ====================
# --- 左轴: MAE (柱状图) ---
bars1 = ax1.bar(dims, mae_metr, color=color_bar, width=0.4, label='MAE', zorder=10, alpha=0.9)
ax1.set_xlabel('编码器层数', fontproperties=font_label) 
ax1.set_ylabel('MAE', fontproperties=font_label, color=color_bar)
ax1.set_title('(a) METR-LA 数据集', fontproperties=font_title, pad=15)
ax1.tick_params(axis='y', colors=color_bar)
ax1.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)

# 自动调整Y轴范围 (为了让柱子不过低，可以适当控制 margin)
y_min, y_max = min(mae_metr), max(mae_metr)
margin = (y_max - y_min) * 0.8
ax1.set_ylim(y_min - margin, y_max + margin)

# --- 右轴: MAPE (折线图) ---
ax2 = ax1.twinx()
line1 = ax2.plot(dims, mape_metr, color=color_line, marker='o', linewidth=2.5, markersize=8, label='MAPE', zorder=20)
ax2.set_ylabel('MAPE (%)', fontproperties=font_label, color=color_line)
ax2.tick_params(axis='y', colors=color_line)

# 自动调整Y轴范围
y_min_2, y_max_2 = min(mape_metr), max(mape_metr)
margin_2 = (y_max_2 - y_min_2) * 0.8
ax2.set_ylim(y_min_2 - margin_2, y_max_2 + margin_2)

# --- 合并图例 ---
lns1 = [bars1[0]] + line1
labs1 = ['MAE', 'MAPE']
ax1.legend(lns1, labs1, loc='upper center', framealpha=0.95, shadow=True, prop=font_legend)


# ==================== 4. 绘制子图 2: PEMS-BAY ====================
# --- 左轴: MAE (柱状图) ---
bars2 = ax3.bar(dims, mae_pems, color=color_bar, width=0.4, label='MAE', zorder=10, alpha=0.9)
ax3.set_xlabel('编码器层数', fontproperties=font_label)
ax3.set_ylabel('MAE', fontproperties=font_label, color=color_bar)
ax3.set_title('(b) PEMS-BAY 数据集', fontproperties=font_title, pad=15)
ax3.tick_params(axis='y', colors=color_bar)
ax3.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)

# 自动调整Y轴范围
y_min_3, y_max_3 = min(mae_pems), max(mae_pems)
margin_3 = (y_max_3 - y_min_3) * 0.8
ax3.set_ylim(y_min_3 - margin_3, y_max_3 + margin_3)

# --- 右轴: MAPE (折线图) ---
ax4 = ax3.twinx()
line2 = ax4.plot(dims, mape_pems, color=color_line, marker='o', linewidth=2.5, markersize=8, label='MAPE', zorder=20)
ax4.set_ylabel('MAPE (%)', fontproperties=font_label, color=color_line)
ax4.tick_params(axis='y', colors=color_line)

# 自动调整Y轴范围
y_min_4, y_max_4 = min(mape_pems), max(mape_pems)
margin_4 = (y_max_4 - y_min_4) * 0.8
ax4.set_ylim(y_min_4 - margin_4, y_max_4 + margin_4)

# --- 合并图例 ---
lns2 = [bars2[0]] + line2
labs2 = ['MAE', 'MAPE']
ax3.legend(lns2, labs2, loc='upper center', framealpha=0.95, shadow=True, prop=font_legend)

# ==================== 5. 保存与展示 ====================
plt.tight_layout()
plt.savefig('layer_sensitivity.png', dpi=300, bbox_inches='tight')
print("✅ 网络层数分析图表已保存为 layer_sensitivity.png")
plt.show()