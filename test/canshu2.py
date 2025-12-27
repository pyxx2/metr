import matplotlib.pyplot as plt
import numpy as np

# ==================== 1. 数据准备 (已填入您的真实数据 - 60min Horizon) ====================
# 横坐标：隐藏层维度
dims = ['16', '32', '64', '128'] 

# --- METR-LA 数据 (60 min / Horizon 12) ---
# 16:  3.6869, 10.87%
# 32:  3.5277, 10.27%
# 64:  3.4460, 9.96%  (最佳)
# 128: 3.4851, 10.06% (反弹)
mae_metr = [3.6869, 3.5277, 3.4460, 3.4851]
mape_metr = [10.87, 10.27, 9.96, 10.06]

# --- PEMS-BAY 数据 (60 min / Horizon 12) ---
# 16:  2.1284, 5.16%
# 32:  2.0809, 5.06%
# 64:  2.0397, 4.81%  (最佳)
# 128: 2.0481, 4.88% (反弹)
mae_pems = [2.1284, 2.0809, 2.0397, 2.0481]
mape_pems = [5.16, 5.06, 4.81, 4.88]

# ==================== 2. 全局设置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 创建画布：1行2列
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 6))

# 配色方案
color_bar = '#5D9CEC'  # 天青蓝 (MAE)
color_line = '#E9573F' # 珊瑚红 (MAPE)

# ==================== 3. 绘制子图 1: METR-LA ====================
# --- 左轴: MAE (柱状图) ---
bars1 = ax1.bar(dims, mae_metr, color=color_bar, width=0.5, label='MAE', zorder=10, alpha=0.9)
ax1.set_xlabel('Hidden Dimension', fontsize=14, fontweight='bold')
ax1.set_ylabel('MAE', fontsize=14, color=color_bar, fontweight='bold')
ax1.set_title('(a) METR-LA Dataset', fontsize=16, fontweight='bold', pad=15) 
ax1.tick_params(axis='y', colors=color_bar)
ax1.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0) 

# 自动调整Y轴范围 (聚焦差异)
y_min, y_max = min(mae_metr), max(mae_metr)
margin = (y_max - y_min) * 0.5
ax1.set_ylim(y_min - margin, y_max + margin)

# --- 右轴: MAPE (折线图) ---
ax2 = ax1.twinx()
line1 = ax2.plot(dims, mape_metr, color=color_line, marker='o', linewidth=2.5, markersize=8, label='MAPE', zorder=20)
ax2.set_ylabel('MAPE (%)', fontsize=14, color=color_line, fontweight='bold')
ax2.tick_params(axis='y', colors=color_line)

# 自动调整Y轴范围
y_min_2, y_max_2 = min(mape_metr), max(mape_metr)
margin_2 = (y_max_2 - y_min_2) * 0.5
ax2.set_ylim(y_min_2 - margin_2, y_max_2 + margin_2)

# --- 合并图例 ---
lns1 = [bars1[0]] + line1
labs1 = ['MAE', 'MAPE']
ax1.legend(lns1, labs1, loc='upper right', framealpha=0.95, shadow=True)


# ==================== 4. 绘制子图 2: PEMS-BAY ====================
# --- 左轴: MAE (柱状图) ---
bars2 = ax3.bar(dims, mae_pems, color=color_bar, width=0.5, label='MAE', zorder=10, alpha=0.9)
ax3.set_xlabel('Hidden Dimension', fontsize=14, fontweight='bold')
ax3.set_ylabel('MAE', fontsize=14, color=color_bar, fontweight='bold')
ax3.set_title('(b) PEMS-BAY Dataset', fontsize=16, fontweight='bold', pad=15)
ax3.tick_params(axis='y', colors=color_bar)
ax3.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)

# 自动调整Y轴范围
y_min_3, y_max_3 = min(mae_pems), max(mae_pems)
margin_3 = (y_max_3 - y_min_3) * 0.5
ax3.set_ylim(y_min_3 - margin_3, y_max_3 + margin_3)

# --- 右轴: MAPE (折线图) ---
ax4 = ax3.twinx()
line2 = ax4.plot(dims, mape_pems, color=color_line, marker='o', linewidth=2.5, markersize=8, label='MAPE', zorder=20)
ax4.set_ylabel('MAPE (%)', fontsize=14, color=color_line, fontweight='bold')
ax4.tick_params(axis='y', colors=color_line)

# 自动调整Y轴范围
y_min_4, y_max_4 = min(mape_pems), max(mape_pems)
margin_4 = (y_max_4 - y_min_4) * 0.5
ax4.set_ylim(y_min_4 - margin_4, y_max_4 + margin_4)

# --- 合并图例 ---
lns2 = [bars2[0]] + line2
labs2 = ['MAE', 'MAPE']
ax3.legend(lns2, labs2, loc='upper right', framealpha=0.95, shadow=True)

# ==================== 5. 保存与展示 ====================
plt.tight_layout()
plt.savefig('hidden_dim_sensitivity.png', dpi=300, bbox_inches='tight')
print("✅ 隐藏层维度分析图表已保存为 hidden_dim_sensitivity.png")
plt.show()