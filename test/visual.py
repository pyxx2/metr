import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse # 建议加上这个方便命令行改参数，不过硬编码也行

# 引入您的模型定义
from model import DGCRN_Model
from data_loader import get_dataloaders

def visualize_prediction(model_path, data_path, node_idx=0, horizon_idx=11, steps=288):
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用设备: {device}")
    
    # 2. 加载数据
    print(f"正在加载数据: {data_path} ...")
    dataloaders, adj_matrix, scaler = get_dataloaders(data_path, batch_size=64)
    scaler = scaler.to(device)
    
    # 3. 加载模型 (⚠️ 关键修改处)
    # 必须与您训练最佳模型时的参数完全一致！
    print("正在构建模型结构 (Layers=1, Embed=16, Hidden=64)...")
    model = DGCRN_Model(
        num_nodes=adj_matrix.shape[0],
        static_adj=adj_matrix.to(device),
        input_dim=1,
        
        # 🟢 [修改 1] 您的最佳超参数
        hidden_dim=64,   
        embed_dim=16,    # 改为 16
        num_layers=1,    # 改为 1
        
        out_seq_len=12,
        decoder_type='residual' # 确保这里也一致
    ).to(device)
    
    # 加载权重
    print(f"正在加载权重: {model_path} ...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 兼容性处理：检查是保存了整个dict还是只保存了state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    # 4. 推理
    preds_list = []
    trues_list = []
    
    print("正在进行推理...")
    with torch.no_grad():
        # 使用测试集 dataloaders['test']
        for x_batch, y_batch in dataloaders['test']:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(x_batch)
            
            # 反归一化
            y_pred_real = scaler.inverse_transform(y_pred)
            y_true_real = scaler.inverse_transform(y_batch.squeeze(-1))
            
            # 取特定节点和时间步
            pred_step = y_pred_real[:, horizon_idx, node_idx]
            true_step = y_true_real[:, horizon_idx, node_idx]
            
            preds_list.append(pred_step.cpu().numpy())
            trues_list.append(true_step.cpu().numpy())
            
            if len(np.concatenate(preds_list)) >= steps:
                break
                
    # 5. 绘图
    preds = np.concatenate(preds_list)[:steps]
    trues = np.concatenate(trues_list)[:steps]
    
    # 创建画布
    plt.figure(figsize=(10, 5))
    
    # 画线
    plt.plot(trues, label='Ground Truth', color='black', alpha=0.7, linewidth=1.5) # 真实值用黑色或灰色更清晰
    plt.plot(preds, label='DG-TVCRN (Ours)', color='#E9573F', linewidth=1.5, linestyle='-') # 预测值用您的主题红
    
    # 装饰
    plt.title(f'Traffic Speed Prediction (Node {node_idx}, 60 min ahead)', fontsize=14, fontweight='bold')
    plt.xlabel('Time Step (5 min intervals)', fontsize=12)
    plt.ylabel('Speed (km/h)', fontsize=12)
    plt.legend(loc='upper right', frameon=True, fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # 保存
    save_path = f'vis_node_{node_idx}.png'
    plt.savefig(save_path, dpi=300)
    print(f"✅ 可视化完成，已保存为: {save_path}")
    plt.show()

if __name__ == '__main__':
    # 🟢 [修改 2] 替换为您具体的文件名
    # MODEL_FILE = "DG_TVCRN_run1_embed_dim_16_hiddden_dim_64_layers_1.pth"
    MODEL_FILE = "DG_TVCRN_run2_embed_dim_16_hiddden_dim_64_layers_1_2.pth"
    DATA_FILE = "PEMS-BAY_processed.npz" # 或者 PEMS-BAY
    
    # 建议多试几个节点，找一个曲线波动大、预测效果好的
    # METR-LA 推荐节点: 11, 112, 50, 200
    # PEMS-BAY 推荐节点: 10, 100, 150
    visualize_prediction(MODEL_FILE, DATA_FILE, node_idx=172, horizon_idx=11)