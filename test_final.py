import torch
import numpy as np
import argparse
import time
from tqdm import tqdm
import os

# 导入你的模型和数据加载器
from model import DGCRN_Model
from data_loader import get_dataloaders

# ==================== 1. 定义评估指标 (与 train.py 逻辑完全一致) ====================
def masked_mae_loss(y_pred, y_true, null_val=0.0):
    mask = (y_true != null_val).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[torch.isnan(loss)] = 0
    return loss.mean()

def masked_mape(y_pred, y_true, null_val=0.0, epsilon=1.0):
    # 过滤掉过小的值，防止除以0或数值不稳定
    mask = (y_true > epsilon).float()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=y_pred.device)
    mask /= mask.mean()
    # loss = |pred - true| / max(|true|, epsilon)
    loss = torch.abs(y_pred - y_true) / torch.clamp(torch.abs(y_true), min=epsilon)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.zeros_like(loss), loss)
    return loss.mean() * 100

def masked_rmse(y_pred, y_true, null_val=0.0):
    mask = (y_true != null_val).float()
    mask /= mask.mean()
    loss = (y_pred - y_true) ** 2
    loss = loss * mask
    loss[torch.isnan(loss)] = 0
    return torch.sqrt(loss.mean())

# ==================== 2. 测试主逻辑 ====================
def test_model(args):
    # --- A. 设置设备 ---
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🚀 正在使用设备: {device}")

    # --- B. 加载数据 ---
    print(f"\n📂 正在加载数据: {args.data} ...")
    # batch_size 测试时可以稍微大一点，因为不用存梯度
    dataloaders, adj_matrix, scaler = get_dataloaders(args.data, batch_size=64)
    scaler = scaler.to(device)
    
    test_loader = dataloaders['test']
    print(f"📊 测试集样本数: {len(test_loader.dataset)}")

    # --- C. 构建模型结构 ---
    # ⚠️ 关键：这里的参数必须与你训练时的参数完全一致！
    print(f"🏗️ 正在构建模型 (Hidden: {args.hidden_dim}, Embed: {args.embed_dim})...")
    
    model = DGCRN_Model(
        num_nodes=adj_matrix.shape[0],
        static_adj=adj_matrix.to(device),
        input_dim=1,
        hidden_dim=args.hidden_dim,   # 从命令行读取
        out_seq_len=12,               # 默认预测12步(60min)
        num_layers=args.num_layers,
        embed_dim=args.embed_dim,     # 🔄 从命令行读取
        decoder_type='residual'
    ).to(device)

    # --- D. 加载权重 ---
    if not os.path.exists(args.checkpoint):
        print(f"❌ 错误: 找不到模型文件 {args.checkpoint}")
        return

    print(f"📥 正在加载权重: {args.checkpoint} ...")
    # weights_only=False 解决 PyTorch 新版本报错
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # 兼容保存整个 checkpoint 字典或只保存 state_dict 的情况
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 成功加载 Checkpoint (Epoch {checkpoint.get('epoch', 'Unknown')}, Val MAE: {checkpoint.get('val_mae', 'Unknown'):.4f})")
    else:
        model.load_state_dict(checkpoint)
        print("✅ 成功加载 State Dict")

    model.eval()

    # --- E. 开始推理 ---
    preds = []
    trues = []
    
    print("\nruning 推理中...")
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc="Testing"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # 1. 前向传播
            y_pred = model(x_batch) # (B, 12, N)

            # 2. 反归一化 (还原为真实速度 km/h)
            y_pred_real = scaler.inverse_transform(y_pred)
            y_true_real = scaler.inverse_transform(y_batch.squeeze(-1))

            preds.append(y_pred_real.cpu())
            trues.append(y_true_real.cpu())

    # 拼接所有 batch
    preds = torch.cat(preds, dim=0) # (Total_Samples, 12, N)
    trues = torch.cat(trues, dim=0)

    # --- F. 计算并打印指标 ---
    print("\n" + "="*50)
    print("   🏆 测试集最终评估结果   ")
    print("="*50)

    # 1. 总体指标
    mae = masked_mae_loss(preds, trues).item()
    rmse = masked_rmse(preds, trues).item()
    mape = masked_mape(preds, trues, epsilon=args.mape_epsilon).item()
    
    print(f"Overall Performance (Avg 0-60 min):")
    print(f"  MAE : {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print("-" * 50)

    # 2. 分步指标 (15min, 30min, 60min)
    # 索引: 2->15min, 5->30min, 11->60min
    horizons = [2, 5, 11] 
    times = ['15 min', '30 min', '60 min']

    for t_idx, t_name in zip(horizons, times):
        pred_h = preds[:, t_idx, :]
        true_h = trues[:, t_idx, :]
        
        h_mae = masked_mae_loss(pred_h, true_h).item()
        h_rmse = masked_rmse(pred_h, true_h).item()
        h_mape = masked_mape(pred_h, true_h, epsilon=args.mape_epsilon).item()
        
        print(f"Horizon {t_idx+1} ({t_name}):")
        print(f"  MAE : {h_mae:.4f}")
        print(f"  RMSE: {h_rmse:.4f}")
        print(f"  MAPE: {h_mape:.2f}%")
        print("-" * 30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 数据集
    parser.add_argument('--data', type=str, default='METR-LA_processed.npz', help='数据集路径')
    # 模型路径
    parser.add_argument('--checkpoint', type=str, required=True, help='训练好的模型路径 (.pth)')
    # 模型参数 (必须与训练时一致)
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--embed_dim', type=int, default=32, help='嵌入维度 (注意：如果训练时改了，这里也要改)')
    # 其他配置
    parser.add_argument('--num_layers', type=int, default=2, help='GCRN编码器的层数')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--mape_epsilon', type=float, default=1.0, help='MAPE计算阈值')
    
    args = parser.parse_args()
    test_model(args)