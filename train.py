import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse 
from tqdm import tqdm 
import time 

# 导入我们自己的模块
from data_loader import get_dataloaders
from model import DGCRN_Model

# --- 1. 定义评估指标 (Metrics) - 修复版本 ---

def masked_mae_loss(y_pred, y_true, null_val=0.0):
    """
    掩码MAE损失
    :param y_pred: 预测值
    :param y_true: 真实值
    :param null_val: 需要掩码的值
    """
    mask = (y_true != null_val).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[torch.isnan(loss)] = 0
    return loss.mean()


def masked_mape(y_pred, y_true, null_val=0.0, epsilon=1.0):
    """
    掩码MAPE - 修复版本
    
    关键修复:
    1. 使用 epsilon 防止除以很小的数
    2. 过滤掉真实值过小的样本
    3. 使用更robust的计算方式
    
    :param y_pred: 预测值
    :param y_true: 真实值  
    :param null_val: 需要掩码的值
    :param epsilon: 防止除零的阈值 (建议1.0，因为交通流速度通常>1)
    """
    # ⚠️ 关键修复1: 过滤掉真实值小于epsilon的样本
    mask = (y_true > epsilon).float()
    
    if mask.sum() == 0:
        # 如果所有值都被过滤，返回0
        return torch.tensor(0.0, device=y_pred.device)
    
    # 归一化mask
    mask /= mask.mean()
    
    # ⚠️ 关键修复2: 在分母上添加epsilon防止数值不稳定
    # loss = |pred - true| / max(|true|, epsilon)
    loss = torch.abs(y_pred - y_true) / torch.clamp(torch.abs(y_true), min=epsilon)
    
    # 应用mask
    loss = loss * mask
    
    # 过滤NaN和Inf
    loss = torch.where(torch.isnan(loss) | torch.isinf(loss), 
                       torch.zeros_like(loss), 
                       loss)
    
    return loss.mean() * 100


def masked_rmse(y_pred, y_true, null_val=0.0):
    """
    掩码RMSE
    """
    mask = (y_true != null_val).float()
    mask /= mask.mean()
    loss = (y_pred - y_true) ** 2
    loss = loss * mask
    loss[torch.isnan(loss)] = 0
    return torch.sqrt(loss.mean())


# --- 2. 训练函数 (Train) - 添加梯度裁剪和异常检测 ---
def train_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm=5.0):
    """
    训练一个epoch
    
    新增功能:
    - 梯度裁剪防止梯度爆炸
    - 异常值检测
    """
    model.train() 
    total_loss = 0
    num_batches = 0
    
    for x_batch, y_batch in tqdm(dataloader, desc="Training"):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device) 
        
        optimizer.zero_grad()
        
        # 前向传播
        y_pred = model(x_batch) 
        
        # 计算损失
        loss = criterion(y_pred, y_batch.squeeze(-1))
        
        # ⚠️ 异常检测
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️ 警告: 检测到异常损失值: {loss.item()}")
            print(f"   预测值范围: [{y_pred.min().item():.4f}, {y_pred.max().item():.4f}]")
            print(f"   真实值范围: [{y_batch.min().item():.4f}, {y_batch.max().item():.4f}]")
            continue  # 跳过这个batch
        
        # 反向传播
        loss.backward()
        
        # ⚠️ 梯度裁剪 (防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    if num_batches == 0:
        return float('inf')
    
    return total_loss / num_batches


# --- 3. 评估函数 (Evaluate) - 使用修复后的MAPE ---
def eval_epoch(model, dataloader, scaler, device, epsilon=1.0):
    """
    评估一个epoch
    
    :param epsilon: MAPE计算中的epsilon阈值
    """
    model.eval() 
    
    metrics = {
        'mae': [[], [], [], []],  # [15m], [30m], [60m], [Overall]
        'mape': [[], [], [], []],
        'rmse': [[], [], [], []]
    }
    
    with torch.no_grad(): 
        for x_batch, y_batch in tqdm(dataloader, desc="Evaluating"):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # 反归一化真实值
            y_true_real_full = scaler.inverse_transform(y_batch.squeeze(-1))
            
            # 预测
            y_pred_norm = model(x_batch)
            
            # ⚠️ 检查预测值是否异常
            if torch.isnan(y_pred_norm).any() or torch.isinf(y_pred_norm).any():
                print("⚠️ 警告: 预测值包含NaN或Inf，跳过此batch")
                continue
            
            # 反归一化预测值
            y_pred_real_full = scaler.inverse_transform(y_pred_norm)
            
            # 计算 "Overall" 指标
            metrics['mae'][3].append(masked_mae_loss(y_pred_real_full, y_true_real_full).item())
            metrics['mape'][3].append(masked_mape(y_pred_real_full, y_true_real_full, epsilon=epsilon).item())
            metrics['rmse'][3].append(masked_rmse(y_pred_real_full, y_true_real_full).item())
            
            # 计算特定时间步 (Horizon)
            for i, horizon_idx in enumerate([2, 5, 11]):  # 15min, 30min, 60min
                y_pred_horizon = y_pred_real_full[:, horizon_idx, :]
                y_true_horizon = y_true_real_full[:, horizon_idx, :]
                
                metrics['mae'][i].append(masked_mae_loss(y_pred_horizon, y_true_horizon).item())
                metrics['mape'][i].append(masked_mape(y_pred_horizon, y_true_horizon, epsilon=epsilon).item())
                metrics['rmse'][i].append(masked_rmse(y_pred_horizon, y_true_horizon).item())
    
    # 返回所有指标的平均值
    final_metrics = {
        '15min_mae': np.mean(metrics['mae'][0]), '15min_mape': np.mean(metrics['mape'][0]), '15min_rmse': np.mean(metrics['rmse'][0]),
        '30min_mae': np.mean(metrics['mae'][1]), '30min_mape': np.mean(metrics['mape'][1]), '30min_rmse': np.mean(metrics['rmse'][1]),
        '60min_mae': np.mean(metrics['mae'][2]), '60min_mape': np.mean(metrics['mape'][2]), '60min_rmse': np.mean(metrics['rmse'][2]),
        'overall_mae': np.mean(metrics['mae'][3]), 'overall_mape': np.mean(metrics['mape'][3]), 'overall_rmse': np.mean(metrics['rmse'][3]),
    }
    
    return final_metrics


# --- 4. 主函数 (Main) ---
def main():
    # --- A. 解析命令行参数 ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='METR-LA_processed.npz')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    # ➕ [新增] 添加 embed_dim 参数，默认值设为 16 (或其他你想要的默认值)
    parser.add_argument('--embed_dim', type=int, default=16, help='节点嵌入维度')
    parser.add_argument('--num_layers', type=int, default=1, help='GCRN编码器的层数')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--save_path', type=str, default='best_model.pth')
    parser.add_argument('--mape_epsilon', type=float, default=1.0, 
                        help='MAPE计算中的epsilon阈值，过滤小于此值的样本')
    parser.add_argument('--max_grad_norm', type=float, default=5.0,
                        help='梯度裁剪的最大范数')
    args = parser.parse_args()
    
    # --- B. 设置设备 ---
    if args.device == 'auto':
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device(args.device)
    print(f"正在使用设备: {DEVICE}")
    
    # --- C. 加载数据 ---
    print(f"\n正在从 '{args.data}' 加载数据...")
    dataloaders, adj_matrix, scaler = get_dataloaders(args.data, args.batch_size)
    adj_matrix = adj_matrix.to(DEVICE)
    scaler = scaler.to(DEVICE)
    N_NODES = adj_matrix.shape[0]
    
    # ⚠️ 打印数据统计信息
    print("\n" + "="*60)
    print("数据统计信息:")
    # 打印形状而不是 .item()，以兼容 PerNodeScaler
    print(f"  - Scaler Mean 形状: {scaler.mean.shape}")
    print(f"  - Scaler Std 形状: {scaler.std.shape}")
    
    # 检查一个batch
    sample_x, sample_y = next(iter(dataloaders['train']))
    # ⚠️ 修复: 将sample_y移动到与scaler相同的设备
    sample_y = sample_y.to(DEVICE)
    sample_y_real = scaler.inverse_transform(sample_y.squeeze(-1))
    print(f"  - 训练集真实值范围: [{sample_y_real.min().item():.2f}, {sample_y_real.max().item():.2f}]")
    print(f"  - MAPE Epsilon: {args.mape_epsilon} (过滤掉小于此值的样本)")
    print("="*60 + "\n")
    
    INPUT_DIM = 1
    OUT_SEQ_LEN = 12
    
    # --- D. 构建模型 ---
    print("正在构建模型...")
    model = DGCRN_Model(
        num_nodes=N_NODES,
        static_adj=adj_matrix,
        input_dim=INPUT_DIM,
        hidden_dim=args.hidden_dim,
        out_seq_len=OUT_SEQ_LEN,
        num_layers=args.num_layers,
        embed_dim=args.embed_dim
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}\n")
    
    # --- E. 设置优化器和损失函数 ---
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # ⚠️ 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    
    criterion = masked_mae_loss 
    
    # --- F. 训练循环 ---
    print("开始训练...")
    print("="*60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        start_time = time.time()
        
        # 1. 训练
        train_loss = train_epoch(
            model, dataloaders['train'], optimizer, criterion, DEVICE, 
            max_grad_norm=args.max_grad_norm
        )
        
        # 2. 验证
        val_metrics = eval_epoch(
            model, dataloaders['val'], scaler, DEVICE, 
            epsilon=args.mape_epsilon
        )
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        # 3. 更新学习率
        scheduler.step(val_metrics['overall_mae'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # 4. 打印结果
        print(f"Epoch {epoch} 耗时: {epoch_time:.2f} 秒 | 学习率: {current_lr:.6f}")
        print(f"训练损失 (MAE, 归一化): {train_loss:.4f}")
        print("验证指标 (真实值):")
        print(f"  [Overall] MAE: {val_metrics['overall_mae']:.4f} | MAPE: {val_metrics['overall_mape']:.2f}% | RMSE: {val_metrics['overall_rmse']:.4f}")
        print(f"  [15 min]  MAE: {val_metrics['15min_mae']:.4f} | MAPE: {val_metrics['15min_mape']:.2f}% | RMSE: {val_metrics['15min_rmse']:.4f}")
        print(f"  [30 min]  MAE: {val_metrics['30min_mae']:.4f} | MAPE: {val_metrics['30min_mape']:.2f}% | RMSE: {val_metrics['30min_rmse']:.4f}")
        print(f"  [60 min]  MAE: {val_metrics['60min_mae']:.4f} | MAPE: {val_metrics['60min_mape']:.2f}% | RMSE: {val_metrics['60min_rmse']:.4f}")
        
        # ⚠️ 异常检测
        if val_metrics['overall_mape'] > 100:
            print(f"⚠️ 警告: MAPE={val_metrics['overall_mape']:.2f}% 异常高！")
            print("   建议检查:")
            print("   1. 数据归一化是否正确")
            print("   2. 是否需要调整 --mape_epsilon")
            print("   3. 模型是否发散 (检查loss)")
        
        # 5. 保存最佳模型
        current_val_loss = val_metrics['overall_mae']
        
        if current_val_loss < best_val_loss:
            print(f"✓ 验证 MAE 从 {best_val_loss:.4f} 降低到 {current_val_loss:.4f}。正在保存模型...")
            best_val_loss = current_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': current_val_loss,
                'val_mape': val_metrics['overall_mape'],
            }, args.save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"验证 MAE 没有改善。Patience: {patience_counter}/{args.patience}")
        
        # 每10轮保存检查点
        if epoch % 10 == 0:
            base_name, ext = os.path.splitext(args.save_path)
            checkpoint_path = f"{base_name}_epoch_{epoch}{ext}"
            print(f"达到第 {epoch} 轮，正在保存检查点到: {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
        
        # 6. 早停检查
        if patience_counter >= args.patience:
            print(f"\n达到早停耐心 ({args.patience})。终止训练。")
            break
    
    # --- G. 最终测试 ---
    print("\n" + "="*60)
    print("训练完成。正在加载最佳模型并在测试集上运行...")
    
    checkpoint = torch.load(args.save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载了第 {checkpoint['epoch']} 轮的模型 (验证MAE: {checkpoint['val_mae']:.4f})")
    
    test_metrics = eval_epoch(
        model, dataloaders['test'], scaler, DEVICE,
        epsilon=args.mape_epsilon
    )
    
    print("\n" + "="*60)
    print("--- 最终测试结果 (最佳模型) ---")
    print("="*60)
    print(f"  [Overall] MAE: {test_metrics['overall_mae']:.4f} | MAPE: {test_metrics['overall_mape']:.2f}% | RMSE: {test_metrics['overall_rmse']:.4f}")
    print(f"  [15 min]  MAE: {test_metrics['15min_mae']:.4f} | MAPE: {test_metrics['15min_mape']:.2f}% | RMSE: {test_metrics['15min_rmse']:.4f}")
    print(f"  [30 min]  MAE: {test_metrics['30min_mae']:.4f} | MAPE: {test_metrics['30min_mape']:.2f}% | RMSE: {test_metrics['30min_rmse']:.4f}")
    print(f"  [60 min]  MAE: {test_metrics['60min_mae']:.4f} | MAPE: {test_metrics['60min_mape']:.2f}% | RMSE: {test_metrics['60min_rmse']:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()