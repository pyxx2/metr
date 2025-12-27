import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TrafficDataset(Dataset):
    """
    一个为 .npz 文件设计的自定义 PyTorch Dataset 类。
    (此版本兼容 PerNodeScaler)
    """
    def __init__(self, data, data_type):
        """
        初始化 Dataset。
        :param data: 从 np.load() 加载的 .npz 文件数据。
        :param data_type: 'train', 'val', 或 'test'。
        """
        print(f"正在为 '{data_type}' 初始化 TrafficDataset...")
        
        # 根据 data_type 加载正确的数据
        if data_type == 'train':
            self.X = data['X_train']
            self.Y = data['Y_train']
        elif data_type == 'val':
            self.X = data['X_val']
            self.Y = data['Y_val']
        elif data_type == 'test':
            self.X = data['X_test']
            self.Y = data['Y_test']
        else:
            raise ValueError("data_type 必须是 'train', 'val', 或 'test' 之一。")

        # 确保数据是 float32 (与 .astype(np.float32) 对应)
        self.X = torch.from_numpy(self.X).float()
        self.Y = torch.from_numpy(self.Y).float()
        
        print(f"  X 形状 (PyTorch 张量): {self.X.shape}")
        print(f"  Y 形状 (PyTorch 张量): {self.Y.shape}")
        print("初始化完成。")

    def __len__(self):
        """
        返回数据集中的样本总数。
        """
        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        获取一个样本。
        :param idx: 样本的索引。
        :return: (x, y) 元组，其中 x 和 y 都是张量。
        """
        x_sample = self.X[idx]
        y_sample = self.Y[idx]
        return x_sample, y_sample

def get_dataloaders(npz_file_path, batch_size, num_workers=4):
    """
    (主函数) 创建训练、验证和测试的 DataLoader。
    
    :param npz_file_path: 指向 '..._processed.npz' 的路径。
    :param batch_size: 批次大小。
    :param num_workers: DataLoader 使用的子进程数。
    :return: 包含 data_loaders (字典) 和 scaler (字典) 的字典。
    """
    print(f"正在从 '{npz_file_path}' 加载数据...")
    
    # --- 1. 加载 .npz 文件 ---
    try:
        data = np.load(npz_file_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {npz_file_path}。")
        return None
    
    # --- 2. 创建 Datasets ---
    train_dataset = TrafficDataset(data, 'train')
    val_dataset = TrafficDataset(data, 'val')
    test_dataset = TrafficDataset(data, 'test')
    
    # --- 3. 创建 DataLoaders ---
    print("\n正在创建 DataLoaders...")
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    print(f"  训练集 batches: {len(train_loader)}")
    print(f"  验证集 batches: {len(val_loader)}")
    print(f"  测试集 batches: {len(test_loader)}")

    # --- 4. 提取邻接矩阵和归一化参数 ---
    
    # 将邻接矩阵转换为 PyTorch 张量
    adj_matrix = torch.from_numpy(data['adj_matrix']).float()
    
    # 创建 scaler 字典
    scaler_params = {
        'mean': data['scaler_mean'],
        'std': data['scaler_std']
    }
    
    # (PerNodeScaler "对象")
    
    class SimpleScaler:
        def __init__(self, mean, std):
            # (*** 关键修复 ***)
            # 加载 (1, N, 1) 并 squeeze 掉最后一个维度，
            # 得到 (1, N) (例如 (1, 207))
            # 这对于与 (B, T, N) 形状的数据进行广播是必需的
            self.mean = torch.tensor(mean).float().squeeze(-1)
            self.std = torch.tensor(std).float().squeeze(-1)
            
            # (已修复的 print 语句，兼容 PerNodeScaler)
            print(f"\nPerNodeScaler 已创建: Mean shape={self.mean.shape}, Std shape={self.std.shape}")

        def to(self, device):
            """将 mean 和 std 移动到 GPU"""
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
            return self
            
        def inverse_transform(self, data):
            """
            反归一化 (Z-Score)
            data 形状: (B, T, N)
            self.std 形状: (1, N)
            PyTorch 会自动广播
            """
            return (data * self.std) + self.mean

    scaler_obj = SimpleScaler(scaler_params['mean'], scaler_params['std'])
    
    # --- 5. 返回所有内容 ---
    
    data_loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    return data_loaders, adj_matrix, scaler_obj


# --- 示例用法 (您可以取消注释以测试此文件) ---
if __name__ == '__main__':
    
    # 假设 'METR-LA_processed.npz' 在同一目录中
    NPZ_PATH = 'METR-LA_processed.npz'
    BATCH_SIZE = 64

    # 1. 获取 DataLoaders
    try:
        loaders, adj_mx, scaler = get_dataloaders(NPZ_PATH, BATCH_SIZE)

        # 2. 检查一个批次
        print("\n" + "="*30)
        print("检查一个训练批次...")
        
        x_batch, y_batch = next(iter(loaders['train']))
        
        # 打印形状
        print(f"  X_batch 形状: {x_batch.shape}")
        print(f"  Y_batch 形状: {y_batch.shape}")
        
        # 预期形状: (取决于您的 stride)
        # (B, 12, 207, 1)
        # (B, 12, 207, 1)

        # 3. 检查邻接矩阵和 Scaler
        print("\n" + "="*30)
        print("检查邻接矩阵和 Scaler...")
        print(f"  邻接矩阵形状: {adj_mx.shape}")
        print(f"  Scaler 均值形状: {scaler.mean.shape}") # 应为 (1, 207)
        
        # 4. 检查反归一化 (关键测试)
        y_batch_real = scaler.inverse_transform(y_batch.squeeze(-1))
        print(f"\n  Y_batch (归一化) 均值: {y_batch.mean().item():.4f}")
        print(f"  Y_batch (反归一化) 均值: {y_batch_real.mean().item():.4f}")
        print(f"  Y_batch (反归一化) 形状: {y_batch_real.shape}")
        print("  (反归一化均值应接近 METR-LA 的真实均值 ~55-60)")
        print("  (反归一化形状应为 [B, T_out, N])")

    except FileNotFoundError:
        print(f"\n请确保 '{NPZ_PATH}' 位于此目录中以运行测试。")
    except RuntimeError as e:
        if "num_workers" in str(e):
             print("\n注意: 在主脚本中运行 DataLoader 测试可能会在某些系统上引发 'num_workers' 错误。")
             print("这在 PyTorch 训练脚本中通常不是问题。")
             print("您可以尝试将 get_dataloaders() 中的 num_workers 设置为 0 来进行此测试。")
        else:
             raise e