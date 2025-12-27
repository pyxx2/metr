import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==================== 辅助函数 (保持不变) ====================
def calculate_normalized_laplacian(adj):
    adj_with_selfloop = adj + torch.eye(adj.shape[0], device=adj.device)
    d = torch.sum(adj_with_selfloop, dim=1)
    d_inv_sqrt = torch.pow(d, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = torch.mm(torch.mm(d_mat_inv_sqrt, adj_with_selfloop), d_mat_inv_sqrt)
    return normalized_laplacian

def batch_calculate_normalized_laplacian(adj_batch):
    B, N, _ = adj_batch.shape
    I = torch.eye(N, device=adj_batch.device).unsqueeze(0).expand_as(adj_batch)
    adj_with_selfloop = adj_batch + I
    d_batch = torch.sum(adj_with_selfloop, dim=2)
    d_inv_sqrt_batch = torch.pow(d_batch, -0.5)
    d_inv_sqrt_batch[torch.isinf(d_inv_sqrt_batch)] = 0.
    d_mat_inv_sqrt_batch = torch.diag_embed(d_inv_sqrt_batch)
    L_batch = torch.bmm(torch.bmm(d_mat_inv_sqrt_batch, adj_with_selfloop), d_mat_inv_sqrt_batch)
    return L_batch

def batch_on_batch_graph_conv(x, adj_batch):
    return torch.einsum('bnm,bmd->bnd', adj_batch, x)

# ==================== 模块 1: Cell (保持不变) ====================
class FullyBatchedGCRNCell(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, static_adj, embed_dim=32):
        super(FullyBatchedGCRNCell, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.register_buffer('L_static', calculate_normalized_laplacian(static_adj))
        self.graph_fusion_alpha = nn.Parameter(torch.tensor(0.5))
        self.E_s = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.E_t = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.fc_dynamic_s = nn.Linear(input_dim + hidden_dim, embed_dim)
        self.fc_dynamic_t = nn.Linear(input_dim + hidden_dim, embed_dim)
        self.dynamic_scale = nn.Parameter(torch.tensor(0.1))
        gate_input_dim = input_dim + hidden_dim
        self.fc_r = nn.Linear(gate_input_dim, hidden_dim)
        self.fc_z = nn.Linear(gate_input_dim, hidden_dim)
        self.fc_h = nn.Linear(gate_input_dim, hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.E_s)
        nn.init.xavier_uniform_(self.E_t)

    def generate_time_varying_graph(self, x_t, h_t_minus_1):
        B, N, _ = x_t.shape
        A_base = torch.mm(self.E_s, self.E_t.T).unsqueeze(0).expand(B, -1, -1)
        context = torch.cat([x_t, h_t_minus_1], dim=-1)
        D_s = self.fc_dynamic_s(context)
        D_t = self.fc_dynamic_t(context)
        A_offset = torch.einsum('bnd,bmd->bnm', D_s, D_t)
        A_raw = F.relu(A_base + self.dynamic_scale * A_offset)
        return batch_calculate_normalized_laplacian(F.softmax(A_raw, dim=2))

    def forward(self, x_t, h_t_minus_1):
        L_t = self.generate_time_varying_graph(x_t, h_t_minus_1)
        alpha = torch.sigmoid(self.graph_fusion_alpha)
        A_fused = alpha * self.L_static.unsqueeze(0) + (1 - alpha) * L_t
        x_h = torch.cat([x_t, h_t_minus_1], dim=-1)
        r = torch.sigmoid(batch_on_batch_graph_conv(self.fc_r(x_h), A_fused))
        z = torch.sigmoid(batch_on_batch_graph_conv(self.fc_z(x_h), A_fused))
        h_tilde = torch.tanh(batch_on_batch_graph_conv(self.fc_h(torch.cat([x_t, r * h_t_minus_1], dim=-1)), A_fused))
        return (1 - z) * h_t_minus_1 + z * h_tilde

# ==================== 模块 2: Encoder (修改处：w/o Attention) ====================
class EnhancedGCRN_Encoder(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, static_adj, num_layers=2, embed_dim=32, FullyBatchedGCRNCell=None):
        super(EnhancedGCRN_Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.cells = nn.ModuleList([
            FullyBatchedGCRNCell(num_nodes, input_dim if i==0 else hidden_dim, hidden_dim, static_adj, embed_dim)
            for i in range(num_layers)
        ])
        
        # 🔴 [修改 1] 注释掉注意力层定义
        # self.temporal_attention = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 4), nn.Tanh(), nn.Linear(hidden_dim // 4, 1)
        # )
        print("!!! Ablation Mode: Temporal Attention Pooling REMOVED !!!")

    def forward(self, X_seq):
        B, S, N, D = X_seq.shape
        h_list = [torch.zeros(B, N, self.hidden_dim, device=X_seq.device) for _ in range(self.num_layers)]
        X_seq = X_seq.permute(1, 0, 2, 3) 
        h_sequence = []
        
        for t in range(S):
            x = X_seq[t]
            new_h_list = []
            for l, cell in enumerate(self.cells):
                h = cell(x, h_list[l])
                new_h_list.append(h)
                x = h
            h_list = new_h_list
            h_sequence.append(h_list[-1])
        
        h_sequence = torch.stack(h_sequence, dim=1) # (B, T, N, D)
        
        # 🔴 [修改 2] 去掉注意力计算，直接取最后一个时间步的输出
        # 原代码：
        # attn = F.softmax(self.temporal_attention(h_sequence), dim=1)
        # h_pooled = (h_sequence * attn).sum(dim=1)
        
        # 修改为：取最后一个时间步 (Last Hidden State)
        # h_sequence shape: [Batch, Time, Node, Dim]
        # 我们取 Time 维度的最后一个索引 -1
        h_pooled = h_sequence[:, -1, :, :]  # Shape: (B, N, D)
        
        # 备选方案：如果你想做平均池化 (Mean Pooling)，可以用下面这行代替上面那行
        # h_pooled = h_sequence.mean(dim=1) 

        return h_pooled, h_sequence

# ==================== 模块 3: Decoder (保持不变) ====================
class ResidualMLPDecoder(nn.Module):
    def __init__(self, hidden_dim, out_seq_len, num_nodes):
        super(ResidualMLPDecoder, self).__init__()
        self.out_seq_len = out_seq_len
        self.hidden_dim = hidden_dim
        self.time_embedding = nn.Embedding(out_seq_len, hidden_dim)
        self.main_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, out_seq_len, kernel_size=1)
        )
        self.residual_branch = nn.Conv2d(hidden_dim, out_seq_len, kernel_size=1)
        self.fusion_alpha = nn.Parameter(torch.tensor(0.7))
    
    def forward(self, h_T):
        B, N, D = h_T.shape
        time_ids = torch.arange(self.out_seq_len, device=h_T.device)
        time_emb = self.time_embedding(time_ids).mean(dim=0, keepdim=True)
        h_T_enhanced = h_T + time_emb.unsqueeze(0)
        h_T_enhanced = h_T_enhanced.permute(0, 2, 1).unsqueeze(-1)
        main_output = self.main_branch(h_T_enhanced)
        residual_output = self.residual_branch(h_T_enhanced)
        alpha = torch.sigmoid(self.fusion_alpha)
        output = alpha * main_output + (1 - alpha) * residual_output
        return output

# ==================== 模块 4: 主模型 (保持不变) ====================
class DGCRN_Model(nn.Module):
    def __init__(self, num_nodes, static_adj, input_dim, hidden_dim, out_seq_len, 
                 num_layers=2, embed_dim=32, decoder_type='residual'):
        super(DGCRN_Model, self).__init__()
        # 使用修改后的无注意力 Encoder
        self.encoder = EnhancedGCRN_Encoder(num_nodes, input_dim, hidden_dim, static_adj, 
                                            num_layers, embed_dim, FullyBatchedGCRNCell)
        
        self.decoder = ResidualMLPDecoder(hidden_dim, out_seq_len, num_nodes)
        
        print(f"✓ 模型构建完成: ResidualMLPDecoder (Ablation: w/o Attention)")

    def forward(self, X_seq):
        h_pooled, _ = self.encoder(X_seq)
        Y_pred = self.decoder(h_T=h_pooled)
        return Y_pred.squeeze(-1)

if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DGCRN_Model(207, torch.rand(207,207), 1, 64, 12).to(DEVICE)
    x = torch.randn(64, 12, 207, 1).to(DEVICE)
    try:
        y = model(x)
        print("✓ 测试通过")
        print("输出形状:", y.shape) # 应为 (64, 12, 207)
    except Exception as e:
        print(f"✗ 错误: {e}")