import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
# from conv import GraphConvolution as GCNConv
from torch_geometric.nn import GCNConv, PairNorm
from torch_geometric.utils import dropout_adj
from utils import *

# 1.加载数据，数据集划分 Cora or Citeseer
dataset = Planetoid('/data/lhq/code/gcn/dataset', name='Citeseer', transform=T.NormalizeFeatures()) # 包括数据集的下载，若root路径存在数据集则直接加载数据集
data = dataset[0] #该数据集只有一个图len(dataset)

features = get_normalize_features(data.x).cuda()
adj = edge_to_adj(data.edge_index, data.num_nodes).cuda()

activations = {
    'relu': torch.relu,
    'sigmoid': torch.sigmoid,
    'tanh': torch.tanh,
    }

# 2.设计网络
class GCN(torch.nn.Module):
    # 初始化
    def __init__(self, hidden_channels, n_layers, act: str = 'relu', add_self_loops: bool = True,\
                    pair_norm: bool = True, dropout: float = .0, drop_edge: float =.0):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.drop_edge = drop_edge
        self.pair_norm = pair_norm
        self.act = activations[act] if isinstance(act, str) else act
        self.conv_list = torch.nn.ModuleList()
        for i in range(n_layers):
            in_c, out_c = hidden_channels, hidden_channels
            if i == 0:
                in_c = dataset.num_features
            elif i == n_layers - 1:
                out_c = dataset.num_classes
            self.conv_list.append(GCNConv(in_c, out_c, add_self_loops=add_self_loops))
    def forward(self, x, edge_index):
        edge_index, _ = dropout_adj(edge_index, p=self.drop_edge)
        for i, conv in enumerate(self.conv_list):
            x = conv(x, edge_index)
            if self.pair_norm:
                x = PairNorm()(x)
            if i < len(self.conv_list) - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# 3.训练模型
def train(model,data,optimizer,criterion):
    model.train()
    optimizer.zero_grad()  # 梯度置零
    out = model(data.x, data.edge_index)  # 模型前向传播
    loss = criterion(out[data.train_mask],data.y[data.train_mask])  # 计算loss
    loss.backward()  # 反向传播
    optimizer.step()  # 优化器梯度下降
    return loss

# 4.测试
@torch.no_grad()
def test(model,data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # 使用最大概率的类别作为预测结果
    val_correct = pred[data.val_mask] == data.y[data.val_mask]  # 获取正确标记的节点
    val_acc = int(val_correct.sum()) / int(data.val_mask.sum()) # 计算正确率
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # 获取正确标记的节点
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum()) # 计算正确率
    return val_acc, test_acc


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(hidden_channels=16, n_layers=2, act='tanh', add_self_loops=True, \
                pair_norm=False, dropout=.1, drop_edge=.5).to(device) # 实例化模型
    data = data.to(device)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # 选择loss，二值交叉熵
    criterion = torch.nn.CrossEntropyLoss()
    best_test_acc = 0
    # 训练、验证和测试保存模型
    for epoch in range(1,101):
        loss = train(model,data,optimizer,criterion)
        val_acc, test_acc = test(model,data)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    print(f'Test: {best_test_acc:.4f}')

    # import matplotlib.pyplot as plt
    # from sklearn.manifold import TSNE

    # #可视化
    # def visualize(out, color):
    #     z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
    #     plt.figure(figsize=(10,10))
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    #     plt.savefig('/data/lhq/code/gcn/show.png')
    #     plt.close()
        
    # model.eval()
    # out = model(data.x, data.edge_index)
    # visualize(out, color=data.y.cpu().numpy())