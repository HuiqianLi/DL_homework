import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

# 参考 https://zhuanlan.zhihu.com/p/114980874

def data(size):
    # 生成数据, size个等距, dim=1 1维的数据转换成2维
    x = torch.linspace(0,4*np.pi,size)
    x = torch.unsqueeze(x,dim=1)
    y = np.sin(x) + np.exp(-x)
    # 生成x,y; 按8:1:1划分
    x_train, x_val, x_test = torch.utils.data.random_split(x, [int(0.8*size), int(0.1*size), int(0.1*size)],torch.manual_seed(0))
    y_train, y_val, y_test = torch.utils.data.random_split(y, [int(0.8*size), int(0.1*size), int(0.1*size)],torch.manual_seed(0))
    x_train, x_val, x_test = x[x_train.indices], x[x_val.indices], x[x_test.indices]
    y_train, y_val, y_test = y[y_train.indices], y[y_val.indices], y[y_test.indices]
    
    # 将tensor置入Variable中
    x_train,y_train,x_val,y_val,x_test,y_test =(Variable(x_train),Variable(y_train),
        Variable(x_val),Variable(y_val),Variable(x_test),Variable(y_test))
    
    # 数据可视化，此处打印val的数据集是因为113行x_train,y_train,x_test,y_test,x_val,y_val = data(5000)不小心将test和val设置反了。。。
    plt.scatter(x_val.data,y_val.data)
    # 或者采用如下的方式也可以输出x,y
    # plt.scatter(x.data.numpy(),y.data.numpy())
    plt.savefig("/home/lihuiqian/hw/lab1/fig/data.jpg")

    return x_train,y_train,x_val,y_val,x_test,y_test

# 搭建网络FNN，两个全连接层组成的隐藏层
class Net(nn.Module): # 继承nn.Module
    def __init__(self,n_input,n_hidden,n_output,num_layers,activation):
        super(Net,self).__init__() # 获得Net类的父类的构造方法
        # 定义每层结构形式，num_layers个隐藏层
        activation = activation.lower()
        act_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'leakyrelu': nn.LeakyReLU,
        }
        self.fc_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.fc_layers.append(nn.Linear(n_input,n_hidden)) # 第一个隐藏层
        for i in range(num_layers-1):
            self.fc_layers.append(nn.Linear(n_hidden,n_hidden))
            if i < num_layers - 1:
                self.activations.append(act_map[activation]())
        self.fc_layers.append(nn.Linear(n_hidden,n_output)) # 预测层
    # 将各层的神经元搭建成完整的神经网络的前向通路
    def forward(self,input):
        x = input
        for i, layer in enumerate(self.fc_layers):
            x = layer(x)
            if i < len(self.activations):
                x = self.activations[i](x) # 对隐藏层激活
        return x

def train(hidde_size,num_layers,activation,epoch,batch_size,lr,x,y,x_val,y_val):
    # 定义神经网络，打印输出net的结构（隐藏层20个节点）   
    net = Net(1,hidde_size,1,num_layers,activation)
    print(net)

    # 加载数据
    # batch_size = 32
    dataset = TensorDataset(x,y)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    valset = TensorDataset(x_val,y_val)
    validation_data = DataLoader(valset, 1, shuffle=True)

    # 优化器和损失函数
    optimizer = torch.optim.SGD(net.parameters(),lr)
    loss_func = torch.nn.MSELoss()

    # 网络训练过程。随机梯度下降，设置学习率为0.1，迭代epoch次
    # epoch = 1000
    mean_losses,val_losses = [],[]
    for t in range(epoch):
        train_loss = 0  # 统计loss
        for x, y in dataloader:
            prediction = net(x) # 数据x传给net，输出预测值
            loss = loss_func(prediction,y) # 计算误差，注意参数顺序
            optimizer.zero_grad()  # 清空上一步的更新参数值
            loss.backward()        # 误差反向传播
            optimizer.step()       # 计算得到的更新值赋给net.parameters()
            train_loss += loss.item() # 统计loss

        with torch.no_grad(): # 计算验证集loss
            val_loss = 0    # 统计loss
            for val_x, val_y in validation_data:
                out = net(val_x)
                loss = loss_func(out, val_y)
                val_loss += loss.item()
            val_loss = round(val_loss/len(validation_data),6)

        mean_loss = round(train_loss/len(dataloader),6)
        print('epoch:', t, ', loss:', mean_loss,', val_loss:', val_loss)
        mean_losses.append(mean_loss)
        val_losses.append(val_loss)
    plt.cla()
    plt.plot(range(epoch), mean_losses, 'r-', lw=2)
    plt.plot(range(epoch), val_losses, 'b-', lw=2)
    plt.xlabel('epoches')
    plt.ylabel('Train loss (red), Val loss (blue)')
    plt.savefig("/home/lihuiqian/hw/lab1/fig/train.jpg")
    return net

def test(net,x,y):
    dataset = TensorDataset(x,y)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    loss_func = torch.nn.MSELoss()
    for x, y in dataloader:
        prediction = net(x) # 数据x传给net，输出预测值
        loss = loss_func(prediction,y) # 计算误差，注意参数顺序
        print(loss.item())
        # 可视化训练过程
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.scatter(x.data.numpy(), prediction.data.numpy(), c='r')
    plt.suptitle('test: Loss = %.4f'% loss.data, fontsize = 20)
    plt.ioff()
    plt.xlabel('x')
    plt.ylabel('y or pred')
    plt.savefig("/home/lihuiqian/hw/lab1/fig/test.jpg")
    return 0

if __name__ == '__main__':
    x_train,y_train,x_test,y_test,x_val,y_val = data(5000)
    hidde_size = 20     # 隐藏层节点数(大小，宽度)
    num_layers = 5      # 隐藏层数量(深度)
    activation = 'relu' # 激活函数
    epoch = 100         # 训练轮次
    batch_size = 16     # 批大小
    lr = 1e-2           # 学习率
    net = train(hidde_size,num_layers,activation,epoch,batch_size,lr,x_train,y_train,x_val,y_val)
    # test(net,x_val,y_val)
    test(net,x_test,y_test)