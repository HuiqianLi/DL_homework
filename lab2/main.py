import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import collections
import os
import random
import tarfile
import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 解压数据
def Decompress(DATA_ORIGIN_ROOT):
    fname = os.path.join(DATA_ORIGIN_ROOT, 'aclImdb_v1.tar.gz')
    # 将压缩文件进行解压
    if not os.path.exists(os.path.join(DATA_ORIGIN_ROOT, 'aclImdb')):
        print("从压缩包解压...")
        with tarfile.open(fname, 'r') as f:
            f.extractall(DATA_ORIGIN_ROOT) # 解压文件到此指定路径
    return DATA_ORIGIN_ROOT+'/aclImdb/'

# 2. 读取数据
def readData(folder, DATA_ROOT):
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(DATA_ROOT, folder, label) # 拼接文件路径
        for file in os.listdir(folder_name): # 读取文件路径下的所有文件名，并存入列表中
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', ' ').lower()
                data.append([review, 1 if label == 'pos' else 0]) # 将每个文本读取的内容和对应的标签存入data列表中
    random.seed(0)      # 设置随机数种子，保证每次生成的结果都是一样的
    random.shuffle(data) # 打乱data列表中的数据排列顺序
    return data

# 2.1 划分训练集和验证集
def split_train_val(data, val_ratio):
    val_set_size = int(len(data) * val_ratio)
    return data[val_set_size:], data[:val_set_size]

# 2.2 预处理数据
# 空格分词
def get_tokenized_imdb(data):
    '''
    :param data: list of [string, label]
    '''
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data] # 只从data中读取review(评论)内容而不读取标签(label)，对review使用tokenizer方法进行分词

# 创建词典
def get_vocab_imdb(data):
    tokenized_data = get_tokenized_imdb(data) # 调用get_tokenized_imdb()空格分词方法, 获取到分词后的数据tokenized_data
    counter = collections.Counter([tk for st in tokenized_data for tk in st]) # 读取tokenized_data列表中每个句子的每个词，放入列表中。
    specials = ['<unk>']
    return Vocab.Vocab(counter, min_freq=5, specials=specials) # 去掉词频小于5的词

# 对data列表中的每行数据进行处理，将词转换为索引，并使每行数据等长
def process_imdb(data, vocab):
    max_len = 500 # 每条评论通过截断或者补0，使得长度变成500
    def pad(x):
        # x[:max_len] 只获取前max_len个词
        # x + [0]*(max_len - len(x)) 词数小于max_len, 用pad=0补长到max_len
        return x[:max_len] if len(x) > max_len else x + [0]*(max_len - len(x)) 
    tokenized_data = get_tokenized_imdb(data) # 调用方法获取分词后的数据
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data]) # 将词转换为vocab词典中对应词的索引
    labels = torch.tensor([score for _, score in data])
    return features, labels

# 2.3 创建数据迭代器
def data(DATA_ROOT, batch_size = 64):
    train_val, test_data = readData('train', DATA_ROOT), readData('test', DATA_ROOT)
    train_data, val_data = split_train_val(train_val, 0.2)
    print(len(train_data), "train +", len(val_data), "val +", len(test_data), "test")
    vocab = get_vocab_imdb(train_data)
    # print(len(vocab))
    # print(vocab.get_stoi()['hello'])
    train_set = Data.TensorDataset(*process_imdb(train_data, vocab))
    val_set = Data.TensorDataset(*process_imdb(val_data, vocab))
    test_set = Data.TensorDataset(*process_imdb(test_data, vocab))
    train_iter = Data.DataLoader(train_set, batch_size, True)
    val_iter = Data.DataLoader(val_set, batch_size, True)
    test_iter = Data.DataLoader(test_set, batch_size)
    return train_iter, val_iter, test_iter, vocab

# 3. 创建循环神经网络
# 在下面实现的BiRNN类中，Embedding实例即嵌入层，LSTM实例即为序列编码的隐藏层，Linear实例即生成分类结果的输出层。
class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.encoder = nn.LSTM(
            input_size=embed_size,
            hidden_size=num_hiddens,
            num_layers=num_layers,
            # batch_first=True,
            bidirectional=True # bidirectional设为True即得到双向循环神经网络
        )
        self.decoder = nn.Linear(4*num_hiddens, 2)

    def forward(self, inputs):
        # inputs: [batch_size, seq_len], LSTM需要将序列长度(seq_len)作为第一维，所以需要将输入转置后再提取词特征
        # 输出形状 outputs: [seq_len, batch_size, embedding_dim]   embedding_dim词向量维度
        embeddings = self.embedding(inputs.permute(1, 0))
        # rnn.LSTM只传入输入embeddings, 因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(seq_len, batch_size, 2*num_hiddens)
        outputs, _ = self.encoder(embeddings)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。
        # 它的形状为 : [batch_size, 4 * num_hiddens]
        encoding = torch.cat((outputs[0], outputs[-1]), dim=-1)
        outs = self.decoder(encoding)
        return outs

# 4. 加载预训练的词向量
def load_pretrained_embedding(words, pretrained_vocab):
    '''从训练好的vocab中提取出words对应的词向量'''
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0]) # pretrained_vocab.vectors[0].shape # torch.Size([100])
    oov_count = 0 # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx] # 将第i行用预训练的单词向量替换
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print("There are %d oov words." % oov_count)
    return embed

# 5. 评估函数
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

# 6. 训练网络
def train(vocab, train_iter, val_iter, device, lr, epochs, wordEmbedding_dim):
    # 定义神经网络   
    net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
    print(net)

    # 为词典vocab中的每个词加载dim=100维的GloVe词向量
    glove_vocab = Vocab.GloVe(name='6B', dim=wordEmbedding_dim, cache=os.path.join(DATA_ROOT, 'glove'))
    # print(len(glove_vocab.stoi)) # 400000
    # print(glove_vocab[0].shape)
    net.embedding.weight.data.copy_(
        load_pretrained_embedding(vocab.itos, glove_vocab)
    )
    net.embedding.weight.requires_grad = False # 直接加载预训练好的，所以不需要更新它 

    # 是否gpu训练
    net = net.to(device)
    print("training on ", device)

    # 优化器和损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr)
    loss = torch.nn.CrossEntropyLoss()

    # 网络训练过程。随机梯度下降，设置学习率为lr，迭代epoch次
    batch_count = 0
    train_accs, val_accs = [], []
    Note=open('/home/lihuiqian/hw/lab2/preds4.txt',mode='w')
    for epoch in range(epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        val_acc = evaluate_accuracy(val_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, val acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, val_acc, time.time() - start))
            # 统计预测框（漏检率误诊率用）
        Note.write(str('epoch %d, loss %.4f, train acc %.3f, val acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, val_acc, time.time() - start))+'\n')
        train_accs.append(train_acc_sum / n)
        val_accs.append(val_acc)
    Note.close()
    plt.cla()
    plt.plot(range(epochs), train_accs, 'r-', lw=2)
    plt.plot(range(epochs), val_accs, 'b-', lw=2)
    plt.xlabel('epoches')
    plt.ylabel('Train acc (red), Val acc (blue)')
    plt.savefig("/home/lihuiqian/hw/lab2/train.jpg")
    return net

if __name__ == '__main__':
    # 解压文件
    # DATA_ORIGIN_ROOT = '/home/lihuiqian/hw/lab2'
    # DATA_ROOT = Decompress(DATA_ORIGIN_ROOT)

    DATA_ROOT = '/home/lihuiqian/hw/lab2/aclImdb/'
    
    # 数据读取
    train_iter, val_iter, test_iter, vocab = data(DATA_ROOT, batch_size = 64)

    # 训练
    embed_size, num_hiddens, num_layers = 100, 100, 4 # 网络参数
    lr, epochs = 0.005, 100   # 超参数
    wordEmbedding_dim = 100   # 词向量维度
    net = train(vocab, train_iter, val_iter, device, lr, epochs, wordEmbedding_dim)

    # 测试
    test_acc = evaluate_accuracy(test_iter, net, device)
    print('test_acc: %.3f' % test_acc)