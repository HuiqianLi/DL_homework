import os
import torch
import random
import numpy as np
from torchtext import data
from torchtext import datasets
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
import time

# 设置随机种子
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# 加载预先训练的bert-base-uncased标记器
tokenizer = BertTokenizer.from_pretrained('/home/lhq/hw/lab3/bert-base-uncased')
# print(len(tokenizer.vocab))

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id
# print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)

max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
# print(max_input_length)

# 最大长度比实际的最大长度小2。因为需要向每个序列添加两个标记，一个开始一个结束。
def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[ :max_input_length-2]
    return tokens

# 定义字段
TEXT = data.Field(
        batch_first=True,
        use_vocab=False,
        tokenize=tokenize_and_cut,
        preprocessing=tokenizer.convert_tokens_to_ids,
        init_token=init_token_idx,
        eos_token=eos_token_idx,
        unk_token=unk_token_idx,
        pad_token=pad_token_idx
)
LABEL = data.LabelField(dtype=torch.float)

# 加载数据并创建验证分割
train_data, test_data = datasets.IMDB.splits(
        text_field=TEXT,
        label_field=LABEL
)
train_data, valid_data = train_data.split(random_state = random.seed(SEED))
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

# 为标签构建词汇表
LABEL.build_vocab(train_data)
# print(LABEL.vocab.stoi)

# 创建数据迭代器
BATCH_SIZE = 2 # 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)

# 加载预先训练过的模型
bert = BertModel.from_pretrained('/home/lhq/hw/lab3/bert-base-uncased')

# 定义实际的模型
class BERTGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(BERTGRUSentiment, self).__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                          batch_first=True, dropout= 0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else  hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch_size, sent_len]
        with torch.no_grad():
            embedded = self.bert(text)[0]
        # embedded = [batch_size, sent_len, emb_dim]
        _, hidden = self.rnn(embedded)
        # hidden = [n_layers * n_directions, batch_size, emb_dim]
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        # hidden = [batch_size, hidden_dim]
        output = self.out(hidden)
        # output = [batch_size, output_dim]
        return output

# 使用标准超参数创建模型的实例
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

model = BERTGRUSentiment(bert, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

# 检查模型有多少参数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')

# 冻结参数
for name, param in model.named_parameters():
    if name.startswith('bert'):
        param.requires_grad =  False

def count_paraeters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')

# 再次检查可训练参数的名称，确保它们是有意义的
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

# 计算准确率
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

# 执行训练epoch
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# 执行评估epoch
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# 计算训练/评估epoch需要多长时间
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# 训练模型
N_EPOCHS = 5
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut6-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

# 加载最佳验证损失的参数
model.load_state_dict(torch.load('tut6-model.pt'))
test_loss, test_acc = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

# # 推断
# def predict_sentiment(model, tokenizer, sentence):
#     model.eval()
#     tokens = tokenizer.tokenize(sentence)
#     tokens = tokens[:max_input_length-2]
#     indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
#     tensor = torch.LongTensor(indexed).to(device)
#     tensor = tensor.unsqueeze(0)
#     prediction = torch.sigmoid(model(tensor))
#     return prediction.item()

# res = predict_sentiment(model, tokenizer, "This film is terrible")
# print(res)

# res = predict_sentiment(model, tokenizer, "This film is great")
# print(res)
