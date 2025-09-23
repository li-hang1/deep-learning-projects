import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os

class MnistDataset(Dataset):
    def __init__(self, root_dir, flatten):
        super().__init__()
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.flatten = flatten

        for label in os.listdir(self.root_dir):
            label_dir = os.path.join(self.root_dir, label)
            for file_name in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file_name)
                self.data.append(np.load(file_path))
                self.labels.append(int(label))

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        if self.labels.dtype != np.int64:
            self.labels = self.labels.astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image = self.data[item]
        label = self.labels[item]
        image = torch.from_numpy(image)
        if self.flatten:
            image = image.flatten()
        return image, label

class MLP_NoBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.out = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h1 = torch.sigmoid(self.fc1(x))  # 对每个分量求sigmoid函数值
        h2 = torch.sigmoid(self.fc2(h1))
        h3_input = self.fc3(h2)
        h3 = torch.sigmoid(h3_input)
        logits = self.out(h3)  # 交叉熵自带softmax
        return logits, h3_input


class MLP_BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.out = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h1 = torch.sigmoid(self.bn1(self.fc1(x)))
        h2 = torch.sigmoid(self.bn2(self.fc2(h1)))
        h3_input = self.bn3(self.fc3(h2))
        h3 = torch.sigmoid(h3_input)
        logits = self.out(h3)
        return logits, h3_input

trainset = MnistDataset("C:/python/pythonProject/deep_learning/mnist_numpy/train", flatten=True)
testset = MnistDataset("C:/python/pythonProject/deep_learning/mnist_numpy/test", flatten=True)
train_loader = DataLoader(trainset, batch_size=60, shuffle=True)
test_loader  = DataLoader(testset, batch_size=1000, shuffle=False)

# 初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_no_bn = MLP_NoBN().to(device)
model_bn    = MLP_BN().to(device)

optimizer_no_bn = torch.optim.SGD(model_no_bn.parameters(), lr=0.01)
optimizer_bn    = torch.optim.SGD(model_bn.parameters(), lr=0.01)

criterion = nn.CrossEntropyLoss()

# 存储信息
record_steps = [i + 1 for i in range(50000) if i % 100 == 0]
test_acc_no_bn, test_acc_bn = [], []
activations_no_bn, activations_bn = [], []

step = 0
while step < 50000:
    for (x, y) in train_loader:
        x, y = x.to(device), y.to(device)

        for model, optimizer, acc_list, act_list in [
            (model_no_bn, optimizer_no_bn, test_acc_no_bn, activations_no_bn),
            (model_bn, optimizer_bn, test_acc_bn, activations_bn)
        ]:
            model.train()
            optimizer.zero_grad()
            logits, h3_input = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            if step + 1 in record_steps:
                # 测试精度
                model.eval()
                correct = total = 0
                with torch.no_grad():
                    for x_test, y_test in test_loader:
                        x_test, y_test = x_test.to(device), y_test.to(device)
                        logits, _ = model(x_test)
                        preds = logits.argmax(dim=1)
                        correct += (preds == y_test).sum().item()
                        total += y_test.size(0)
                acc_list.append(correct / total)

                # 记录最后一层隐藏层一个神经元的 sigmoid 输入值
                h3_input_sample = h3_input[:, 0].detach().cpu().numpy()
                act_list.append(h3_input_sample)
        step += 1
        if step >= 50000:
            break

# Figure 1(a)
plt.figure(figsize=(6, 4))
plt.plot(record_steps, test_acc_no_bn, label='Without BN')
plt.plot(record_steps, test_acc_bn, label='With BN')
plt.xlabel('Training Steps')
plt.ylabel('Test Accuracy')
plt.legend()  # plt.legend()用来在图表上添加图例
plt.title('Test Accuracy vs Training Steps')
plt.grid()  # plt.grid()用来在图表中添加网格线
plt.xticks([10000, 20000, 30000, 40000, 50000])
plt.show()

# Figure 1(b) and 1(c)
def plot_percentiles(activations, title):
    percentiles = [15, 50, 85]
    values = np.array([[np.percentile(act, p) for act in activations] for p in percentiles])
    # np.percentile()是一个用于计算数组中特定百分位数的函数。第一个参数是输入的数组，第二个参数是要计算的百分位数，输出是数组中指定百分位数位置的值
    for pval, label in zip(values, ['15th', '50th', '85th']):
        plt.plot(record_steps, pval, label=label)
    plt.title(title)
    plt.xlabel('Training Steps')
    plt.ylabel('Percentile Value')
    plt.legend()
    plt.grid()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plot_percentiles(activations_no_bn, "Without BN")
plt.subplot(1, 2, 2)
plot_percentiles(activations_bn, "With BN")
plt.xticks([10000, 20000, 30000, 40000, 50000])
plt.show()
