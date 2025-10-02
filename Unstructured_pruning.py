import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2),  # [B, 32, 13, 13]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [B, 32, 6, 6]
            nn.Flatten(),
            nn.Linear(1152, 10),
        )
    def forward(self, x):
        return self.layers(x)

def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            _, predicted = torch.max(outputs.data, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    model.train()
    return correct / total

class MnistDataset(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        for label in os.listdir(path):
            label_dir = os.path.join(path, label)
            for file_name in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file_name)
                image = np.load(file_path)
                self.data.append(image)
                self.labels.append(int(label))
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        if self.data.dtype != np.float32:
            self.data = self.data.astype(np.float32)
        if self.labels.dtype != np.int64:
            self.labels = self.labels.astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        image = torch.from_numpy(image)
        if self.transform:
            image = self.transform(image)
        return image, label

def train(model, device, train_loader, optimizer):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

# 非结构化剪枝函数
def prune_by_threshold(model, threshold=1):
    with torch.no_grad():
        for name, param in model.named_parameters():
            # model.named_parameters()作用是返回模型中所有可学习参数（parameters）的名称和参数本身的迭代器。
            # 同时获取每个参数的名称（字符串）和对应的参数张量（torch.Tensor 对象）
            if "weight" in name:
                mask = param.abs() >= threshold
                param *= mask  # 小于阈值的权重直接置零
    print(f"Pruned with threshold={threshold}")

def main():
    # 配置
    batch_size = 64
    epochs = 3   # 少训几轮演示即可
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 数据
    train_dataset = MnistDataset("C:/python/pythonProject/deep_learning/mnist_numpy/train")
    test_dataset = MnistDataset("C:/python/pythonProject/deep_learning/mnist_numpy/test")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # 模型
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 训练
    for epoch in range(1, epochs+1):
        train(model, device, train_loader, optimizer)
        acc = calculate_accuracy(train_loader, model)
        print(f"Epoch {epoch}, Accuracy: {acc:.2f}")
    # 剪枝前精度
    acc_before = calculate_accuracy(test_loader, model)
    print("Before pruning test accuracy:", acc_before)
    # 剪枝
    prune_by_threshold(model, threshold=0.1)
    # 剪枝后精度（不微调）
    acc_after = calculate_accuracy(test_loader, model)
    print("After pruning test accuracy (no fine-tune):", acc_after)
    # 剪枝后 + 微调
    for epoch in range(1, 2):  # 微调1轮
        train(model, device, train_loader, optimizer)
    acc_ft = calculate_accuracy(test_loader, model)
    print("After pruning + fine-tune test accuracy:", acc_ft)

if __name__ == "__main__":
    main()





