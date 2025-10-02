import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

class MNIST(Dataset):
    def __init__(self, path):
        self.data = []
        self.label = []
        for label in os.listdir(path):
            label_path = os.path.join(path, label)
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                self.data.append(torch.from_numpy(np.load(image_path)))
                self.label.append(torch.tensor(int(label)))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

def calculate_accuracy(model, loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data, label in loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    model.train()
    return correct / total

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ------------------ 低秩分解替换层 ------------------
class LowRankLinear(nn.Module):
    def __init__(self, linear_layer, rank):
        super().__init__()
        W = linear_layer.weight.data
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # 取前 rank 个奇异值
        U_r = U[:, :rank]
        S_r = torch.diag(S[:rank])
        V_r = Vh[:rank, :]

        # 分解为两个小的 Linear 层
        self.fc1 = nn.Linear(V_r.size(1), rank, bias=False)
        self.fc2 = nn.Linear(rank, U_r.size(0), bias=True)

        # 初始化权重
        self.fc1.weight.data = V_r
        self.fc2.weight.data = (U_r @ S_r)
        self.fc2.bias.data = linear_layer.bias.data.clone()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

train_dataset = MNIST("C:/python/pythonProject/deep_learning/mnist_numpy/train")
test_dataset = MNIST("C:/python/pythonProject/deep_learning/mnist_numpy/test")
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

def train(model, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss = {total_loss/len(train_loader):.4f}")

model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train(model, optimizer, epochs=3)
print("Before decomposition test_set accuracy:", calculate_accuracy(model, test_loader))

# 用 SVD 分解 fc1
rank = 8  # 保留前 32 个奇异值
model.fc1 = LowRankLinear(model.fc1, rank)
print("After decomposition test_set accuracy:", calculate_accuracy(model, test_loader))

# 微调 (fine-tuning)
optimizer = optim.Adam(model.parameters(), lr=0.001)
train(model, optimizer, epochs=1)
print("After decomposition and fine-tuning test_set accuracy:", calculate_accuracy(model, test_loader))

