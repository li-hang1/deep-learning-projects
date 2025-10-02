import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

class SmallCNN(nn.Module):
    def __init__(self, conv1_out=16, conv2_out=32):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_out, 3, padding=1)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, 3, padding=1)
        self.fc1 = nn.Linear(conv2_out*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # [B, conv1_out, 28,28]
        x = F.max_pool2d(x, 2)      # [B, conv1_out, 14,14]
        x = F.relu(self.conv2(x))   # [B, conv2_out, 14,14]
        x = F.max_pool2d(x, 2)      # [B, conv2_out, 7,7]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ====== 剪枝：选择保留的通道 ======
def channel_prune(model, keep_ratio=0.5):
    """
    对 conv1 和 conv2 按 L1 范数剪枝，保留前 keep_ratio 比例的通道
    """
    with torch.no_grad():
        conv1_weights = model.conv1.weight.data
        l1_norms = conv1_weights.abs().sum(dim=(1,2,3))  # 每个输出通道的 L1 范数
        num_keep = int(len(l1_norms) * keep_ratio)
        keep_idx1 = torch.topk(l1_norms, num_keep).indices
        # torch.topk用于获取张量中前 k 个最大（或最小）元素及其索引的函数，返回一个元组(values, indices)：
        # values：包含前k个最大（或最小）元素的张量。indices：这些元素在原张量中的索引
        # torch.topk返回的不是普通元组，而是一个具名元组（named tuple），它同时支持两种访问方式：
        # 通过索引访问，如[1]：这是元组的常规访问方式，[0]取元素值，[1]取索引
        # 通过属性名访问，如.indices：因为返回的具名元组定义了values和indices两个属性，分别对应第 0 个和第 1 个元素

        conv2_weights = model.conv2.weight.data
        l1_norms2 = conv2_weights.abs().sum(dim=(1,2,3))
        num_keep2 = int(len(l1_norms2) * keep_ratio)
        keep_idx2 = torch.topk(l1_norms2, num_keep2).indices

    pruned_model = SmallCNN(conv1_out=num_keep, conv2_out=num_keep2)

    # ====== 复制被保留的权重 ======
    with torch.no_grad():
        # Conv1
        pruned_model.conv1.weight.copy_(model.conv1.weight[keep_idx1])
        pruned_model.conv1.bias.copy_(model.conv1.bias[keep_idx1])
        # Conv2 (注意输入通道也要同步裁剪)
        pruned_model.conv2.weight.copy_(model.conv2.weight[keep_idx2][:, keep_idx1])
        pruned_model.conv2.bias.copy_(model.conv2.bias[keep_idx2])
        # FC1
        # 原来是 [conv2_out*7*7, 128]，现在需要重新对齐输入维度
        old_fc1_w = model.fc1.weight.data[:, :]  # 加.data作用是变成纯张量，没有梯度
        idx_map = []
        for c in keep_idx2:
            start = c * 7 * 7
            end = (c + 1) * 7 * 7
            idx_map.extend(range(start, end))
        idx_map = torch.tensor(idx_map)
        pruned_model.fc1.weight.copy_(old_fc1_w[:, idx_map])
        pruned_model.fc1.bias.copy_(model.fc1.bias.data)
        # FC2
        pruned_model.fc2.weight.copy_(model.fc2.weight.data)
        pruned_model.fc2.bias.copy_(model.fc2.bias.data)

    return pruned_model

class MNIST(Dataset):
    def __init__(self, path):
        self.data = []
        self.label = []
        for label in os.listdir(path):
            label_dir = os.path.join(path, label)
            for img in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img)
                self.data.append(np.load(img_path))
                self.label.append(int(label))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.label[idx]
        image = torch.from_numpy(image).float()
        return image, torch.tensor(label)

train_dataset = MNIST("C:/python/pythonProject/deep_learning/mnist_numpy/train")
test_dataset = MNIST("C:/python/pythonProject/deep_learning/mnist_numpy/test")
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

def train(model, device, train_loader, optimizer):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(data), target)
        loss.backward()
        optimizer.step()

def calculate_accuracy(model, loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for image, label in loader:
            outputs = model(image)
            _, predicted = torch.max(outputs.data, dim=1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    model.train()
    return correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 训练原始模型
    model = SmallCNN().to(device)
    optimizer = optim.Adam(model.parameters())
    for epoch in range(2):  # 少训几轮演示
        train(model, device, train_loader, optimizer)
    acc_before = calculate_accuracy(model, test_loader)
    print("Before pruning test accuracy:", acc_before)
    # 剪枝
    pruned_model = channel_prune(model, keep_ratio=0.5).to(device)
    acc_after = calculate_accuracy(pruned_model, test_loader)
    print("After structured pruning test accuracy (no fine-tune):", acc_after)
    # 微调
    optimizer = optim.Adam(pruned_model.parameters())
    for epoch in range(1):
        train(pruned_model, device, train_loader, optimizer)
    acc_ft = calculate_accuracy(pruned_model, test_loader)
    print("After structured pruning + fine-tune test accuracy:", acc_ft)

if __name__ == "__main__":
    main()




