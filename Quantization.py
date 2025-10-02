import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MNIST(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = []
        self.labels = []
        for label in os.listdir(path):
            label_dir = os.path.join(path, label)
            for img in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img)
                self.data.append(torch.from_numpy(np.load(img_path)))
                self.labels.append(torch.tensor(int(label)))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def fake_quantize(tensor, num_bits=8):
    """模拟量化（Fake Quantization）"""
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    max_val = tensor.abs().max()
    scale = max_val / qmax if max_val != 0 else 1.0
    # 量化 + 反量化
    q_tensor = torch.clamp((tensor / scale).round(), qmin, qmax)
    # .round()方法的作用是对张量中的每个元素进行四舍五入取整操作，返回一个新的张量，其中每个元素都是原张量对应元素经过四舍五入后的整数结果。
    return q_tensor * scale, scale  # 返回模拟量化后的 tensor 和 scale

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

train_dataset = MNIST("C:/python/pythonProject/deep_learning/mnist_numpy/train")
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.008)

for epoch in range(10):
    model.train()
    total_loss = 0
    calculate = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        with torch.no_grad():  # 前向：模拟量化权重
            for name, param in model.named_parameters():
                # model.named_parameters()每次迭代会返回一个元组 (name, parameter)，其中：
                # name 是参数的名称（字符串），通常反映参数在模型中的位置（如 conv1.weight、fc2.bias 等）。
                # parameter 是参数对应的张量（torch.Tensor），包含实际的权重或偏置值。
                if "bias" not in name:
                    q_param, scale = fake_quantize(param)
                    param.copy_(q_param)  # 假量化
        loss = F.cross_entropy(model(data), target)
        total_loss += loss
        loss.backward()  # 反向（梯度仍是 FP32）
        optimizer.step()
    total_loss = total_loss / len(train_loader)
    calculate += calculate_accuracy(model, train_loader)
    print(f"Epoch {epoch+1}: loss={total_loss.item():.4f} train calculate={calculate:.4f}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def model_size_mb(model, quantized=False):
    total_params = count_parameters(model)
    if not quantized:
        return total_params * 4 / (1024 ** 2)
    else:
        int8_params = 0  # 权重参数总字节
        fp32_params = 0  # 偏置参数总字节
        scale_params = 0  # 每个 weight tensor 一个 scale
        for name, param in model.named_parameters():
            if "bias" in name:  # 偏置保持 FP32
                fp32_params += param.numel() * 4
            else:
                int8_params += param.numel() * 1  # 权重量化为 INT8
                scale_params += 4  # 每个 weight tensor 需要存一个 FP32 scale
    return (int8_params + fp32_params + scale_params) / (1024 ** 2)

print("模型参数总数:", count_parameters(model))
print(f"未量化模型大小: {model_size_mb(model, quantized=False):.2f} MB")
print(f"量化模型大小: {model_size_mb(model, quantized=True):.2f} MB")

# 推理阶段真正量化
def quantize_tensor(tensor, num_bits=8):
    """真正量化存储为整数"""
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    max_val = tensor.abs().max()
    scale = max_val / qmax if max_val != 0 else 1.0
    q_tensor = torch.clamp((tensor / scale).round(), qmin, qmax).to(torch.int8)
    return q_tensor, scale

# 量化推理函数
def int8_inference(model, test_loader):
    model.eval()
    # 先把所有权重量化
    q_params = {}
    scales = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "bias" not in name:
                q_param, scale = quantize_tensor(param)
                q_params[name] = q_param
                scales[name] = scale
    total = 0
    correct = 0
    # 推理阶段
    with torch.no_grad():
        for data, label in test_loader:
            # conv1
            w_q = q_params["conv1.weight"].float() * scales["conv1.weight"]
            b = model.conv1.bias
            x = F.conv2d(data, w_q, b, stride=1, padding=1)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            # conv2
            w_q = q_params["conv2.weight"].float() * scales["conv2.weight"]
            b = model.conv2.bias
            x = F.conv2d(x, w_q, b, stride=1, padding=1)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            # flatten
            x = x.view(x.size(0), -1)
            # fc1
            w_q = q_params["fc1.weight"].float() * scales["fc1.weight"]
            b = model.fc1.bias
            x = F.linear(x, w_q, b)
            x = F.relu(x)
            # fc2
            w_q = q_params["fc2.weight"].float() * scales["fc2.weight"]
            b = model.fc2.bias
            x = F.linear(x, w_q, b)
            _, predicted = torch.max(x.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    return correct / total

test_dataset = MNIST("C:/python/pythonProject/deep_learning/mnist_numpy/test")
test_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 量化推理精度
acc_int8 = int8_inference(model, test_loader)
print(f"量化推理精度: {acc_int8:.4f}")



