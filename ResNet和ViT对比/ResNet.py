import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class CifarDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.tranform = transform
        self.data = []
        self.labels = []

        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for file_name in os.listdir(label_dir):
                    file_path = os.path.join(label_dir, file_name)
                    try:
                        image = np.load(file_path)
                        self.data.append(image)
                        self.labels.append(int(label))
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        if self.data.dtype != np.float32:
            self.data = self.data.astype(np.float32)
        if self.labels.dtype != np.int64:
            self.labels = self.labels.astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image = self.data[item]
        label = self.labels[item]
        image = torch.from_numpy(image).permute(2, 0, 1)  # 将numpy数组转化为pytorch张量 permute用来重新排列张量的维度顺序
        image = image.float() / 255.0
        if self.tranform:
            image = self.tranform(image)
        return image, label

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

# 残差网络每个stage内输出的通道数和特征图大小都相同
# 通道数和特征图大小改变只发生在每个stage的第一个block的第一个卷积中
class BasicBlock(nn.Module):
    expansion = 1  # 本代码中可以忽略expansion这个参数，BottleneckBlock才考虑
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)  # 有的时候卷积会改变特征图通道数和大小，所以要有下采样使得输入能够与输出相加
        # 下采样使用 1×1 卷积，它只改变通道，不破坏空间信息，
        # 1×1 卷积的感受野是单个像素，它不会融合邻域信息，所以空间结构（H×W）几乎保持原样，主要在通道维度上做线性组合。
        # 如果不改变特征图尺寸，ResNet 通常直接用 identity shortcut（不做卷积），这样信息完全保留。
        # 1×1 卷积在调整通道时比大卷积核更保留局部像素的原始特征，所以契合残差网络“保留原信息 + 学习残差”的理念。
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # no maxpool
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)  # 每个layer表示每个阶段
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 每个stage由若干个basic block组成
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # nn.AdaptiveAvgPool2d()将输入的二维特征图调整到指定的目标尺寸
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # 阶段（Stage）是指ResNet中一组连续的残差块（Residual Blocks）构成的模块
        # 阶段内的所有残差块通常有相同的特征图通道数，相同的特征图的空间尺寸（宽高不变）
        # 阶段之间，一般会通过步长为2的卷积或池化来降低空间分辨率，同时增加通道数，实现特征的下采样和通道维度提升。

    def _make_layer(self, block, out_channels, blocks, stride):
        # 每个stage的第一个block会做两件事：spatial下采（stride=2）和通道升维（out_channels × expansion）（也可能不做）
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )  # shortcut设计初衷是为了输入保留，并且shortcut的目标只是匹配维度(通道和特征图大小)，不是学习复杂表示，所以用1×1卷积
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        # 如果上面的if没触发那么这里的block和下面for循环中的block一样
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):  # 前面有了个layers.append所以这里从1开始
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)  # *作用是逐个把layers列表中的元素传给nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# 数据增强 + 归一化
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

full_train_dataset = CifarDataset("C:/python/pythonProject/deep_learning/CIFAR10_npy/train", transform=transform_train)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
test_dataset = CifarDataset("C:/python/pythonProject/deep_learning/CIFAR10_npy/test", transform=transform_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

epochs = 100

model = ResNet18()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    if epoch % 1 == 0:
        train_acc = calculate_accuracy(train_loader, model)
        val_acc = calculate_accuracy(val_loader, model)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, train_acc: {train_acc:.2f}, val_acc: {val_acc:.2f}")

test_acc = calculate_accuracy(test_loader, model)
print(f"test_acc: {test_acc:.2f}")


# 残差网络结构（ResNet-34为例）：
"""
Input Image (e.g., 224x224x3)
    ↓
7x7 Conv, stride=2, padding=3 → BN → ReLU
    ↓
3x3 MaxPool, stride=2  （本代码中没有这个MaxPool）
    ↓
Stage 1: Residual Blocks × 3 (channels 64)
    ↓
Stage 2: Residual Blocks × 4 (channels 128, first block downsamples)
    ↓
Stage 3: Residual Blocks × 6 (channels 256, first block downsamples)
    ↓
Stage 4: Residual Blocks × 3 (channels 512, first block downsamples)
    ↓
Global Average Pooling
    ↓
Fully Connected Layer (e.g., 512 → 1000 classes)
"""
