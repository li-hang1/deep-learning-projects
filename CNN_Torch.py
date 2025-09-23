import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MnistDataset(Dataset):
    def __init__(self, root_dir, transform=None, flatten=True):
        self.root_dir = root_dir
        self.transform = transform
        self.flatten = flatten
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
        if self.transform:
            image = self.transform(image)
        if self.flatten:
            image = image.flatten()
        return image, label

def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():  # with作用是临时禁用梯度计算，作用范围仅限于缩进块内的代码
        for x, y in loader:
            outputs = model(x)
            _, predicted = torch.max(outputs.data, dim=1)  # 前面返回最大值，后面返回索引
            # .data是张量的一个属性，用于获取该张量的 “数据部分”，输入一个张量，返回一个新的张量，其数值与outputs完全相同。但没有梯度信息。
            total += y.size(0)  # 返回张量的第()维的大小
            correct += (predicted == y).sum().item()
    model.train()
    return correct / total

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, stride=1),  # [B, 32, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [B, 32, 16, 16]
            nn.Conv2d(32, 64, 3, padding=1, stride=1),  # [B, 64, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [B, 64, 8, 8]
            nn.Conv2d(64, 128, 3, padding=1, stride=1),  # [B, 128, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [B, 128, 4, 4]
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.layers(x)

model_path = "C:/python/pythonProject/deep_learning/torch_mlp.pth"

transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 元组形式的mean和std是为了支持通道级的独立归一化，元组中第一个数第一个通道，第二个数第二个通道，以此类推。
full_train_dataset = MnistDataset("C:/python/pythonProject/deep_learning/CIFAR10_npy/train", transform=transform, flatten=False)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
test_dataset = MnistDataset("C:/python/pythonProject/deep_learning/CIFAR10_npy/test", transform=transform, flatten=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

epochs = 20

model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# if os.path.exists(model_path):
#     model.load_state_dict(torch.load(model_path))  # torch.load输出一个字典，键是模型每层参数的名称，值是对应的tensor
#     print("Loaded saved model parameters.")

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

#torch.save(model.state_dict(), model_path)
#print(f"Model saved to {model_path}")

test_acc = calculate_accuracy(test_loader, model)
print(f"test_acc: {test_acc:.2f}")