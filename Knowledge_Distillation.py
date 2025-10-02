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
        self.labels = []
        for label in os.listdir(path):
            label_dir = os.path.join(path, label)
            for image in os.listdir(label_dir):
                self.data.append(torch.from_numpy(np.load(os.path.join(label_dir, image))))
                self.labels.append(int(label))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        return image, torch.tensor(label)

class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (12, 12))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, 1)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.adaptive_avg_pool2d(x, (8, 8))
        x = torch.flatten(x, 1)
        return self.fc1(x)

# ------------------ 蒸馏训练函数 ------------------
def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.8):
    """
    student_logits: 学生模型输出 (B, num_classes)
    teacher_logits: 教师模型输出 (B, num_classes)
    labels: 真实标签
    T: 温度 (soft label 平滑)
    alpha: 硬标签与软标签的权重平衡
    """
    hard_loss = F.cross_entropy(student_logits, labels)

    # Soft label 部分
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)

    return alpha * hard_loss + (1 - alpha) * soft_loss

def train_kd(student, teacher, optimizer, train_loader, epochs=3):
    teacher.eval()  # teacher 固定参数
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            s_logits = student(data)
            with torch.no_grad():
                t_logits = teacher(data)
            loss = distillation_loss(s_logits, t_logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss = {total_loss/len(train_loader):.4f}")

def calculate_accuracy(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    model.eval()
    return correct / total

teacher = TeacherNet()
student = StudentNet()
train_dataset = MNIST("C:/python/pythonProject/deep_learning/mnist_numpy/train")
test_dataset = MNIST("C:/python/pythonProject/deep_learning/mnist_numpy/test")
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

teacher_opt = optim.Adam(teacher.parameters())
for epoch in range(3):
    teacher.train()
    for data, target in train_loader:
        teacher_opt.zero_grad()
        output = teacher(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        teacher_opt.step()
print("Teacher test accuracy:", calculate_accuracy(teacher, test_loader))

# 蒸馏训练 Student
student_opt = optim.Adam(student.parameters())
train_kd(student, teacher, student_opt, train_loader, epochs=3)
print("Student with KD test accuracy:", calculate_accuracy(student, test_loader))

# optimizer = optim.Adam(student.parameters())
# for epoch in range(3):
#     student.train()
#     for data, target in train_loader:
#         optimizer.zero_grad()
#         output = student(data)
#         loss = F.cross_entropy(output, target)
#         loss.backward()
#         optimizer.step()
# print("Student test accuracy:", calculate_accuracy(student, test_loader))