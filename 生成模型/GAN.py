import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os

path = "C:/python/pythonProject/deep_learning/mnist_numpy/train/8"
a = np.load(os.path.join(path, "90.npy"))  # 形状（1，28，28），数值0到1
plt.figure(figsize=(3, 3))
plt.subplot(1, 1, 1)  # subplot用来把画布划分为多个子图
plt.imshow(a[0, :, :], cmap='gray')  # imshow()显示的图像格式要求：灰度图像：(H, W)，彩色图像（RGB）：(H, W, 3)
# imshow() 只是绘图命令，不是显示命令。
# plt.imshow() 的作用是：把图像数据加载并渲染到“当前画布（Figure）”的某个子图上。但它不会触发图形界面的更新或弹出窗口，只是准备好了图像。
plt.show()
# plt.show()才是触发显示的命令。它的作用是：告诉后台图形系统：现在开始显示图像窗口！它会阻塞程序，等待你关闭图像窗口或交互式操作结束。


class Image(Dataset):
    def __init__(self, path, transforms=None):
        self.transforms = transforms
        self.X_list = [np.load(os.path.join(path, f)) for f in os.listdir(path) if f.endswith('.npy')]

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        image = self.X_list[idx]
        image = torch.from_numpy(image)
        if self.transforms:
            image = self.transforms(image)
        return image

# 1. 超参数
latent_dim = 100
batch_size = 128
lr = 0.0002
epochs = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载数据
transform = transforms.Compose([transforms.Normalize([0.5], [0.5])])  # 将像素从[0,1]变成[-1,1]
data = Image(path, transform)
loader = DataLoader(data, batch_size=batch_size, shuffle=True)

# 3. 判别器 D：输入图像，输出真假概率
D = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid()
).to(device)

# 4. 生成器 G：输入随机噪声，输出图像
G = nn.Sequential(
    nn.Linear(latent_dim, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 28*28),
    nn.Tanh()  # 输出范围是 [-1, 1]，与 Normalize 配合
    # nn.Tanh()是对输入张量中每个元素逐个计算双曲正切函数tanh(x)的值。
).to(device)

# 5. 损失函数和优化器
loss_fn = nn.BCELoss()  # BCELoss()就是二元交叉熵
optimizer_D = optim.Adam(D.parameters(), lr=lr)
optimizer_G = optim.Adam(G.parameters(), lr=lr)

# 6. 训练
for epoch in range(epochs):
    for i, real_imgs in enumerate(loader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)
        # 生成真实标签和伪造标签
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        # 训练判别器 D
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = G(z).view(-1, 1, 28, 28)  # .view()方法自动调整张量维度，哪个维度是-1哪个维度自适应

        real_output = D(real_imgs)
        fake_output = D(fake_imgs.detach())

        loss_D = loss_fn(real_output, real_labels) + loss_fn(fake_output, fake_labels)
        # real_labels和fake_labels的作用就是决定loss中log(p)还是log(1 - p)被激活，1对应log(p)，0对应log(1 - p)

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
        # 训练生成器 G
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = G(z).view(-1, 1, 28, 28)
        fake_output = D(fake_imgs)

        loss_G = loss_fn(fake_output, real_labels)  # 让假图被判为真

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}]  Loss_D: {loss_D.item():.2f}  Loss_G: {loss_G.item():.2f}")

    # 每10轮显示图像
    if (epoch+1) % 100 == 0:
        G.eval()
        with torch.no_grad():
            z = torch.randn(9, latent_dim).to(device)
            fake_imgs = G(z).view(-1, 1, 28, 28)
            grid = fake_imgs.cpu().numpy()  # tensor不能作为plt.imshow()的输入，所以要转成numpy数组
            plt.figure(figsize=(9,9))
            for i in range(9):
                plt.subplot(3, 3, i+1)
                plt.imshow(grid[i, 0, :, :]*0.5 + 0.5, cmap='gray')  # [-1,1]→[0,1]
                plt.axis('off')
            plt.show()
        G.train()
