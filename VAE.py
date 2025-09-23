import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

class ImageDataset(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.X_list = [np.load(os.path.join(path, f)) for f in os.listdir(path) if f.endswith('.npy')]
    def __len__(self):
        return len(self.X_list)
    def __getitem__(self, item):
        image = self.X_list[item]
        image = torch.from_numpy(image)
        if self.transform:
            image = self.transform(image)
        return image

# --- 定义 VAE 模型 ---
class VAE(nn.Module):
    def __init__(self, latent_dim=20):  # latent_dim就是z的维数
        super().__init__()
        self.latent_dim = latent_dim
        # 编码器网络 q_phi(z|x)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 400),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(400, latent_dim)      # 均值
        self.fc_logvar = nn.Linear(400, latent_dim)  # log方差，直接学习方差的对数
        # 解码器网络 p_theta(x|z)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 28*28),
            nn.Sigmoid(),  # 输出像素概率[0,1]，生成非二值化的图像需要换激活函数和损失函数
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)     # 标准差，乘以0.5所以是标准差
        eps = torch.randn_like(std)       # 标准正态噪声
        return mu + eps * std             # 重参数化采样

    def decode(self, z):
        x_recon = self.decoder(z)
        x_recon = x_recon.view(-1, 1, 28, 28)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# --- 损失函数：重构损失 + KL散度 ---
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')   # 重构误差，二元交叉熵
    # KL散度的解析表达式
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # .pow()是tensor的的一个方法，作用是作用是对张量中的每个元素执行次方操作
    return BCE + KLD

# --- 训练 ---
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(data)
        loss = loss_function(x_recon, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    if epoch % 1 == 0:
        print(f'Train Epoch:{epoch} Loss:{loss.item():.2f}')

# --- 测试：重构 ---
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon, mu, logvar = model(data)
            test_loss += loss_function(recon, data, mu, logvar).item()
    print(f'Test set loss: {test_loss:.2f}')

path_train = "C:/python/pythonProject/deep_learning/mnist_numpy/train/8"
path_test = "C:/python/pythonProject/deep_learning/mnist_numpy/test/8"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = ImageDataset(path_train)
test_dataset = ImageDataset(path_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 初始化模型和优化器
model = VAE(latent_dim=20).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 100
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# --- 生成新图像 ---
def generate_samples(model, device, num_samples=9):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)  # 从 N(0,I) 采样 z
        image_tensor = model.decode(z).view(-1, 1, 28, 28)  # 用 decoder 生成图像
        image = image_tensor.cpu().numpy()
        plt.figure(figsize=(5,5))
        for i in range(9):
            plt.subplot(3, 3, i+1)
            plt.imshow(image[i, 0, :, :], cmap='gray')
            plt.axis('off')
        plt.show()

generate_samples(model, device)