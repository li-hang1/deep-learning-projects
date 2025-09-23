import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import matplotlib.pyplot as plt

class Image(Dataset):
    def __init__(self, path):
        self.data = [torch.load(os.path.join(path, f)) for f in os.listdir(path)]
        self.data = torch.cat(self.data, dim=0)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        image = self.data[idx]
        return image

path = "/root/LH/deeplearning/Face_pt/"
# 设置超参数
noise_size = 100      # 输入噪声维度
ngf = 64      # 生成器特征通道数
ndf = 64      # 判别器特征通道数
image_channel = 3        # 图像通道数 (彩色图像：3，灰度图像：1)
batch_size = 128
image_size = 32
lr = 0.0002
beta1 = 0.5   # Adam 中的 β1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = Image(path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义生成器 G
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 输入 Z: (batch_size, nz, 1, 1) nn.ConvTranspose2d输入必须是四维张量[batch_size, in_channels, height, width]
            # nn.ConvTranspose2d输出也是四维张量[batch_size, out_channels, height, width]
            nn.ConvTranspose2d(noise_size, ngf*4, 4, 1, 0, bias=False),  # 前两个参数为in_channels,out_channels,
            # output_padding的作用是当卷积后生成的图像不是想要的尺寸时可以改变成想要的尺寸
            # padding=0时，output_padding多出来的像素值一定为0，padding不为0时，output_padding把原来的值填回去
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(),
            # 状态：ngf*8 x 4 x 4
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(),
            # 状态：ngf*4 x 8 x 8
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            # 状态：ngf*2 x 16 x 16
            nn.ConvTranspose2d(ngf, image_channel, 4, 2, 1, bias=False),
            nn.Tanh()  # 输出范围 [-1, 1]
            # 状态：ngf x 32 x 32
        )

    def forward(self, x):
        return self.main(x)

# 定义判别器 D
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(image_channel, ndf, 4, 2, 1, bias=False),  # 输入：nc x 32 x 32
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),  # 状态：ndf x 16 x 16
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),  # 状态：ndf*2 x 8 x 8
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf*4, 1, 4, 1, 0, bias=False),  # 状态：ndf*4 x 4 x 4
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(x.size(0), -1)

# 初始化模型
netG = Generator().to(device)
netD = Discriminator().to(device)

# 权重初始化
def weights_init(m):
    classname = m.__class__.__name__  # m.__class__获取对象 m 的类。返回类型对象
    # m.__class__.__name__的作用是获取对象 m 所属类的类名，返回一个字符串
    if classname.find('Conv') != -1:  # Python中字符串的.find()方法用于查找子字符串在原字符串中第一次出现的位置，如果没找到，返回-1。
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        # torch.nn.init.normal_(tensor, mean=0.0, std=1.0)用于将张量（tensor）原地初始化为正态分布（高斯分布）随机值
        # .weight.data 是一种访问和操作神经网络层中权重张量的底层数据的方式
        # .weight 是 nn.Module（如 nn.Linear, nn.Conv2d 等）中的一个 nn.Parameter 对象，是这个模块需要学习的参数。
        # .weight.data 是一个纯张量（torch.Tensor），不再是 Parameter 类型
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init)
netD.apply(weights_init)
# .apply(fn) 会对调用它的模块本身以及其所有子模块（包括子模块的子模块，递归遍历）依次调用 fn(module)。
# 这个 fn 是你自己定义的函数，接受一个模块（nn.Module 对象）作为参数。
# 典型用途是对模型中所有层统一执行某些操作，比如初始化权重、打印结构信息、修改参数等。

# 损失函数与优化器
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# 训练
num_epochs = 10
for epoch in range(num_epochs):
    for real_images in dataloader:
        netD.zero_grad()
        real_images = real_images.to(device)
        b_size = real_images.size(0)
        real_labels = torch.ones(b_size, 1).to(device)
        fake_labels = torch.zeros(b_size, 1).to(device)
        output = netD(real_images)
        errD_real = criterion(output, real_labels)
        noise = torch.randn(b_size, noise_size, 1, 1, device=device)
        fake = netG(noise)
        output = netD(fake.detach())
        errD_fake = criterion(output, fake_labels)
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()
        netG.zero_grad()
        output = netD(fake)
        errG = criterion(output, real_labels)
        errG.backward()
        optimizerG.step()

    print(f"epoch:{epoch} Loss_D:{errD.item():.2f} Loss_G:{errG.item():.2f}")

    if (epoch + 1) % 2 == 0:
        netG.eval()
        with torch.no_grad():
            fixed_noise = torch.randn(9, noise_size, 1, 1, device=device)
            fake_imgs = netG(fixed_noise)
            grid = fake_imgs.cpu().numpy()
            plt.figure(figsize=(9, 9))
            for i in range(9):
                plt.subplot(3, 3, i + 1)  # subplot() 后必须跟某种绘图函数，否则子图为空
                plt.imshow(grid[i].transpose(1, 2, 0)* 0.5 + 0.5)  # [-1,1]→[0,1]
                plt.axis('off')
            plt.show()
        netG.train()
