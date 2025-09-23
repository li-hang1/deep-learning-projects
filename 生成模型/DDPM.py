import os, math
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import matplotlib.pyplot as plt

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)
    # torch.linspace(start, end, steps)会生成一个包含steps个元素的张量，这些元素从start开始，到end结束，且相邻元素之间的差值相等。（包含开头结尾）

class DiffusionConstants:  # 把β_t，α_t，α_t连乘，α_{t-1}, β_t弯等准备好
    def __init__(self, timesteps=1000, device='cpu'):
        self.timesteps = timesteps
        betas = linear_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # 把α连乘准备好
        # torch.cumprod计算张量在指定维度上的累积连乘积（cumulative product），输入和输出维度一致
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])  # t-1时的α连乘
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)  # 就是βt弯

        # register to device
        self.device = device
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = posterior_variance.to(device)

# time embedding
def sinusoidal_pos_emb(timesteps, dim):  # timesteps是一个batch的时间步索引，例如[10, 57, 200]。
    # timesteps: (batch,) long
    device = timesteps.device
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=device) * -emb)  # torch.arange(num)生成0到num-1一维张量
    emb = timesteps[:, None].float() * emb[None, :]  # timesteps[:, None]把timesteps变成列向量，每行的数都与emb[None, :]相乘
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:  # 如果是奇数的话，右边填充一列0
        emb = F.pad(emb, (0,1,0,0))
    return emb  # F.pad用于在张量的各个维度两端填充数值，按维度顺序逆向指定，默认零填充，后面元组中前两个数左右，后两个数上下
# emb.shape = (batch_size, dim)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch) if out_ch >= 8 else nn.BatchNorm2d(out_ch)
        # nn.GroupNorm将输入张量的通道分成若干组，在每组内部计算均值和方差并进行归一化，其不依赖批量大小，适用于小批量场景
        # 当分组数为1时等价于LayerNorm，当分组数等于通道数时等价于InstanceNorm
        self.norm2 = nn.GroupNorm(8, out_ch) if out_ch >= 8 else nn.BatchNorm2d(out_ch)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch) if time_emb_dim is not None else None
        # self.time_mlp作用是把时间步编码的维度(time_emb_dim)映射到通道维度(out_ch)，从而可以和卷积特征相加。
        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        # 1×1卷积用于改变通道数，以备后面用来残差连接
    def forward(self, x, t_emb):  # (卷积+归一+激活) + 时间步编码 + (卷积+归一+激活) + 残差连接
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.relu(h)
        if self.time_mlp is not None:
            time_out = self.time_mlp(t_emb)[:, :, None, None]  # time_out形状；[batch,C,1,1]
            h = h + time_out  # 对于h中的每个batch中的每个通道中整个H×W特征图加同一个值
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.relu(h)
        return h + self.res_conv(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, base_ch=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim*2),
            nn.ReLU(),
            nn.Linear(time_emb_dim*2, time_emb_dim)
        )
        # encoder
        self.enc1 = ResidualBlock(in_channels, base_ch, time_emb_dim)
        self.enc2 = ResidualBlock(base_ch, base_ch*2, time_emb_dim)
        self.enc3 = ResidualBlock(base_ch*2, base_ch*4, time_emb_dim)
        # bottleneck
        self.bottom = ResidualBlock(base_ch*4, base_ch*4, time_emb_dim)
        # decoder
        self.dec3 = ResidualBlock(base_ch*8, base_ch*2, time_emb_dim)
        self.dec2 = ResidualBlock(base_ch*4, base_ch, time_emb_dim)
        self.dec1 = ResidualBlock(base_ch*2, base_ch, time_emb_dim)
        # final
        self.final = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch, in_channels, kernel_size=1)
        )
        self.down = nn.AvgPool2d(2)  # kernel_size=2，stride默认和kernel_size相同
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        # nn.Upsample 是用于对输入张量进行放大的模块，可通过指定的缩放因子或目标大小来调整张量的空间维度
        # scale_factor=2表示所有空间维度（如高度、宽度）按相同比例放大（例如将 3×3 的特征图放大为 6×6）。
        # mode='nearest'表示最邻近插值
        # nn.Upsample只对特征图的空间维度（高度和宽度）进行放大操作，不会改变批量大小和通道数。

    def forward(self, x, t):  # x是加噪后的图像，形状是(B,C,H,W)，t是每张图对应的时间步，形状是(B,)，元素是int
        t_emb = sinusoidal_pos_emb(t, self.time_mlp[0].in_features).to(x.device)
        # self.time_mlp[0]表示nn.Sequential中第一个，.in_features查询Linear层的输入维度
        t_emb = self.time_mlp(t_emb)  # t_emb是时间步编码经一个MLP后的输出

        e1 = self.enc1(x, t_emb)           # -> base_ch
        e2 = self.enc2(self.down(e1), t_emb)  # -> base_ch*2
        e3 = self.enc3(self.down(e2), t_emb)  # -> base_ch*4

        b = self.bottom(self.down(e3), t_emb)    # U形最底部
        # self.bot不改变通道数，e3与d3通道数相同，e2与d2通道数相同，e1与d1通道数相同

        d3 = self.up(b)
        d3 = torch.cat([d3, e3], dim=1)  # 在通道维度拼接
        # 跳跃连接，结合全局+局部信息
        d3 = self.dec3(d3, t_emb)

        d2 = self.up(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2, t_emb)

        d1 = self.up(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1, t_emb)

        out = self.final(d1)
        # 最后一层卷积的作用是将解码器得到的特征图通道数压缩到目标输出通道数（如 3 通道图像）
        # 同时在不改变空间分辨率的前提下进一步融合特征，保证生成结果既维度正确又更自然。
        return out

def q_sample(x0, t, noise, constants: DiffusionConstants):  # 获取第任意步正向传播
    # 函数名和def q_sample(x0, t, noise, constants): 效果完全相同，冒号后DiffusionConstants表示这个参数应该传入DiffusionConstants实例
    sqrt_alphas_cumprod_t = constants.sqrt_alphas_cumprod[t].reshape(-1,1,1,1)
    sqrt_one_minus_alphas_cumprod_t = constants.sqrt_one_minus_alphas_cumprod[t].reshape(-1,1,1,1)
    return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

@torch.no_grad()  # @torch.no_grad()下面紧挨着的函数不计算梯度，中间不能有其他语句，但可以有空行
def p_sample(model, x_t, t, constants: DiffusionConstants):  # 根据x_t和预测噪声计算x_{t-1}
    # 获取当前时间步的系数
    betas_t = constants.betas[t].reshape(-1,1,1,1)
    sqrt_one_minus_alphas_cumprod_t = constants.sqrt_one_minus_alphas_cumprod[t].reshape(-1,1,1,1)
    alphas_t = constants.alphas[t].reshape(-1, 1, 1, 1)
    # 模型预测噪声
    pred_noise = model(x_t, t)

    posterior_mean = (x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * pred_noise) / torch.sqrt(alphas_t)
    # 计算后验方差
    posterior_variance_t = constants.posterior_variance[t].reshape(-1, 1, 1, 1)
    # 采样噪声（最后一步t=0不加噪声）
    noise = torch.randn_like(x_t)
    nonzero_mask = (t != 0).float().reshape(-1, 1, 1, 1)
    sample = posterior_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise
    return sample

@torch.no_grad()
def p_sample_loop(model, shape, constants: DiffusionConstants, device):  # 用于在推理时生成图像，运行完从 T 到 0，p_sample只是从上步得到下步
    # shape[0]决定了推理时一次生成多少张图像
    x_t = torch.randn(shape, device=device)
    for i in tqdm(reversed(range(constants.timesteps)), desc='sampling timesteps'):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)  # 为每张图像生成对应的时间步
        # torch.full用于创建一个填充了指定值的张量的函数，第一个参数张量形状，第二个参数填充的值
        x_t = p_sample(model, x_t, t, constants)
    return x_t
# reversed(range(constants.timesteps))生成一个从constants.timesteps-1倒序到0的序列，tqdm()包装了这个序列，为循环创建了一个进度条
# desc='sampling timesteps'为这个进度条设置了描述文字"sampling timesteps"。
# 当程序执行这个循环时，会在控制台显示一个动态更新的进度条，包含以下信息：
# 1、已完成的迭代占总迭代的百分比。2、已用时间。3、估计剩余时间。4、迭代速度（每秒多少次迭代）。5、自定义的描述文字 "sampling timesteps"
# 这在处理大量迭代（比如这里的时间步采样）时非常有用，可以让用户了解程序的执行进度，避免误以为程序卡住。

class CifarDataset(Dataset):
    def __init__(self, path, transform):
        self.X_list = [np.load(os.path.join(path, f)) for f in os.listdir(path) if f.endswith('npy')]
        self.transform = transform
    def __len__(self):
        return len(self.X_list)
    def __getitem__(self, item):
        image = torch.from_numpy(self.X_list[item]).permute(2, 0, 1)
        image = image.float() / 255.0
        if self.transform:
            image = self.transform(image)
        return image

def sample_show(args):
    image = np.load(os.path.join(args.data_dir, "100.npy"))
    plt.imshow(image)
    plt.show()

# Training function
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 32
    channels = 3
    transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # normalize to [-1,1]

    image = CifarDataset(args.data_dir, transform)
    dataloader = DataLoader(image, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    model = SimpleUNet(in_channels=channels, base_ch=args.base_channels, time_emb_dim=args.time_emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    constants = DiffusionConstants(timesteps=args.timesteps, device=device)

    global_step = 0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader)  # 给数据加载器添加一个可视化进度条，方便在训练或推理时看到循环进度。
        for x_batch in pbar:
        # pbar也是一个可迭代对象（内部就是调用 dataloader.__iter__()）。不同点是：pbar在每次__next__()时会多做一件事，更新进度条显示。
            optimizer.zero_grad()
            x = x_batch
            x = x.to(device)
            b = x.shape[0]
            t = torch.randint(0, args.timesteps, (b,), device=device).long()
            noise = torch.randn_like(x)
            x_t = q_sample(x, t, noise, constants)
            pred_noise = model(x_t, t)
            loss = F.mse_loss(pred_noise, noise)

            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % args.log_interval == 0:
                pbar.set_description(f"epoch {epoch} loss {loss.item():.5f}")
            if global_step % args.sample_interval == 0:
                # sample and save images
                model.eval()
                samples = p_sample_loop(model, (args.sample_batch, channels, img_size, img_size), constants, device)
                samples = (samples.clamp(-1,1)+1)/2  # to [0,1]
                # torch.clamp()用于将张量元素限制在指定范围内，它可以将张量中小于下限的值设为下限，大于上限的值设为上限，范围内的值保持不变。
                utils.save_image(samples, os.path.join(args.out_dir, f"samples_{global_step}.png"), nrow=3)
                # 保存张量的形状如果是(B,C,H,W)，那么一张图中有B个图像按每行nrow个排列
                model.train()
            if global_step % args.save_interval == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'args': vars(args)
                }, os.path.join(args.out_dir, f"ddpm_{global_step}.pt"))

        # end epoch
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'args': vars(args)
    }, os.path.join(args.out_dir, f"DDPM_final.pt"))

# sampling only (load model)
def sample_only(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    channels = 1
    img_size = 32

    model = SimpleUNet(in_channels=channels, base_ch=args.base_channels, time_emb_dim=args.time_emb_dim).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    constants = DiffusionConstants(timesteps=args.timesteps, device=device)
    samples = p_sample_loop(model, (args.num_samples, channels, img_size, img_size), constants, device)
    samples = (samples.clamp(-1,1)+1)/2  # 变成[0,1]，从而合法显示
    os.makedirs(args.out_dir, exist_ok=True)
    utils.save_image(samples, os.path.join(args.out_dir, f"samples_{os.path.basename(args.model_path)}.png"), nrow=8)
    # nrow 参数用于指定保存图像网格时，每一行显示的图像数量。它决定了如何将批量（batch）中的多张图像排列成一张单独的输出图片。
    print("Saved samples to", args.out_dir)

# main & arg parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train','sample'])
    parser.add_argument('--data_dir', type=str, default='C:/python/pythonProject/deep_learning/mnist_numpy/train/8')  # 训练数据位置
    parser.add_argument('--out_dir', type=str, default='C:/python/pythonProject/deep_learning/aa')  # 参数和输出图片的保存位置
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--time_emb_dim', type=int, default=256)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--sample_interval', type=int, default=500)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--sample_batch', type=int, default=9)
    # sampling args
    parser.add_argument('--model_path', type=str, default='C:/python/pythonProject/deep_learning/aa')  # 推理时用的已训练好的参数的位置
    parser.add_argument('--num_samples', type=int, default=9)  # 推理时生成的图像数量
    args = parser.parse_args()

    sample_show(args)

    if args.mode == 'train':
        train(args)
    else:
        sample_only(args)


















