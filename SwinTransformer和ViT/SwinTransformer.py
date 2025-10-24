import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class DropPath(nn.Module):  # DropPath作用是把一个批次的输出中的一部分直接置零
    # 让一部分样本的某个残差分支（residual branch）输出直接为 0，从而随机地“跳过”某些层（相当于让网络更浅）。
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob  # drop_prob：表示“丢弃整条残差路径”的概率，通常在训练时使用一个小数值（如0.1~0.2）。

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x  # self.training是PyTorch中nn.Module类自带的一个属性，无需用户手动定义，它会自动被初始化并维护
        keep_prob = 1 - self.drop_prob  # keep_prob表示“保留残差路径”的概率
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # x.ndim是张量（Tensor）的一个属性，用于返回张量的维度数量
        # 这里补1为了可以进行可以广播，从而后面可以相乘
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # torch.rand是[0,1)上均匀分布随机数
        random_tensor.floor_()  # 用于对张量中的每个元素执行向下取整（floor）运算，并直接修改原张量的值（不返回新张量）。
        return x.div(keep_prob) * random_tensor  # 张量的.div()方法用于对张量进行除法运算，支持标量、其他张量与当前张量进行元素级（element-wise）的除法操作。

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        # 如果out_features为“空值”（如None、0、空字符串等），则用in_features替代；否则保持out_features原来的值。
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size) -> torch.Tensor:  # 窗口划分
    """将特征图划分为不重叠的若干个窗口"""
    B, H_patch, W_patch, C = x.shape
    x = x.view(B, H_patch // window_size, window_size, W_patch // window_size, window_size, C)
    # 在不使用padding的情况下，window_size必须能整除H/patch_size和W/patch_size
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = x.view(-1, window_size, window_size, C)
    return windows  # [num_windows*B, window_size, window_size, C]

def window_reverse(windows, window_size, H_patch, W_patch) -> torch.Tensor:
    """Reverse windows back to feature map. windows: (num_windows*B, window_size, window_size, C). returns: (B, H_patch, W_patch, C)"""
    B = int(windows.shape[0] / (H_patch * W_patch / window_size / window_size))
    x = windows.view(B, H_patch // window_size, W_patch // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H_patch, W_patch, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads: int, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()  # window_size: (Wh, Ww)  Wh*Ww就是一个窗口里有多少token
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # relative position bias table
        Wh, Ww = window_size
        relative_position_table_size = (2 * Wh - 1) * (2 * Ww - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(relative_position_table_size, num_heads))
        # 相对位置偏置表，形状为(relative_position_table_size, num_heads)
        # 这样设计表明模型只关心两个token之间的相对位置关系，而不关心它们在整幅图中的绝对位置。
        # 这样泛化能力强，参数更少，模型不会“记住具体坐标”，而是学会“左边、右边、上方”等普遍空间关系，
        # ViT的绝对位置编码“我是第17个token”、“我在图像的第3行第5列”关心绝对位置
        # Swin的相对位置偏置“我在你右边一个格子”、“我在你下方两个格子”关心相对关系

        # 建立相对位置编码索引，行偏移和列偏移相同的两个token应该映射到同一索引，不同的应该映射到不同索引
        # 若两对token的相对偏移(Δrow, Δcol)完全相同那么它们使用同一个相对位置偏置索引（共享同一组可学习偏置）。
        # 若两对token的相对偏移不同那么必须映射到不同的索引（对应不同的偏置）。
        coords_h = torch.arange(Wh)  # 一维张量[0,1,...,Wh-1]
        coords_w = torch.arange(Ww)  # 一维张量[0,1,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        # torch.meshgrid的输出是一个元组（tuple），元组中包含的元素数量与输入张量的数量一致，每个元素对应一个维度上的网格矩阵。
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += Wh - 1  # 将索引值范围变成0到2*Wh-2
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= 2 * Ww - 1
        # 乘完2*Ww-1后每个行偏移都有一个独立的索引区间，每个行偏移的容量就变了，这样每个行偏移都能与可能的列偏移有唯一的表示
        # 如果不乘的话，（行偏移1，列偏移2）和（行偏移2，列偏移1）的和就一样了
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)  # 截断正态分布初始化，__init__方法中直接使用就可以

    def forward(self, x, mask = None):  # x: (num_windows*B, N, C)，这里N = 一个窗口内的token数量，就是Wh*Ww
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B_, num_heads, N, head_dim)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, num_heads, N, N)
        # 注意力分数矩阵与相对位置偏置相加
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)  # N, N, num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # 1, num_heads, N, N
        attn = attn + relative_position_bias  # (B_, num_heads, N, N)
        if mask is not None:
            # mask: (num_windows, N, N)
            nW = mask.shape[0]  # num_windows
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)  # (B_, num_heads, N, N)
        else:
            attn = F.softmax(attn, dim=-1)  # (B_, num_heads, N, N)
        attn = self.attn_drop(attn)  # (B_, num_heads, N, N)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # (attn @ v).shape: (B_, num_heads, N, head_dim)，x.shape: (B_, N, C)
        x = self.proj(x)  # 这个就是Wo
        x = self.proj_drop(x)
        return x  # x.shape: (B_, N, C)

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.input_resolution = input_resolution  # (H_patch, W_patch)
        self.window_size = window_size  # window_size表示一个局部窗口中包含的patch数量（沿每个空间维度）。
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(window_size, window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

        if self.shift_size > 0:
            H_patch, W_patch = self.input_resolution
            img_mask = torch.zeros((1, H_patch, W_patch, 1))
            h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            cnt = 0  # count
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            # 先给同一个窗口里的token打上编号，然后经过展平后，如果相减后为0的话那么原来就属于同一个区域，如果不为0的话就不属于同一个区域
            mask_windows = window_partition(img_mask, window_size)  # [num_window, window_size, window_size, 1]
            mask_windows = mask_windows.view(-1, window_size * window_size)  # [num_window, window_size*window_size]
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [num_window, window_size*window_size, window_size*window_size]
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))  # [num_window, window_size*window_size, window_size*window_size]
            # tensor.masked_fill(mask, value)方法用于根据掩码（mask）将张量中满足条件的元素替换为指定值（value）。
            # 其核心逻辑是：掩码中为True的位置，对应张量的元素会被替换成value：为False的位置则保持原值。
            # 这行的作用是找出所有不等于0的位置（说明两个token属于不同区域）；把这些位置的值替换为-100.0
            # 找出所有等于0的位置（说明两个token属于同一窗口）；把这些位置的值替换为0.0；
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor):  # x.shape: [B, H/patch*W/patch, C]
        B, L, C = x.shape
        H_patch, W_patch = self.input_resolution
        assert L == H_patch * W_patch, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)  # x.shape: [B, H/patch*W/patch, C]
        x = x.view(B, H_patch, W_patch, C)  # x.shape: [B, H_patch, W_patch, C]

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # torch.roll用于对张量进行循环移位（滚动）的函数，即按照指定维度和步长将张量元素“循环移动”，移出边界的元素会从另一侧重新进入。
            # torch.roll(input, shifts, dims=None)  input：输入张量。返回：移位后的新张量，原张量不变。
            # shifts：移位的步长（可以是整数或元组）。正数表示沿维度增大的方向移位，负数表示沿维度减小的方向移位。
            # dims：指定移位的维度（可以是整数或元组，默认None表示对所有元素展平后移位）。
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # x_windows.shape: [nW*B, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # x_windows.shape: [nW*B, window_size*window_size, C]

        # W-MSA or SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # attn_windows.shape: [nW*B, window_size*window_size, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # attn_windows.shape: [nW*B, window_size, window_size, C]
        shifted_x = window_reverse(attn_windows, self.window_size, H_patch, W_patch)  # shifted_x.shape: [B, H_patch, W_patch, C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H_patch * W_patch, C)  # x.shape: [B, H_patch * W_patch, C]

        # FFN
        x = shortcut + self.drop_path(x)  # x.shape: [B, H_patch * W_patch, C]
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # x.shape: [B, H_patch * W_patch, C]
        return x

class PatchEmbed(nn.Module):
    """Image to Patch Embedding using a conv layer (like ViT patch embedding). default patch_size = 4"""
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):  # x.shape: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, H/patch*W/patch, C]
        x = self.norm(x)
        return x

class PatchMerging(nn.Module):
    """Downsample (H, W) -> (H/2, W/2) and increase channels."""
    def __init__(self, input_resolution, dim: int):
        super().__init__()  # input_resolution: (int, int)
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):  # x.shape: [B, H_patch*W_patch, C]
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W
        x = x.view(B, H, W, C)
        # 把相邻的2×2 patch在通道维上拼接起来，再通过一个线性层将通道数压缩到原来的1/2
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # B, H/2, W/2, 4*C
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x

class BasicLayer(nn.Module):
    """A basic Swin layer for one stage that contains multiple Swin blocks and an optional downsampling (PatchMerging)."""
    def __init__(self, dim, input_resolution, depth, num_heads,
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., downsample=None):
        super().__init__()  # input_resolution: (int, int)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            block = SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                                         window_size=window_size, shift_size=shift_size,
                                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                                         attn_drop=attn_drop, drop_path=drop_path if isinstance(drop_path, float) else (drop_path[i] if isinstance(drop_path, list) else 0.))
            self.blocks.append(block)

        self.downsample = downsample(input_resolution, dim) if downsample is not None else None

    def forward(self, x: torch.Tensor):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class SwinTransformer(nn.Module):
    def __init__(self, img_size=512, patch_size=4, in_chans=3, num_classes=10,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        patches_resolution = (img_size // patch_size, img_size // patch_size)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # drop path rule 逐层递增的随机深度衰减规则。
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        dims = [embed_dim, embed_dim * 2, embed_dim * 4]
        cur = 0
        for i_layer in range(len(depths)):
            layer = BasicLayer(dim=dims[i_layer], input_resolution=patches_resolution, depth=depths[i_layer],
                               num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[cur:cur + depths[i_layer]], downsample=PatchMerging if (i_layer < len(depths) - 1) else None)
            self.layers.append(layer)
            cur += depths[i_layer]
            if i_layer < len(depths) - 1:  # 更新分辨率
                patches_resolution = (patches_resolution[0] // 2, patches_resolution[1] // 2)

        self.norm = nn.LayerNorm(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # nn.AdaptiveAvgPool1d固定对输入张量的最后一个维度进行池化
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()  # 权重初始化

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):  # x.shape: [B, C, H, W]，输入的x是原始图像
        x = self.patch_embed(x)  # patch_embed输出的x的形状是[B, H/patch*W/patch, C]
        x = self.pos_drop(x)  # 矩阵中元素随机置0

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # [B, N_final, C_final]
        x = x.transpose(1, 2)  # [B, C_final, N_final]
        x = self.avgpool(x)  # [B, C_final, 1]
        x = torch.flatten(x, 1)  # [B, C_final]
        x = self.head(x)  # [B, num_classes]
        return x

class CIFAR10(Dataset):
    def __init__(self, path, transform=None):
        self.cifar10_dir = torch.load(path)
        self.transform = transform
    def __len__(self):
        return self.cifar10_dir["data"].shape[0]
    def __getitem__(self, idx):
        image = self.cifar10_dir["data"][idx]
        if self.transform:
            image = self.transform(image)
        return image, self.cifar10_dir["labels"][idx]

def calculate_accuracy(loader, model, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            _, predicted = torch.max(output.data, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    model.train()
    return correct / total

transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
full_train_dataset = CIFAR10("C:/python/pythonProject/deep_learning/CIFAR10_npy/CIFAR10_train.pt", transform=transform)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
test_dataset = CIFAR10("C:/python/pythonProject/deep_learning/CIFAR10_npy/CIFAR10_test.pt")
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100
model = SwinTransformer(img_size=32, patch_size=2, in_chans=3, num_classes=10, embed_dim=48, depths=[2, 2, 6], num_heads=[2, 4, 8], window_size=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    if epoch % 1 == 0:
        train_acc = calculate_accuracy(train_loader, model, device)
        val_acc = calculate_accuracy(val_loader, model, device)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Train Accuracy: {train_acc:.2f}, Val Accuracy: {val_acc:.2f}")

test_acc = calculate_accuracy(test_loader, model, device)
print(f"Test Accuracy: {test_acc:.2f}")




