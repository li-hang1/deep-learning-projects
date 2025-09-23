import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # 用一个卷积把每个patch映射到embedding 维度
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 把每个patch的3个通道像素展平后拼成一个大向量，再统一乘以一个权重矩阵W效果完全相同
        # 即和这样效果相同，输入 patch: [B, 3, P, P]
        # patch = patch.flatten(1)  # [B, 3*P*P]
        # out = patch @ W.T + b     # [B, embed_dim]
    def forward(self, x):  # x: [B, C, H, W]
        x = self.proj(x)              # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2)              # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)         # [B, num_patches, embed_dim]
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc = nn.Linear(embed_dim, embed_dim)  # 这个就是W_o

    def forward(self, x):
        B, num_patches, embed_dim = x.shape
        qkv = self.qkv(x)                     # [B, num_patches, 3*embed_dim]
        qkv = qkv.reshape(B, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)     # 3, B, num_heads, num_patches, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 前两维对齐保留，最后两维做矩阵乘法。
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, heads, num_patches, num_patches]，就是Transformer中的QK转置
        attn = attn.softmax(dim=-1)
        out = attn @ v                       # [B, heads, num_patches, head_dim]
        out = out.transpose(1, 2).reshape(B, num_patches, embed_dim)
        out = self.fc(out)  # W_o
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # 残差连接 + 注意力
        x = x + self.mlp(self.norm2(x))    # 残差连接 + MLP
        return x

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=64, depth=6, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 分类 token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))  # 位置编码
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])  # Transformer Encoder
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)  # 分类头

    def forward(self, x):
        x = self.patch_embed(x)               # [B, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # expand()方法用于将张量扩展到新的形状，-1表示保持该维度的原始大小不变
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+num_patches, embed_dim]
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]  # [B, embed_dim]
        out = self.head(cls_out)  # [B, num_classes]
        return out

class CifarDataset(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        for label in os.listdir(path):
            label_dir = os.path.join(path, label)
            for file_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, file_name)
                image = np.load(image_path)
                self.data.append(image)
                self.labels.append(int(label))
        self.data = np.array(self.data).astype(np.float32)
        self.labels = np.array(self.labels).astype(np.int64)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = image.float() / 255.0
        if self.transform:
            image = self.transform(image)
        return image, label

def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            _, predicted = torch.max(outputs, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    model.train()
    return correct / total

full_train_dataset = CifarDataset("C:/python/pythonProject/deep_learning/CIFAR10_npy/train")
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
test_dataset = CifarDataset("C:/python/pythonProject/CIFAR10_npy/test")
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

epochs = 10
model = ViT(img_size=32, patch_size=4, in_channels=3, num_classes=10)
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
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Train Accuracy: {train_acc:.2f}, Val Accuracy: {val_acc:.2f}")

test_acc = calculate_accuracy(test_loader, model)
print(f"Test Accuracy: {test_acc:.2f}")


