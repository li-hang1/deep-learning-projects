import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.act = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # kaiming 初始化
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # 零初始化

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x

# def center_crop(feat: torch.Tensor, target_spatial) -> torch.Tensor:
#     """把一个 feature map 的空间尺寸（高 H、宽 W）裁剪到指定的大小 (th, tw)，并且居中裁剪。
#     Assumes feat size >= target size.
#     """
#     _, _, H, W = feat.shape
#     th, tw = target_spatial
#     dh = (H - th) // 2  # 计算上下裁多少，(H - th)和(W - tw)不是整数的话图像偏左上
#     dw = (W - tw) // 2  # 计算左右裁多少
#     return feat[:, :, dh:dh + th, dw:dw + tw]


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_ch: int = 16):
        super().__init__()
        # Contracting path
        self.down1 = DoubleConv(in_channels, base_ch)          # -> 64
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base_ch, base_ch * 2)          # -> 128
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base_ch * 2, base_ch * 4)      # -> 256
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(base_ch * 4, base_ch * 8)      # -> 512
        self.pool4 = nn.MaxPool2d(2)
        self.bottom = DoubleConv(base_ch * 8, base_ch * 16)    # -> 1024

        # Expansive path
        self.upconv4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.up4 = DoubleConv(base_ch * 16, base_ch * 8)
        self.upconv3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.up3 = DoubleConv(base_ch * 8, base_ch * 4)
        self.upconv2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.up2 = DoubleConv(base_ch * 4, base_ch * 2)
        self.upconv1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.up1 = DoubleConv(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)
        nn.init.kaiming_normal_(self.out_conv.weight, mode='fan_in', nonlinearity='relu')
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        # Down
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.down4(self.pool3(x3))
        x5 = self.bottom(self.pool4(x4))

        # Up with cropping and concatenation
        u4 = self.upconv4(x5)
        # c4 = center_crop(x4, (u4.shape[2], u4.shape[3]))  # 按通道拼接，拼接后形状[B, C_d + C_e, H, W]
        u4 = self.up4(torch.cat([u4, x4], dim=1))

        u3 = self.upconv3(u4)
        # c3 = center_crop(x3, (u3.shape[2], u3.shape[3]))
        u3 = self.up3(torch.cat([u3, x3], dim=1))

        u2 = self.upconv2(u3)
        # c2 = center_crop(x2, (u2.shape[2], u2.shape[3]))
        u2 = self.up2(torch.cat([u2, x2], dim=1))

        u1 = self.upconv1(u2)
        # c1 = center_crop(x1, (u1.shape[2], u1.shape[3]))
        u1 = self.up1(torch.cat([u1, x1], dim=1))

        logits = self.out_conv(u1)
        return logits

class HandNailDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(label_path).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        mask = torch.where(mask > 0, 1.0, 0.0)  # 二值化 mask
        # 输出张量的形状与作为条件的张量（这里是 mask > 0 的结果，其形状与 mask 一致）保持一致
        # 只是根据条件对每个位置的元素进行替换（大于 0 的位置替换为 1.0，否则替换为 0.0）
        return image, mask
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

img_dir = "C:/python/pythonProject/deep_learning/nails_segmentation/images"
label_dir = "C:/python/pythonProject/deep_learning/nails_segmentation/labels"
train_dataset = HandNailDataset(img_dir, label_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, num_classes=1).to(device)  # 单通道输出
model.load_state_dict(torch.load("C:/python/pythonProject/deep_learning/unet_hand_nail.pth", map_location=torch.device('cpu')))
# criterion = nn.BCEWithLogitsLoss()  # 二值交叉熵
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# num_epochs = 50
#
# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0
#     for images, masks in train_loader:
#         images = images.to(device)
#         masks = masks.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(images)  # [B,1,H,W]
#         loss = criterion(outputs, masks)
#         loss.backward()
#         optimizer.step()
#
#         epoch_loss += loss.item()
#     print(f"Epoch [{epoch + 1}], Loss: {epoch_loss :.4f}")

# torch.save(model.state_dict(), "unet_hand_nail.pth")
model.eval()
with torch.no_grad():
    img = Image.open("C:/python/pythonProject/deep_learning/nails_segmentation/test_images/F6F9B3E6-FA7B-4DAC-B08C-1AD19BC43A76.jpg").convert("RGB")
    orig_size = img.size[::-1] # (W,H)

    # 预处理：resize → tensor
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    inp = preprocess(img).unsqueeze(0)

    # 预测
    out = model(inp)
    pred = torch.sigmoid(out).cpu().squeeze(0).squeeze(0)  # [128,128]

    # resize 回原图大小
    pred_mask = transforms.Resize(orig_size)(pred.unsqueeze(0)).squeeze(0)
    final_mask = (pred_mask > 0.5).float()
    image = final_mask.detach().numpy()

    plt.imshow(image, cmap='gray')
    plt.show()  # 显示图片