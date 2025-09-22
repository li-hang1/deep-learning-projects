import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import random
import os
"""
在 YOLOv3 里，特征图的每个像素点就是一个网格 cell，对应原图的一个区域；
每个 cell 放置 3 个 anchor
每个 anchor 预测 1 个候选框
通过 (c_x, c_y) + sigmoid 偏移量来解码回原图坐标。
"""
# 基础卷积块：Conv2d + BatchNorm + LeakyReLU
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2  # same padding手动实现
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))
# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels // 2, kernel_size=1),  # 先压缩再提取空间特征，再还原通道 + 残差连接
            ConvBlock(channels // 2, channels, kernel_size=3)
        )
    def forward(self, x):
        return x + self.block(x)
# Darknet-53 主干网络
class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(3, 32, 3),
            ConvBlock(32, 64, 3, stride=2)
        )
        # 每个阶段包含若干残差块
        self.stage1 = nn.Sequential(ResidualBlock(64))
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, 3, stride=2),
            *[ResidualBlock(128) for _ in range(2)]
        )
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256, 3, stride=2),
            *[ResidualBlock(256) for _ in range(8)]
        )
        self.stage4 = nn.Sequential(
            ConvBlock(256, 512, 3, stride=2),
            *[ResidualBlock(512) for _ in range(8)]
        )
        self.stage5 = nn.Sequential(
            ConvBlock(512, 1024, 3, stride=2),
            *[ResidualBlock(1024) for _ in range(4)]
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        out3 = x  # 用于 52x52 特征图
        x = self.stage4(x)
        out2 = x  # 用于 26x26 特征图
        x = self.stage5(x)
        out1 = x  # 用于 13x13 特征图
        return out1, out2, out3

# YOLOv3 检测头
class YOLOv3Head(nn.Module):
    def __init__(self, num_classes=80, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # 13x13
        self.head1 = nn.Conv2d(1024, num_anchors * (5 + num_classes), 1)
        # 26x26
        self.head2 = nn.Conv2d(512, num_anchors * (5 + num_classes), 1)
        # 52x52
        self.head3 = nn.Conv2d(256, num_anchors * (5 + num_classes), 1)

    def forward(self, f1, f2, f3):
        out1 = self.head1(f1)  # 13x13，大物体
        out2 = self.head2(f2)  # 26x26，中物体
        out3 = self.head3(f3)  # 52x52。小物体
        return out1, out2, out3
"""
对于 head1 (13×13 特征图)，输出 (B,255,13,13)：
在 cell(5,8) 位置：
通道 [0:85] → anchor1 的 (tx, ty, tw, th, to, c1..c80)
通道 [85:170] → anchor2 的预测
通道 [170:255] → anchor3 的预测
这样一个 cell 上的 255 维向量 就存了 3 个候选框的完整信息。

cell 决定框的中心位置（落在哪个 stride×stride 的区域）
cell(i,j) → 原图上 ( (j+0.5)*stride, (i+0.5)*stride )
anchor 决定框的大小
给定 (p_w, p_h)，直接在原图上放置这个尺寸的矩形框
这样，在原图上，每个 cell + 每个 anchor = 一个候选框。
"""
# YOLOv3 完整网络（结构）
class YOLOv3(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = Darknet53()
        self.head = YOLOv3Head(num_classes=num_classes)

    def forward(self, x):
        f1, f2, f3 = self.backbone(x)
        return self.head(f1, f2, f3)

def wh_to_boxes(wh):  # 把宽高变成样本框
    """
    在 K-means 聚类 anchors 的时候，我们其实关心的是不同宽高 (w,h) 的相似性。但 IoU 需要矩形框的 (x1,y1,x2,y2) 格式，
    所以就用这个函数把 (w,h) 统一转换为“以 (0,0) 为起点的 box”，这样 IoU 的比较只体现宽高，不会受到位置影响。
    输入wh：tensor，[N, 2]，每一行表示一个框的宽 w 和 高 h。输出boxes：tensor，[N, 4]，每一行是 (x1,y1,x2,y2)。
    下面实际 实际使用当中N等于k，即锚框的个数
    """
    boxes = torch.zeros((wh.shape[0], 4), dtype=torch.float32)
    boxes[:, 2] = wh[:, 0]  # x2 = w
    boxes[:, 3] = wh[:, 1]  # y2 = h
    return boxes

def compute_iou(proposals, gt_boxes):
    """
    proposals: [N, 4] (x1, y1, x2, y2)
    gt_boxes: [M, 4] (x1, y1, x2, y2)
    return: IoU matrix [N, M]
    """
    N = proposals.shape[0]
    M = gt_boxes.shape[0]

    anchors_area = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1])
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    iou = torch.zeros((N, M), device=proposals.device)
    for i in range(M):
        xx1 = torch.max(proposals[:, 0], gt_boxes[i, 0])
        yy1 = torch.max(proposals[:, 1], gt_boxes[i, 1])
        xx2 = torch.min(proposals[:, 2], gt_boxes[i, 2])
        yy2 = torch.min(proposals[:, 3], gt_boxes[i, 3])
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        union = anchors_area + gt_area[i] - inter
        iou[:, i] = inter / union
    return iou

def kmeans_anchors_iou(dataset, k=9, img_size=416, max_iter=100):
    # 1. 提取 wh
    wh = []
    for img_name in dataset.img_files:
        label_path = os.path.join(dataset.label_dir, img_name.replace('.jpg', '.txt'))
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls, x1, y1, x2, y2 = line.strip().split()
                    x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
                    w = x2 - x1
                    h = y2 - y1
                    wh.append([w, h])
    wh = np.array(wh)
    if len(wh) == 0:
        raise ValueError("没有标注框，无法聚类 anchors")  # 只有在整个数据集都没有任何框的时候，才会报错。

    # 所有框映射到网络输入的默认的大小
    wh = wh / wh.max() * img_size  # array.max()方法返回这个数组中所有数中的最大值
    wh_t = torch.tensor(wh, dtype=torch.float32)  # [N, 2]，所有样本宽高

    # 2. 随机初始化 k 个 anchors
    indices = random.sample(range(len(wh)), k)
    # random.sample(population, k)
    # population：需要从中选择元素的序列（必须是可迭代对象，且长度至少为 k）。k：需要选择的元素数量（必须是正整数，且不能大于population的长度）
    anchors = wh_t[indices].clone()  # clone()方法得到的新张量拥有独立的内存空间，修改原张量不会影响副本，反之亦然

    # 3. KMeans 迭代
    for _ in range(max_iter):
        # (a) 计算 IoU
        gt_boxes = wh_to_boxes(wh_t)  # shape: [N, 4]，所有样本框
        prop_boxes = []
        for i in range(k):
            w, h = anchors[i]
            prop_boxes.append([0, 0, w.item(), h.item()])
        prop_boxes = torch.tensor(prop_boxes, dtype=torch.float32)  # shape: [k, 4]，聚类中先选定的样本框

        ious = compute_iou(prop_boxes, gt_boxes)  # (k, N)
        assignments = torch.argmax(ious, dim=0)   # (N,) 每个样本分到哪个 anchor

        # (b) 更新 anchors
        new_anchors = []
        for i in range(k):
            cluster_points = wh_t[assignments == i]
            if len(cluster_points) > 0:
                new_anchors.append(cluster_points.mean(0))
            else:
                new_anchors.append(anchors[i])  # 在当前迭代里没有任何样本分到这个anchor上时，anchor不更新，保持原值。
        new_anchors = torch.stack(new_anchors)

        # (c) 收敛检查
        if torch.allclose(new_anchors, anchors, atol=1e-2):
            break
        # torch.allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False)，输出True或False
        # 用于比较两个张量是否在数值上“近似相等”的函数。它主要用于判断两个张量的所有元素是否在指定的容差范围内相等，常用于数值计算中的精度验证（如测试结果是否符合预期）。
        anchors = new_anchors

    # 4. 排序（按面积）
    areas = anchors[:, 0] * anchors[:, 1]  # 计算面积
    order = torch.argsort(areas)  # 按面积升序排序
    # torch.argsort()是一个用于获取张量元素排序后索引位置的函数。它返回的不是排序后的元素本身，而是原张量中元素按指定顺序（升序或降序）排列时对应的索引值。
    # 返回的是与输入张量形状相同的索引张量（数据类型为 long）
    anchors = anchors[order]  # 重新排序
    return [(int(w), int(h)) for w, h in anchors]

class YOLOv3Loss(nn.Module):
    def __init__(self, anchors, num_classes=80, img_size=416, ignore_thresh=0.5, lambda_coord=5, lambda_noobj=0.5):
        """
        anchors: list of tuples [[w1,h1], [w2,h2], ...] 原图尺度
        通常 YOLOv3 9 个 anchor, 三个尺度，每个尺度 3 个 anchor
        """
        super().__init__()
        self.anchors = torch.tensor(anchors, dtype=torch.float32, device=device)  # (9,2)
        self.num_classes = num_classes
        self.img_size = img_size
        self.ignore_thresh = ignore_thresh
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, preds, targets):
        """
        preds: list of 3 feature map输出 [(B,3*(5+num_classes),H,W), ...]
        targets: list[Tensor], 长度 = B，每个tensor是二维张量，分别对应一个图中的目标框，每行分别对应[x,y,w,h,class], normalized 0~1
        x,y真实框的中心点坐标（相对于整张图，归一化到 0~1）。w,h真实框的宽和高（相对于整张图，归一化到 0~1）。class类别索引（整数）。
        preds, anchor_mask要匹配，比如第一个都是大物体，第二个都是中物体，第三个都是小物体
        一张图里大多数cell+anchor的标签都是0，只有和真实框对齐的少数几个cell+anchor会被赋予具体的偏移值和one-hot类别。YOLO的训练就是在这种极度稀疏的监督下进行的。
        """
        device = preds[0].device
        total_loss = 0
        num_pos_all = 0
        anchor_masks = [[6,7,8],[3,4,5],[0,1,2]]  # YOLOv3 anchor assignment to feature maps
        for i, pred in enumerate(preds):
            B, _, H, W = pred.shape  # H, W是当前特征图的高度和宽度
            num_anchors = len(anchor_masks[i])
            pred = pred.view(B, num_anchors, 5+self.num_classes, H, W).permute(0,1,3,4,2).contiguous()
            anchor_wh = self.anchors[anchor_masks[i]] / self.img_size  # anchor尺度(num_anchors, 2)，w,h归一化到 0~1
            # 把真实框(ground truth)转换成和网络输出同格式的“监督标签”，这样才能和预测值一一对应计算损失函数。
            # 网络输出是特征图坐标系下的(tx, ty, tw, th)；(tx, ty)不是直接预测中心点，而是预测相对网格cell的偏移量
            # (tw, th)不是直接预测宽高，而是预测相对anchor的对数缩放量。所以必须把gt变换成t*这种格式。
            tx = torch.zeros_like(pred[...,0])  # (B, num_anchors, H, W)  num_anchors是单个尺度的anchor数量
            ty = torch.zeros_like(pred[...,1])  # (B, num_anchors, H, W)
            tw = torch.zeros_like(pred[...,2])  # (B, num_anchors, H, W)
            th = torch.zeros_like(pred[...,3])  # (B, num_anchors, H, W)
            tconf = torch.zeros_like(pred[...,4])  # (B, num_anchors, H, W)
            tcls = torch.zeros_like(pred[...,5:])  # (B, num_anchors, H, W, num_classes)
            # tx,ty表示真实框中心点在网格cell内的偏移量[0,1)。监督模型学sigmoid输出。
            # tw,th表示真实框相对anchor的比例（取 log）。监督模型学线性输出。
            # tconf表示这个cell+anchor是否有目标。监督模型学sigmoid输出。
            # tcls：One-hot标签，表示类别。损失用独立的二元交叉熵

            # ignore_mask，用于标记哪些负样本需要忽略，与真实框有较高的 IoU，但不是最佳匹配的那些锚框。这些如果算作负样本，会惩罚过头，妨碍收敛。
            ignore_mask = torch.zeros_like(tconf, dtype=torch.bool)  # (B, num_anchors, H, W)
            for b in range(B):
                if len(targets[b]) == 0:
                    continue  # 如果一个样本图像中没有框，那么tconf都是0，所以网络输出中的类别标签都拟合为0
                gt = targets[b]  # (gt_num,5)，每行为[x, y, w, h, cls]
                gt_cls = gt[:, 4].long()  # .long()是张量的一个类型转换方法，用于将张量的数据类型转换为64位整数类型（int64）。
                gt_boxes = gt[:, :4]  # (gt_num,4)
                ious = []
                ignore_mask_b = torch.zeros((num_anchors, H, W), dtype=torch.bool, device=device)
                for n in range(len(gt)):
                    gx, gy, gw, gh = gt_boxes[n]  # gt框的中心点和长宽
                    gt_x1, gt_y1 = gx - gw / 2, gy - gh / 2  # gt框的左上角和右下角，这四个都是数
                    gt_x2, gt_y2 = gx + gw / 2, gy + gh / 2
                    # 把锚框放在同一中心点 (gx,gy)，只比较宽高
                    anchor_boxes = torch.zeros((num_anchors, 4), device=device)
                    anchor_boxes[:, 0] = gx - anchor_wh[:, 0] / 2  # x1
                    anchor_boxes[:, 1] = gy - anchor_wh[:, 1] / 2  # y1
                    anchor_boxes[:, 2] = gx + anchor_wh[:, 0] / 2  # x2
                    anchor_boxes[:, 3] = gy + anchor_wh[:, 1] / 2  # y2
                    # 计算交集
                    inter_x1 = torch.max(gt_x1, anchor_boxes[:, 0])  # (num_anchors,)
                    inter_y1 = torch.max(gt_y1, anchor_boxes[:, 1])  # (num_anchors,)
                    inter_x2 = torch.min(gt_x2, anchor_boxes[:, 2])  # (num_anchors,)
                    inter_y2 = torch.min(gt_y2, anchor_boxes[:, 3])  # (num_anchors,)
                    inter_w = (inter_x2 - inter_x1).clamp(min=0)  # (num_anchors,)
                    inter_h = (inter_y2 - inter_y1).clamp(min=0)  # (num_anchors,)
                    inter_area = inter_w * inter_h  # (num_anchors,)
                    # 计算并集
                    gt_area = gw * gh  # (num_anchors,)
                    anchor_area = anchor_wh[:, 0] * anchor_wh[:, 1]  # (num_anchors,)
                    union_area = gt_area + anchor_area - inter_area + 1e-16  # (num_anchors,)
                    # IoU
                    iou = inter_area / union_area  # (num_anchors,)
                    ious.append(iou)
                    gi, gj = min(int(gx * W), W - 1), min(int(gy * H), H - 1)
                    ignore_mask_b[:, gj, gi] |= (iou > self.ignore_thresh)  # 高 IoU anchor 忽略负样本 loss
                    # ignore_mask_b[:, gj, gi]形状(num_anchors,)，|表示两个条件有一个为True那么就是True（相当于保留原有的True并添加新的True）
                ious = torch.stack(ious, dim=0)  # (len(gt), num_anchors)
                best_n = torch.argmax(ious, dim=1)  # (len(gt),)
                ignore_mask[b] = ignore_mask_b

                for idx, n in enumerate(best_n):
                    gx, gy, gw, gh = gt[idx,:4]  # gt.shape:(gt_num,5)，[x, y, w, h, cls]
                    gi = min(int(gx*W), W-1)  # 特征图坐标
                    gj = min(int(gy*H), H-1)
                    tx[b,n,gj,gi] = gx * W - gi  # (gx * W)和(gy * H)是真实框中心点在特征图网格上的位置
                    ty[b,n,gj,gi] = gy * H - gj  # (gj, gi)定位的是真实框属于哪个网格
                    tw[b,n,gj,gi] = torch.log(gw/anchor_wh[n,0])  # gw*W/anchor_wh[n,0]*W
                    th[b,n,gj,gi] = torch.log(gh/anchor_wh[n,1])  # gh*H/anchor_wh[n,1]*H
                    tconf[b,n,gj,gi] = 1  # 在特征图上标记哪一个像素存在目标框。
                    tcls[b,n,gj,gi,gt_cls[idx]] = 1

            # losses
            # 坐标损失 (只在正样本位置)
            loss_x = F.binary_cross_entropy_with_logits(pred[..., 0][tconf == 1], tx[tconf == 1], reduction='sum')
            loss_y = F.binary_cross_entropy_with_logits(pred[..., 1][tconf == 1], ty[tconf == 1], reduction='sum')
            # 宽高损失 (只在正样本位置，常用 MSE)
            loss_w = F.mse_loss(pred[..., 2][tconf == 1], tw[tconf == 1], reduction='sum')
            loss_h = F.mse_loss(pred[..., 3][tconf == 1], th[tconf == 1], reduction='sum')
            # 置信度损失（分为有目标和无目标）
            loss_conf_obj = F.binary_cross_entropy_with_logits(pred[..., 4][tconf == 1], tconf[tconf == 1], reduction='sum')
            mask_noobj = (tconf == 0) & (~ignore_mask)  # 负样本 loss，只计算未被 ignore 的负样本，~是取反
            loss_conf_noobj = F.binary_cross_entropy_with_logits(pred[..., 4][mask_noobj], tconf[mask_noobj], reduction='sum')
            # 类别损失 (只在正样本位置)
            loss_cls = F.binary_cross_entropy_with_logits(pred[..., 5:][tconf == 1], tcls[tconf == 1], reduction='sum')

            num_pos_all += tconf.sum().item()
            total_loss += self.lambda_coord*(loss_x + loss_y + loss_w + loss_h) + loss_conf_obj + self.lambda_noobj*loss_conf_noobj + loss_cls

        if num_pos_all > 0:
            return total_loss / num_pos_all
        else:
            return total_loss / B

class YOLODataset(Dataset):
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
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls, x1, y1, x2, y2= line.strip().split()
                    cls = int(cls)
                    x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
                    # 计算中心点坐标和宽高
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    box_width = x2 - x1
                    box_height = y2 - y1
                    # 如果需要归一化到[0,1]范围（YOLO通常使用这种方式）
                    center_x /= width
                    center_y /= height
                    box_width /= width
                    box_height /= height
                    boxes.append([center_x, center_y, box_width, box_height, cls])
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        return img, boxes

def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(targets)

# transform = transforms.Compose([transforms.Resize((416, 416)), transforms.ToTensor()])
# dataset = YOLODataset(
#     img_dir="C:/python/pythonProject/deep_learning/object_detection/Dataset/Train/images",
#     label_dir="C:/python/pythonProject/deep_learning/object_detection/Dataset/Train/labels",
#     transform=transform
# )
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# epochs = 2
# num_classes = 6
# anchors = kmeans_anchors_iou(dataset)
# model = YOLOv3(num_classes=num_classes).to(device)
# criterion = YOLOv3Loss(anchors=anchors, num_classes=num_classes)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# for epoch in range(epochs):
#     model.train()
#     for images, targets in dataloader:
#         images = images.to(device)
#         targets = [gt.to(device) for gt in targets]
#         optimizer.zero_grad()
#         out1, out2, out3 = model(images)
#         loss = criterion([out1, out2, out3], targets)
#         loss.backward()
#         optimizer.step()
#         if epoch % 1 == 0:
#             print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

def nms(boxes, scores, iou_threshold=0.7):
    """
    boxes: numpy array, shape [num_anchors,4], format [x1,y1,x2,y2]
    scores: numpy array, shape [num_anchors]
    iou_threshold: float, IoU阈值
    """
    # 按分数从高到低排序
    scores_sorted, indices = torch.sort(scores, descending=True)
    # torch.sort返回一个元组(sorted_tensor, indices)，包含两个元素：
    # sorted_tensor：排序后的张量，形状与输入张量相同。indices：整数张量，存储了排序后元素在原张量中的索引位置，形状与输入张量相同。
    keep = []  # 存放的最终保留下来的框的索引。
    while indices.numel() > 0:
        current = indices[0].item()  # 取当前所有候选框中前景概率最大的那个框的索引
        keep.append(current)
        if indices.numel() == 1:  # 上一步current = indices[0].item()已经保存了最后一个，所以这里是1
            break
        # 找出当前框（current）与剩下所有候选框（rest）的交集区域坐标。
        rest = indices[1:]  # 下面四个形状都是一维数组，长度是余下的框的数量
        x1 = torch.maximum(boxes[current,0], boxes[rest,0])
        y1 = torch.maximum(boxes[current,1], boxes[rest,1])
        x2 = torch.minimum(boxes[current,2], boxes[rest,2])
        y2 = torch.minimum(boxes[current,3], boxes[rest,3])

        inter_w = x2 - x1  # 这两行是当前框与剩余框交集的宽高
        inter_h = y2 - y1

        has_inter = (inter_w > 0) & (inter_h > 0)  # 形状[num_rest]，每个元素是True或False
        # Intersection over Union 交并比
        iou = torch.zeros_like(rest, dtype=torch.float)
        if torch.any(has_inter.float()):
            inter_area = inter_w[has_inter] * inter_h[has_inter]  # 计算交集面积
            area_current = (boxes[current, 2] - boxes[current, 0]) * (boxes[current, 3] - boxes[current, 1])  # (x2-x1)*(y2-y1)
            #   # \ 是续行符，表示下一行是当前行的延续，Python 会把它和下一行当作一行代码执行，等价于写一行，\后不能写注释
            area_rest = (boxes[rest[has_inter], 2] - boxes[rest[has_inter], 0]) * \
                        (boxes[rest[has_inter], 3] - boxes[rest[has_inter], 1])
            # area_rest是一维数组，长度是当前框与剩余框有交集的框的数量，每个元素对应一个有交集的剩余框的面积
            iou_temp = inter_area / (area_current + area_rest - inter_area)
            iou[has_inter] = iou_temp  # 只有有交集的框才更新 iou
        # 保留 IoU <= 阈值的索引
        indices = rest[(iou <= iou_threshold)]
    return torch.tensor(keep)

def decode_all_scales(preds, anchors, num_classes, img_size=416, conf_thresh=0.1):  # 把网络输出转化成可用于显示的标签形式
    """
    preds: list of 3 outputs [out1, out2, out3]
    anchors: list of (w,h) tuples
    返回: list(len=B) 每项 Tensor (总预测框数量, 6) -> [x1,y1,x2,y2,score,class]
    """
    anchor_masks = [[6,7,8],[3,4,5],[0,1,2]]
    device = preds[0].device
    B = preds[0].shape[0]  # 如果只对一张图片进行推理，那么B就是1
    outputs = []

    for b in range(B):
        all_boxes = []
        for i, pred in enumerate(preds):
            mask = anchor_masks[i]
            num_anchors = len(anchor_masks[i])
            B, _, H, W = pred.shape
            anchors_scale = torch.tensor([anchors[m] for m in mask], dtype=torch.float32, device=device)  # 第i个尺度用到的三个anchor的宽高。
            pred = pred.view(B, num_anchors, 5 + num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()
            B_, na, H, W, C = pred.shape
            p = pred[b]  # (na,H,W,C)

            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
            grid_x, grid_y = grid_x.float(), grid_y.float()

            for a in range(na):
                tx, ty, tw, th, to, *tcls = p[a].permute(2,0,1)  # tx,ty,tw,th,to,*tcls这几个每个形状都是(H,W)
                bx = (torch.sigmoid(tx) + grid_x) / W  # 训练时二元交叉熵已经包含sigmoid了，(H,W)
                by = (torch.sigmoid(ty) + grid_y) / H  # (H,W)
                bw = (anchors_scale[a,0]/img_size) * torch.exp(tw)  # (H,W)
                bh = (anchors_scale[a,1]/img_size) * torch.exp(th)  # (H,W)

                conf = torch.sigmoid(to)  # (H,W)
                cls_score = torch.sigmoid(torch.stack(tcls, dim=-1))  # (H,W,num_classes)
                cls_conf, cls_idx = torch.max(cls_score, dim=-1)  # cls_conf,cls_idx这两个形状都是(H,W)

                final_conf = conf * cls_conf  # (H,W)
                mask_keep = final_conf > conf_thresh  # (H,W)
                if mask_keep.sum() == 0:
                    continue
                # 布尔索引会把(H,W)的张量拉平成一维，选出所有满足条件的元素。所以这四个的形状都是(N,)，N = mask_keep.sum()
                x1 = (bx[mask_keep] - bw[mask_keep]/2) * img_size
                y1 = (by[mask_keep] - bh[mask_keep]/2) * img_size
                x2 = (bx[mask_keep] + bw[mask_keep]/2) * img_size
                y2 = (by[mask_keep] + bh[mask_keep]/2) * img_size

                boxes = torch.stack([x1,y1,x2,y2], dim=1)  # (N, 4)
                scores = final_conf[mask_keep].unsqueeze(1)  # (N,1)
                classes = cls_idx[mask_keep].float().unsqueeze(1)  # (N,1)

                dets = torch.cat([boxes, scores, classes], dim=1)  # (N,6)
                all_boxes.append(dets)

        if len(all_boxes) == 0:
            outputs.append(torch.zeros((0,6), device=device))
        else:
            outputs.append(torch.cat(all_boxes, dim=0))
    return outputs


def apply_nms(dets, iou_thresh=0.5):
    """
    dets: Tensor (N,6) -> [x1,y1,x2,y2,score,class]
    返回: Tensor (M,6)
    """
    if dets.numel() == 0:
        return dets
    keep_boxes = []
    for c in dets[:,5].unique():  # torch.Tensor.unique()方法用于返回张量中所有唯一的元素，并默认按升序排列。
        mask = dets[:,5] == c  # mask.shape = (N,)
        cls_boxes = dets[mask]  # cls_boxes.shape = (K, 6)，K是该类别的候选框数
        keep_idx = nms(cls_boxes[:,:4], cls_boxes[:,4], iou_thresh)
        keep_boxes.append(cls_boxes[keep_idx])  # cls_boxes[keep_idx]形状(M, 6)，M ≤ K
    return torch.cat(keep_boxes, dim=0)


# ----------- 使用示例 -----------
def run_inference(model, image_path, anchors, num_classes, device="cpu", img_size=416, conf_thresh=0.25, iou_thresh=0.4):
    model.eval()
    img0 = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img0.size
    img = img0.resize((img_size,img_size))
    img_t = TF.to_tensor(img).unsqueeze(0).to(device)  # img_t.shape: [1, 3, 416, 416]

    with torch.no_grad():
        preds = model(img_t)  # list of 3
        decoded = decode_all_scales(preds, anchors, num_classes, img_size, conf_thresh)
        if len(decoded) > 0:
            dets = torch.cat(decoded, dim=0)  # (N_total, 6)，N_total是推理阶段所有候选框的数量（在做 NMS 之前）。
        else:
            dets = torch.empty((0, 6), device=device)
        dets = apply_nms(dets, iou_thresh)

        # 映射回原图大小
        if dets.numel() > 0:
            dets[:,[0,2]] *= orig_w/img_size
            dets[:,[1,3]] *= orig_h/img_size
    return dets  # detections结果，(K, 6)，K最终输出的检测框数量。

def visualize_detections(image_path, dets, class_names=None):
    """
    dets: Tensor (N,6) -> [x1,y1,x2,y2,score,class]
    class_names: list, 映射 class_id -> class_name
    """
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, figsize=(12, 8))
    # fig就像画框：是整个图形的 “容器”，包含了所有东西（比如你画的图、标题、边框等）。你可以给画框加个大标题，或者调整画框大小。
    # ax就像画框里的画布：是实际画图的地方，线条、点、柱状图都在这里画，坐标轴（x 轴、y 轴）也在这里，设置坐标轴标签、图例也都是在这个“画布”上操作。
    ax.imshow(img)
    if dets.numel() > 0:
        for *box, score, cls in dets.cpu().numpy():
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            # 类别和置信度
            if class_names is not None:
                label = f"{class_names[int(cls)]}: {score:.2f}"
            else:
                label = f"{int(cls)}: {score:.2f}"
            ax.text(x1, y1 - 5, label, color="yellow", fontsize=10, backgroundcolor="black")
    plt.axis("off")
    plt.show()

model_path = "C:/python/pythonProject/deep_learning/yolo.pth"
img_path = "C:/python/pythonProject/deep_learning/object_detection/Dataset/Test/images"
anchors = [(14, 18), (34, 41), (53, 88), (92, 56), (104, 125), (126, 226), (234, 151), (216, 298), (375, 362)]
img_names = os.listdir(img_path)
model = YOLOv3(num_classes=6)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
for img_name in img_names:
    img_test = os.path.join(img_path, img_name)
    dets = run_inference(model, img_test, anchors, num_classes=6)
    visualize_detections(img_test, dets, class_names=["apple", "banana", "grape", "orange", "ananas", "watermelon"])