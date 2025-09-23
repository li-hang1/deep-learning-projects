import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms

# Backbone 骨干网络，作用是对输入图像进行特征提取，为后续的目标检测任务（区域提议生成和目标分类/回归）提供高质量的特征图。
class Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)  # ResNet-50的输出大小是固定的，输出图像大小=(输入图像大小/32)，通道数=2048
        self.body = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        self.out_channels = 2048
    def forward(self, x):
        return self.body(x)
"""
RPN 预测：修正 anchor 得到 proposals（粗筛，前景/背景 + 粗定位）
Head 预测：proposals 映射回特征图然后做最终分类 + 边框回归（精调，精细分类 + 精准定位）
"""
class RPN(nn.Module):  # Region Proposal Network (RPN) 区域提议网络
    def __init__(self, in_channels, anchor_scales=[64, 128, 256], anchor_ratios=[0.5, 1, 2]):
        # anchor_scales决定anchor的整体尺寸，anchor_ratios决定anchor的长宽比，w=scale*sqrt(ratio)，h=scale/sqrt(ratio)
        super().__init__()
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.k = len(anchor_scales) * len(anchor_ratios)  # 候选框的数量
        self.conv = nn.Conv2d(in_channels, 512, 3, padding=1)
        self.cls_logits = nn.Conv2d(512, self.k * 2, 1)
        # cls_logits的每个通道的每个特征图元素对应一个anchor的一个分类分数，每两个通道对应一个anchor的[bg, fg]
        self.bbox_pred = nn.Conv2d(512, self.k * 4, 1)  # RPN输出的是anchor的偏移delta，不是框的坐标

    def forward(self, feature_map):
        x = F.relu(self.conv(feature_map))
        cls_logits = self.cls_logits(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logits, bbox_pred
# RPN的输出有两部分：
# (1)前景/背景分类（objectness score）：
# 对每个 anchor 输出一个二分类概率：前景（foreground，含有目标的概率），背景（background，不含目标的概率）
# 也就是说，每个anchor都有一个“是不是目标”的概率。
# (2)边界框回归（bbox regression）：
# 对每个anchor预测相对于某个ground-truth box的偏移量（坐标回归）。通常回归的是4个参数：
# Δx, Δy：anchor中心点相对于真框中心的偏移，Δw, Δh：anchor宽高相对于真框宽高的缩放比

def generate_anchors(feature_map_size, strides, img_size, scales=[64, 128, 256], ratios=[0.5, 1, 2], device='cpu'):
    """
    feature_map_size: tuple (H, W) 特征图大小
    stride: (stride_h, stride_w)，分别表示高和宽方向的stride，原图和特征图的缩放比例，也就是特征图的一个像素对应原图多少个像素。
    img_size: tuple (H, W), 原图大小
    scales: list, anchor 尺寸（基于原图像素）
    ratios: list, anchor 长宽比
    """
    H, W = feature_map_size
    stride_h, stride_w = strides
    anchors = []
    for i in range(H):
        for j in range(W):
            ctr_x = j * stride_w + stride_w / 2  # 原图坐标
            ctr_y = i * stride_h + stride_h / 2
            for scale in scales:
                for ratio in ratios:
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)
                    x1 = ctr_x - w / 2
                    y1 = ctr_y - h / 2
                    x2 = ctr_x + w / 2
                    y2 = ctr_y + h / 2
                    anchors.append([x1, y1, x2, y2])
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
    anchors = clip_anchors_to_image(anchors, img_size)
    return anchors  # shape: [H*W*len(scales)*len(ratios), 4]

# apply_deltas_to_anchors函数的作用是对已有的anchor进行位置和大小的调整，把它们变成预测框（proposal）
def apply_deltas_to_anchors(anchors, deltas):  # anchors的行数是anchor的数量，列是anchor坐标，单个anchor的坐标的(x1,y1,x2,y2)
    # anchors作为超参数传入网络，需要提前知道骨干网络输出特征图上的一个点对应原图的哪块区域才能设置anchor
    # anchor个数的设置必须和RPN的输出相匹配也就是h*w*k个
    widths = anchors[:, 2] - anchors[:, 0]  # (x1, y1)是左上角，(x2, y2)是右下角
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths  # anchor中心x坐标，图像左上角为原点(0,0)
    ctr_y = anchors[:, 1] + 0.5 * heights  # anchor中心y坐标

    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]  # 每行delta是[dx,dy,dw,dh]
    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w = widths * torch.exp(dw)
    pred_h = heights * torch.exp(dh)

    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w  # 前两个是左上角
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w  # 后两个是右下角
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes  # (anchors, deltas, pred_boxes)的形状完全一致：

# Non-Maximum Suppression（非极大值抑制）
def nms(boxes, scores, iou_threshold=0.7):  # 正是由于nms筛选的原因，这个步骤不可导，所以Faster R-CNN要设计两个损失函数
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
# 返回的形状是一维张量 [num_keep]，长度 num_keep ≤ num_anchors。类型是torch.int64（因为列表里是整数）。内容是索引值

def clip_anchors_to_image(anchors, img_size):
    """
    anchors: Tensor [num_anchors, 4] (x1, y1, x2, y2)
    img_size: tuple (H, W)
    """
    H, W = img_size
    x1 = anchors[:, 0].clamp(min=0, max=W)
    y1 = anchors[:, 1].clamp(min=0, max=H)
    x2 = anchors[:, 2].clamp(min=0, max=W)
    y2 = anchors[:, 3].clamp(min=0, max=H)
    return torch.stack([x1, y1, x2, y2], dim=1)

# 把RPN的分类和回归预测转换为真实的候选框，并通过NMS筛选出最终用于Fast R-CNN的Proposals。
def generate_proposals(cls_logits, bbox_pred, anchors, img_size, post_nms_top_n=300, nms_thresh=0.7):
    # cls_logits: [B, k*2, H, W]  每个 anchor 前景/背景分类预测
    # bbox_pred: [B, k*4, H, W]   每个 anchor 回归预测(dx,dy,dw,dh)
    # anchors: [num_anchors, 4]   anchor 坐标 (x1,y1,x2,y2)
    # post_nms_top_n是最终保留的候选框数量，原始H*W*k个anchor，现在post_nms_top_n个
    B, _, H, W = cls_logits.shape
    cls_probs = F.softmax(cls_logits.permute(0, 2, 3, 1).reshape(B, -1, 2), dim=-1)[:, :, 1]
    # 最后一个维度是背景分数和前景分数[bg_score, fg_score]
    bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)

    proposals_batch = []
    for i in range(B):
        scores = cls_probs[i]  # num_anchors个数的一维张量，每个元素是对应的anchor是前景的概率
        deltas = bbox_pred[i]  # shape:[num_anchors, 4]，每行是对应anchor的回归预测(dx,dy,dw,dh)
        proposals = apply_deltas_to_anchors(anchors, deltas)
        proposals = clip_anchors_to_image(proposals, img_size)
        keep = nms(proposals, scores, nms_thresh)
        keep = keep[:post_nms_top_n]  # [:post_nms_top_n]控制最终保留下来的候选框数量，小于等于post_nms_top_n
        proposals_kept = proposals[keep]  # keep是索引，proposals当中只保留有索引的（长度 ≤ post_nms_top_n）
        batch_indices = torch.full((proposals_kept.shape[0], 1), i, device=proposals_kept.device)
        proposals_with_idx = torch.cat([batch_indices, proposals_kept], dim=1)  # 前面加上一列单个样本在batch中的索引，供后续roi_align使用
        proposals_batch.append(proposals_with_idx)
    return proposals_batch  # 最终输出的是一个列表，元素是张量，该批次中某个样本的框，形状是[≤ post_nms_top_n, 5]

def compute_iou(proposals, gt_boxes):
    """
    proposals: [N, 5] (batch_idx, x1, y1, x2, y2)
    gt_boxes: [M, 4] (x1, y1, x2, y2)   # 注意：这里假设只传入当前 batch 的 gt
    return: IoU matrix [N, M]
    """
    boxes = proposals[:, 1:5]   # 去掉 batch_idx，保留 (x1,y1,x2,y2)
    N = boxes.shape[0]
    M = gt_boxes.shape[0]
    # 计算 anchors 和 gt 的面积
    anchors_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # (N,)
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])  # (M,)
    # 准备存放结果
    iou = torch.zeros((N, M), device=boxes.device)
    # 遍历每个 gt，和所有 anchor 算 IoU
    for i in range(M):
        xx1 = torch.max(boxes[:, 0], gt_boxes[i, 0])  # (N,)
        yy1 = torch.max(boxes[:, 1], gt_boxes[i, 1])  # (N,)
        xx2 = torch.min(boxes[:, 2], gt_boxes[i, 2])  # (N,)
        yy2 = torch.min(boxes[:, 3], gt_boxes[i, 3])  # (N,)

        w = (xx2 - xx1).clamp(min=0)  # (N,)
        h = (yy2 - yy1).clamp(min=0)  # (N,)
        inter = w * h  # (N,)
        union = anchors_area + gt_area[i] - inter  # (N,)
        iou[:, i] = inter / union
    return iou

def bbox2delta(proposals, gt_boxes):  # 真实标签是在proposal的基础上偏移多少
    """
    proposals: [N, 5]  (batch_idx, x1,y1,x2,y2)
    gt_boxes: [N, 4]   matched gt
    return: [N, 4]     (tx,ty,tw,th)
    """
    boxes = proposals[:, 1:5]   # 只取坐标
    # proposal
    px = (boxes[:, 0] + boxes[:, 2]) * 0.5  # proposal的中心点横坐标
    py = (boxes[:, 1] + boxes[:, 3]) * 0.5  # proposal的中心点纵坐标
    pw = boxes[:, 2] - boxes[:, 0]  # proposal的宽度
    ph = boxes[:, 3] - boxes[:, 1]  # proposal的高度
    # gt
    gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5  # gt_boxes的中心点横坐标
    gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5  # gt_boxes的中心点纵坐标
    gw = gt_boxes[:, 2] - gt_boxes[:, 0]  # gt_boxes的宽度
    gh = gt_boxes[:, 3] - gt_boxes[:, 1]  # gt_boxes的高度

    tx = (gx - px) / pw
    ty = (gy - py) / ph
    tw = torch.log(gw / pw)
    th = torch.log(gh / ph)

    deltas = torch.stack([tx, ty, tw, th], dim=1)
    return deltas

def assign_proposals_to_gt(proposals, gt_boxes, gt_labels, iou_threshold=0.5):  # 为每个proposal匹配一个真实标签
    """
    proposals: [N, 5]     (batch_idx, x1,y1,x2,y2)
    gt_boxes: [M, 4]
    gt_labels: [M]
    return: labels [N], bbox_targets [N,4]
    """
    N = proposals.size(0)
    ious = compute_iou(proposals, gt_boxes)  # [N, M]
    max_iou, max_idx = ious.max(dim=1)       # [N]

    labels = torch.zeros(N, dtype=torch.long, device=proposals.device)
    bbox_targets = torch.zeros((N, 4), dtype=torch.float32, device=proposals.device)
    # 正样本
    pos_inds = max_iou >= iou_threshold  # [N]
    labels[pos_inds] = gt_labels[max_idx[pos_inds]]  # max_idx[pos_inds]形状：[pos_inds中True数量]
    bbox_targets[pos_inds] = bbox2delta(proposals[pos_inds], gt_boxes[max_idx[pos_inds]])

    return labels, bbox_targets

# Region of Interest Align
# ROIAlign 的作用：
# 在原图上确定了一个候选框（RoI），然后在backbone输出的特征图上，精确地取出对应区域的特征，并把它变成固定大小（比如 7×7）的特征图。
# 不是在原图上裁剪，而是在特征图上按比例“对齐”，每个特征图像素对应原图的一块区域（由 stride 决定）
# ROIAlign 相当于：
# 根据RoI的原图坐标 → 映射到特征图，在特征图上生成采样点（均匀划分成 H_out×W_out），用双线性插值取值 → 得到固定大小的特征
# ROIAlign 在 backbone 特征图上操作，而不是原图：
# (1) 利用下采样和卷积得到的语义特征。(2) 减少计算量和内存消耗。(3) 任意大小 RoI → 固定大小特征 → 方便后续分类/回归
def roi_align(features, rois, img_size, output_size=(7, 7)):
    """
    features: Tensor, shape: [B, C, H_f, W_f]
    rois: list, 元素是单个样本候选框构成的张量，张量的shape：[单个样本中框的数量, 5], 每行格式 [batch_idx, x1, y1, x2, y2]，坐标在 feature map 尺度下
    output_size: (H_out, W_out)
    img_size：原图大小
    返回: roi_features: list, 元素是单个样本映射回特征图构成的张量。
    """
    B, C, H_f, W_f = features.shape
    H_out, W_out = output_size
    roi_features = []
    H_img, W_img = img_size[0], img_size[1]  # 原图大小
    stride_h = H_img / H_f  # 特征图与原图的比例
    stride_w = W_img / W_f

    for j in range(B):  # 取同一批次的不同样本
        roi = rois[j]
        sample_roi_features = []
        for i in range(roi.shape[0]):  # 取同一样本中的不同框
            batch_idx = int(roi[i, 0].item())  # 提取batch_idx
            x1, y1, x2, y2 = roi[i, 1:].tolist()
            # 生成采样网格
            x1_f, x2_f, y1_f, y2_f = x1 / stride_w, x2 / stride_w, y1 / stride_h, y2 / stride_h
            grid_y = torch.linspace(y1_f, y2_f, H_out, device=features.device)  # 第一个数是起点，第二个数是终点，总共第三个数个数
            grid_x = torch.linspace(x1_f, x2_f, W_out, device=features.device)
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')  # x和y的形状一样，都是[H_out, W_out]
            # indexing='ij'，x一列一列排，y一行一行排。indexing='xy'，x一行一行排，y一列一列排
            # 归一化到 [-1,1]，grid_sample的要求
            grid_x_norm = (grid_x / (W_f - 1)) * 2 - 1
            grid_y_norm = (grid_y / (H_f - 1)) * 2 - 1
            grid = torch.stack((grid_x_norm, grid_y_norm), dim=-1)  # [H_out, W_out, 2]
            grid = grid.unsqueeze(0)  # [1, H_out, W_out, 2]

            # 双线性插值采样
            roi_feat = F.grid_sample(features[batch_idx : batch_idx + 1], grid, mode='bilinear', align_corners=True)
            # features维度[1, C, H, W]，grid维度[1, H_out, W_out, 2]，返回张量维度[1, C, H_out, W_out]
            # 如果输入是2D特征图 (形状 [B, C, H, W])，那么grid形状是[B, H_out, W_out, 2]
            # 如果输入是3D特征图 (形状 [B, C, D, H, W])，那么grid形状是[B, D_out, H_out, W_out, 3]，以此类推
            # 输出张量的H_out, W_out坐标的像素就是grid中间两个维度取相同数字后最后一个维度的[x,y]坐标对应的输入特征图像素的插值
            # grid的最后一个维度[x, y]：第一列是x坐标，对应宽度（W）方向，第二列是y坐标，对应高度（H）方向
            # 如果输入特征图是多通道的话是对每个通道的图都按grid采样
            # grid_sample的输入和输出特征图批次大小的通道数都相同，只是H和W变成和grid一个大小
            sample_roi_features.append(roi_feat)
        sample_roi_features = torch.cat(sample_roi_features)
        roi_features.append(sample_roi_features)

    return roi_features  # roi_features中一个元素tensor的形状 = [单个样本中框的数量, C, H_out, W_out]

# Fast R-CNN Head
class FastRCNNHead(nn.Module):  # Faster R-CNN的头部模块，把每个RoI映射到类别和边界框参数。
    def __init__(self, in_channels, num_classes, output_size):  # num_classes表示目标检测任务中所有类别的数量
        super().__init__()
        self.fc1 = nn.Linear(in_channels * output_size[0] * output_size[1], 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes + 1)  # 每个ROI属于每个类别的logits（未经过 softmax 的分数）
        self.bbox_pred = nn.Linear(1024, num_classes * 4)  # 每个ROI对每个类别的边界框预测

    def forward(self, x):
        x = x.flatten(start_dim=1)  # start_dim参数用于指定从哪个维度开始进行展平操作，将从该维度开始的所有后续维度合并为一个维度。
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        cls_score = self.cls_score(x)  # 分类头
        bbox_pred = self.bbox_pred(x)  # 回归头
        return cls_score, bbox_pred  # cls_score.shape: [单个样本中框的数量, num_classes], bbox_pred.shape: [单个样本中框的数量, num_classes*4]

# Faster R-CNN
class FasterRCNN(nn.Module):
    def __init__(self, img_size, num_classes, output_size=(7, 7)):
        super().__init__()
        self.img_size = img_size  # tuple
        self.backbone = Backbone()
        self.output_size = output_size
        self.rpn = RPN(self.backbone.out_channels)
        self.head = FastRCNNHead(self.backbone.out_channels, num_classes, output_size)

    def forward(self, images):
        feature_map = self.backbone(images)
        H_f, W_f = feature_map.shape[2], feature_map.shape[3]
        stride_h = self.img_size[0] / H_f  # 原图高 / 特征图高
        stride_w = self.img_size[1] / W_f  # 原图宽 / 特征图宽
        anchors = generate_anchors((H_f, W_f), (stride_h, stride_w), img_size=self.img_size, device=feature_map.device)
        rpn_cls, rpn_bbox = self.rpn(feature_map)
        proposals = generate_proposals(rpn_cls, rpn_bbox, anchors, self.img_size)

        rois = roi_align(feature_map, proposals, self.img_size, output_size=self.output_size)
        # roi_align把任意大小的候选框特征，统一为固定output_size大小像素，用于后续全连接分类和回归，同时尽量保留原始空间信息。

        cls_score, bbox_pred = [], []
        for i in range(len(rois)):
            one_cls_score, one_bbox_pred = self.head(rois[i])
            cls_score.append(one_cls_score), bbox_pred.append(one_bbox_pred)

        return cls_score, bbox_pred, proposals, rpn_cls, rpn_bbox, anchors

# assign_anchors的作用是：
# (1) 为每个 anchor 打标签（label）
# 前景（positive，1）：这个 anchor 和某个 gt box 的 IoU 足够大（≥0.7），或者它是该 gt box 的最佳匹配（IoU 最大的 anchor）。
# 背景（negative，0）：这个 anchor 与所有 gt box 的 IoU ≤0.3。
# 忽略（-1）：落在 0.3～0.7 之间的 anchor，不参与训练。
# (2) 计算回归目标（bbox_targets）
# 对所有正样本anchor，计算它们相对于匹配gt box的偏移量 (dx, dy, dw, dh)。这些就是训练时回归损失的监督信号。
# (3) 保证每个GT至少有一个正样本anchor，避免某些目标没有anchor负责
def assign_anchors(anchors, gt_boxes, positive_iou=0.7, negative_iou=0.3):  # 输入都是单个样本
    """
    anchors: Tensor [num_anchors, 4] (x1, y1, x2, y2)
    gt_boxes: Tensor [num_gt, 4] (x1, y1, x2, y2)
    positive_iou: IoU >= threshold -> positive
    negative_iou: IoU <= threshold -> negative
    Returns:
        labels: Tensor [num_anchors], 1=fg, 0=bg, -1=ignore
        bbox_targets: Tensor [num_anchors, 4], regression targets
    """
    num_anchors = anchors.shape[0]
    num_gt = gt_boxes.shape[0]
    # 初始化 labels 和 bbox_targets
    labels = torch.full((num_anchors,), -1, dtype=torch.long, device=anchors.device)
    bbox_targets = torch.zeros((num_anchors, 4), dtype=torch.float, device=anchors.device)
    if num_gt == 0:
        # 如果没有 gt box，全部都是背景
        labels[:] = 0
        return labels, bbox_targets
    # 计算 IoU
    # 扩展后计算每个 anchor 对每个 gt 的 IoU
    anchors_area = (anchors[:,2]-anchors[:,0]) * (anchors[:,3]-anchors[:,1])
    gt_area = (gt_boxes[:,2]-gt_boxes[:,0]) * (gt_boxes[:,3]-gt_boxes[:,1])
    # anchors_area和gt_area都是一维张量，长度分别为anchor数量和ground truth框数量。
    iou = torch.zeros((num_anchors, num_gt), device=anchors.device)
    # 后面会计算每个anchor与每个gt_box的交并比（IoU），并存入对应位置iou[i, j]
    for i in range(num_gt):
        # 计算每个anchor与第i个ground truth box的交集区域的坐标。
        xx1 = torch.max(anchors[:,0], gt_boxes[i,0])
        yy1 = torch.max(anchors[:,1], gt_boxes[i,1])
        xx2 = torch.min(anchors[:,2], gt_boxes[i,2])
        yy2 = torch.min(anchors[:,3], gt_boxes[i,3])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        union = anchors_area + gt_area[i] - inter
        iou[:,i] = inter / union  # 一列一列放数

    max_iou, max_idx = iou.max(dim=1)  # iou.max(dim=1)的作用是沿着每个anchor对应所有gt的IoU找最大值。
    # dim=1是每行元素的最大值，所以max_iou是每个anchor与tg对应的最大的iou值，max_idx是该gt的索引
    # 标记正负样本
    labels[max_iou >= positive_iou] = 1
    labels[max_iou <= negative_iou] = 0
    # 确保每个 gt 至少有一个 anchor
    # 对每个 gt，找 IoU 最大的 anchor
    best_anchor_for_gt = iou.argmax(dim=0)
    # iou.argmax(dim=0)沿着行方向（dim=0）找每一列的最大值索引（竖着找）。每一列对应一个gt box。
    # 返回值best_anchor_for_gt的长度是num_gt，每个元素是该gt box与哪个anchor的IoU最大。
    labels[best_anchor_for_gt] = 1
    # 前面通常已经通过IoU阈值规则（>0.7前景，<0.3背景，中间忽略）给anchor打标签。
    # 但有可能某些gt box没有任何anchor的IoU≥ 0.7，这样它就没anchor对应了，训练时会丢掉这个目标。

    # 计算 bbox_targets 对应回归偏移 (dx,dy,dw,dh)
    fg_idx = torch.where(labels == 1)[0]
    # torch.where返回一个元组（tuple），包含所有满足条件（condition为True）的元素的索引坐标。
    # 这里fg_idx是个一维张量tensor([])，torch.where返回的是一个元组 (tensor([]),)，其中唯一的元素是一个张量，包含了所有值为1的元素在原张量中的索引位置
    if fg_idx.numel() > 0:  # .numel()方法用于计算并返回张量中元素的总数量（即张量所有维度大小的乘积）（number of elements）
        a = anchors[fg_idx]  # a是被选出的anchor，形状是[被选出的anchor数, 4]
        g = gt_boxes[max_idx[fg_idx]]  # 被选出的anchor对应的gt，g的型状是 [被选出的anchor数, 4]
        # max_idx[fg_idx]一定包含所有gt的索引（因为每个gt至少有一个anchor和它绑定）。
        # 如果IoU ≥ 0.7的anchor有多个，它们会一起绑定同一个gt。如果没有IoU ≥ 0.7，那就只会有一个anchor和该gt绑定。所以g当中可能有重复

        wa = a[:,2] - a[:,0]  # anchor宽度(width)
        ha = a[:,3] - a[:,1]  # anchor高度(height)
        xa = a[:,0] + 0.5 * wa  # anchor中心点x坐标
        ya = a[:,1] + 0.5 * ha  # anchor中心点y坐标

        wg = g[:,2] - g[:,0]  # gt宽度(width)
        hg = g[:,3] - g[:,1]  # gt高度(height)
        xg = g[:,0] + 0.5 * wg  # gt中心点x坐标
        yg = g[:,1] + 0.5 * hg  # gt中心点y坐标

        # 下面四行就是论文中的公式(2)，前面代码中叫deltas
        dx = (xg - xa) / wa  # x方向的偏移（归一化）
        dy = (yg - ya) / ha  # y方向的偏移（归一化）
        dw = torch.log(wg / wa)  # 宽度的缩放比例（对数形式）
        dh = torch.log(hg / ha)  # 高度的缩放比例（对数形式）
        # 以上四个都是一维张量，每个张量元素个数是被选出的anchor数

        # 把前景anchors对应的回归目标(dx, dy, dw, dh)存到bbox_targets里，供训练时计算回归损失用。
        bbox_targets[fg_idx] = torch.stack([dx, dy, dw, dh], dim=1)  # 堆叠后的形状是[被选出的anchor数, 4]
        # 只有被选为前景的anchor的目标回归值被赋值为[dx, dy, dw, dh]，所有未被选中的anchor（背景或忽略的）保持为[0,0,0,0]

    return labels, bbox_targets  # labels就是为每一个anchor打标签，所以它的长度和anchors一样

def rpn_loss(rpn_cls_logits, rpn_bbox_pred, anchors, gt_boxes, positive_iou=0.7, negative_iou=0.3, lambda_reg=1.0):
    """
    rpn_cls_logits: [B, 2*k, H, W]  前景/背景分类
    rpn_bbox_pred: [B, 4*k, H, W]  回归预测
    anchors: [num_anchors, 4] 最后一维是每个 anchor 的坐标 (x1, y1, x2, y2)
    gt_boxes: list of [num_gt,4] 每个 batch 样本的 ground truth
    ground truth boxes（GT boxes）就是训练数据中标注的真实物体边界框（bounding box）
    """
    B = rpn_cls_logits.shape[0]
    all_cls_loss = 0
    all_reg_loss = 0

    for i in range(B):
        # 1. 为每个 anchor 匹配 gt_boxes，生成 labels 和 bbox_targets
        labels, bbox_targets = assign_anchors(anchors, gt_boxes[i], positive_iou, negative_iou)
        # labels: [num_anchors], 1=fg, 0=bg, -1=ignore
        # bbox_targets: [num_anchors, 4]

        # 2. 分类损失
        cls_logits = rpn_cls_logits[i].permute(1,2,0).reshape(-1,2)
        cls_loss = F.cross_entropy(cls_logits, labels, ignore_index=-1)  # k*h*w就是anchor的总数量

        # 3. 回归损失（只对前景 anchor）
        fg_mask = labels == 1
        if fg_mask.sum() > 0:
            reg_loss = F.smooth_l1_loss(rpn_bbox_pred[i].permute(1,2,0).reshape(-1,4)[fg_mask], bbox_targets[fg_mask])
            #平滑L1损失，中间平方，两边线性
        else:
            reg_loss = torch.tensor(0., device=rpn_cls_logits.device)

        all_cls_loss += cls_loss
        all_reg_loss += reg_loss
    # 平均 batch
    return (all_cls_loss / B) + lambda_reg * (all_reg_loss / B)

def fast_rcnn_loss(class_logits, bbox_pred, gt_labels, gt_boxes, num_classes, lambda_reg=1.0):
    """
    Args:
        class_logits: list of [N, K+1]  每个 proposal 的分类预测
        bbox_pred: list of [N, K*4]     每个 proposal 的回归预测
        gt_labels: list of [N]          proposal 对应的真实分类标签 (0=背景, 1..K=前景类)
        gt_boxes: list of [N, 4]        proposal 对应的回归目标 (仅前景有效)
        num_classes: int, 前景类别数 K
        beta: float, smooth_l1 的 beta (Huber 损失参数)
    Returns:
        classification_loss: 标量
        bbox_reg_loss: 标量
    """
    all_cls_loss = []
    all_reg_loss = []
    # 每次迭代都会取出当前batch内单个样本的4个Tensor
    for logits, bbox_pred_i, labels, targets in zip(class_logits, bbox_pred, gt_labels, gt_boxes):  # zip返回的是元组
        # 分类损失
        cls_loss = F.cross_entropy(logits, labels, reduction="mean")  # 输出是个数
        # 回归损失：只对前景 (label > 0) 有效
        pos_inds = torch.nonzero(labels > 0).squeeze(1)
        # torch.nonzero()用于查找张量中非零元素位置的函数，返回的是一个包含非零元素索引的二维张量。
        if pos_inds.numel() > 0:
            # 取出前景类别对应的 bbox 回归分支
            # bbox_pred: [N, K*4]，需要取对应类别的回归结果
            N, _ = bbox_pred_i.shape
            bbox_pred_i = bbox_pred_i.view(N, num_classes, 4)  # [N, K, 4]

            # labels 是 [N]，取 pos_inds 的类别
            cls_inds = labels[pos_inds] - 1  # 前景类别索引 (0..K-1)
            pred_reg = bbox_pred_i[pos_inds, cls_inds]  # [num_pos, 4]

            reg_loss = F.smooth_l1_loss(pred_reg, targets[pos_inds], reduction="mean")
        else:
            reg_loss = bbox_pred_i.sum() * 0.0  # 保持梯度连通
        all_cls_loss.append(cls_loss)
        all_reg_loss.append(reg_loss)
    classification_loss = torch.stack(all_cls_loss).mean()
    bbox_reg_loss = torch.stack(all_reg_loss).mean()
    return classification_loss + lambda_reg * bbox_reg_loss

class DetectionDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        """
        img_dir: 图像文件夹路径
        label_dir: 标签文件夹路径
        transform: 图像增强（torchvision.transforms）
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]  # 获取所有图片名（不带后缀的文件名用于匹配标签）

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
        # 读取图像
        img = Image.open(img_path).convert("RGB")
        # 读取标签
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                # 自动处理文件的打开和关闭，即使操作中出现错误也能保证文件正确关闭，将打开的文件对象赋值给变量f，通过f可以读取文件内容
                for line in f.readlines():
                    # 每一行内容作为列表中的一个元素（包含每行末尾的换行符 \n），一次性读取文件所有内容到内存，适合处理中小型文件
                    cls, x1, y1, x2, y2 = line.strip().split()
                    # str.strip(chars)方法，参数chars(可选)指定要移除的字符，不填则默认移除首尾空白字符。返回处理后的新字符串(原字符串不变)。
                    # split() 方法，按照指定分隔符分割字符串，返回一个包含分割后子串的列表。返回分割后的子串组成的列表。
                    # 默认的分割行为是：1.以任意数量的空白字符作为分隔符（包括空格、制表符 \t、换行符 \n 等）
                    # 2.会自动忽略字符串开头和结尾的空白字符。3.连续的多个空白字符会被视为一个分隔符
                    cls = int(cls)
                    x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])  # 这行实际作用就是把这四个数变成浮点数
                    # map()用于将一个函数应用到可迭代对象（如列表、元组等）的每个元素上，并返回一个包含结果的迭代器（map 对象）
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls)
        # 转为tensor
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        if self.transform:
            img = self.transform(img)
        return img, target

# 自定义 collate_fn：一个 batch 里就是列表形式
def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, dim=0)  # [B, C, H, W]
    return imgs, list(targets)  # targets 保持 list，因为每张图的框数可能不同
# batch = [(img1, target1),(img2, target2),(img3, target3),(img4, target4)]
# batch是一个列表，长度等于batch_size，每个元素是一个(img, target)的二元组。
# * 是序列解包运算符。*batch会把batch这个列表解开，等价于：
# (img1, target1), (img2, target2), (img3, target3), (img4, target4)
# zip()函数返回的可迭代对象（zip 对象）中的每个元素一定是元组。
# zip会“按位置配对”。*batch解开后是好几个(img, target)，zip 会把“所有的第1个元素”放一起，把“所有的第2个元素”放一起。
# zip返回一个可迭代对象，第一个输出是第一个元素放在一起，第二个元素是第二个元素放在一起，以此类推。
# [(img1, img2, img3, img4), (target1, target2, target3, target4)]

# 使用示例
# transform = transforms.Compose([transforms.ToTensor()])
# dataset = DetectionDataset(
#     img_dir="C:/python/pythonProject/deep_learning/object_detection/Dataset/Train/images",
#     label_dir="C:/python/pythonProject/deep_learning/object_detection/Dataset/Train/labels",
#     transform=transform
# )
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
# # collate_fn参数用于自定义如何将多个样本（从数据集取出的单个数据项）组合成一个批次
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# epochs = 2
# num_classes=6
# model = FasterRCNN(img_size=(640, 640), num_classes=num_classes).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# for epoch in range(epochs):
#     model.train()
#
#     for images, targets in dataloader:
#         images = images.to(device)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # 字典构成的列表
#         optimizer.zero_grad()
#         cls_score, bbox_pred, proposals, rpn_cls, rpn_bbox, anchors = model(images)
#         labels, bbox_targets = [], []
#         for proposal, target in zip(proposals, targets):
#             label, bbox_target = assign_proposals_to_gt(proposal, target["boxes"], target["labels"]+1)
#             labels.append(label)
#             bbox_targets.append(bbox_target)
#         loss_rpn = rpn_loss(rpn_cls, rpn_bbox, anchors, [target["boxes"] for target in targets])
#         loss_rcnn = fast_rcnn_loss(cls_score, bbox_pred, labels, bbox_targets, num_classes=num_classes)
#         loss = loss_rpn + loss_rcnn
#         loss.backward()
#         optimizer.step()
#         if epoch % 1 == 0:
#             print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

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
            # patches.Rectangle是Matplotlib库中用于创建矩形图形的类，常用于在图像或图表上绘制矩形区域（如目标检测中的边界框）。
            # (x1, y1)：矩形的起始点坐标，是一个元组。
            # w：矩形的宽度（沿 x 轴方向的长度）。
            # h：矩形的高度（沿 y 轴方向的长度）。
            # linewidth（可选）：矩形边框的线宽，默认值为 1。
            # edgecolor（可选）：矩形边框的颜色，可传入颜色名称（如 'r' 表示红色）或 RGB 值。
            # facecolor（可选）：矩形内部的填充颜色，'none' 表示不填充（透明）。
            ax.add_patch(rect)
            # .add_patch将创建好的图形元素（如矩形、圆形、多边形等）添加到图表中，使其在绘图时显示。
            # 类别和置信度
            if class_names is not None:
                label = f"{class_names[int(cls)]}: {score:.2f}"
            else:
                label = f"{int(cls)}: {score:.2f}"
            ax.text(x1, y1 - 5, label, color="yellow", fontsize=10, backgroundcolor="black")
            # .text在Matplotlib图表的指定位置添加文本标签，通常用于标注图像中的特定区域（如目标检测中的类别标签）。
            # x1, y1-2：文本的起始坐标（x 轴和 y 轴位置），这里y1-2表示文本位置在y1上方2个单位（避免与下方图形重叠）。
            # f"cls {labels[i].item()}"：要显示的文本内容，通过f-string格式化，其中labels[i].item()通常是从张量中提取的类别编号（如目标检测中的类别 ID），最终显示类似cls 3这样的文本。
            # color="yellow"：文本颜色为黄色，在多数背景下对比度较高，易于阅读。
            # fontsize=12：文本字体大小为12。
            # backgroundcolor="black"：文本背景为黑色，进一步增强与图像的对比度，避免被图像内容干扰。
    plt.axis("off")
    plt.show()

def inference(model_backbone, model_rpn, model_head, image_path, device='cpu', num_classes=6, score_thresh=0.5, nms_thresh=0.3):
    model_backbone.eval()
    model_rpn.eval()
    model_head.eval()
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(image).unsqueeze(0).to(device)  # [1,3,H,W]
    img_size = (640, 640)

    with torch.no_grad():
        # 1. Backbone 特征
        features = model_backbone(img)  # [1, C, H/32, W/32]
        B, C, H_f, W_f = features.shape
        # 2. RPN proposals
        cls_logits, bbox_pred = model_rpn(features)
        anchors = generate_anchors((H_f, W_f), strides=(img_size[0] // H_f, img_size[1] // W_f), img_size=img_size, device=device)
        proposals = generate_proposals(cls_logits, bbox_pred, anchors, img_size)  # [N, 5], (batch_idx, x1, y1, x2, y2)
        # 3. ROIAlign
        roi_feats = roi_align(features, proposals, img_size, output_size=(7, 7))  # list
        roi_feats = torch.cat(roi_feats, dim=0)  # [N, C, 7, 7]，torch.cat不会增加维度，只是去掉了list
        # 4. Fast R-CNN Head
        cls_logits, bbox_deltas = model_head(roi_feats)  # [N, num_classes], [N, num_classes*4]
        cls_probs = F.softmax(cls_logits, dim=-1)       # 转概率
        scores, labels = cls_probs.max(dim=1)           # (N,), (N,) 每个ROI的预测类别和分数
        # 5. 应用 bbox_deltas
        deltas = bbox_deltas.view(-1, num_classes, 4)   # [N, num_classes, 4]
        refined_boxes = []
        refined_scores = []
        refined_labels = []

        for i, prop in enumerate(torch.cat(proposals, dim=0)[:, 1:5]):  # 去掉 batch_idx
            cls = labels[i].item()  # 每个ROI的预测类别，这两个都是数
            score = scores[i].item()  # 每个ROI的预测分数
            if cls == 0 or score < score_thresh:  # 背景 or 低置信度跳过
                continue
            delta = deltas[i, cls]
            box = apply_deltas_to_anchors(prop.unsqueeze(0), delta.unsqueeze(0))  # [1, 4]
            refined_boxes.append(box)
            refined_scores.append(torch.tensor([score], device=device))
            refined_labels.append(torch.tensor([cls], device=device))

        if len(refined_boxes) == 0:
            return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)

        refined_boxes = torch.cat(refined_boxes, dim=0)  # [M, 4]，M是经过筛选（背景去掉 + score ≥ 阈值）的候选框数量
        refined_scores = torch.cat(refined_scores, dim=0)  # [M]
        refined_labels = torch.cat(refined_labels, dim=0)  # [M]

        # 6. NMS 按类别进行
        final_boxes, final_scores, final_labels = [], [], []
        for cls_id in range(1, num_classes):
            cls_mask = refined_labels == cls_id
            if cls_mask.sum() == 0:
                continue
            boxes = refined_boxes[cls_mask]
            cls_scores = refined_scores[cls_mask]
            keep = nms(boxes, cls_scores, iou_threshold=nms_thresh)
            final_boxes.append(boxes[keep])
            final_scores.append(cls_scores[keep])
            final_labels.append(torch.full_like(cls_scores[keep], cls_id, dtype=torch.long))

        if len(final_boxes) == 0:
            return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)

        final_boxes = torch.cat(final_boxes, dim=0)  # [K, 4]，K是NMS之后保留下来的目标框数量
        final_scores = torch.cat(final_scores, dim=0)  # [K]
        final_labels = torch.cat(final_labels, dim=0)  # [K]

        return final_boxes, final_scores, final_labels

model_path = "C:/python/pythonProject/deep_learning/faster_r_cnn.pth"
img_path = "C:/python/pythonProject/deep_learning/object_detection/Dataset/Test/images"
img_names = os.listdir(img_path)
full_model = FasterRCNN(img_size=(640, 640), num_classes=6)
full_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
for img_name in img_names:
    img_test = os.path.join(img_path, img_name)
    final_boxes, final_scores, final_labels = inference(full_model.backbone, full_model.rpn, full_model.head, img_test)  # 这里名字必须和FasterRCNN里定义的一样
    dets = torch.cat([final_boxes, final_scores.unsqueeze(1), final_labels.unsqueeze(1)], dim=1)
    visualize_detections(img_test, dets, class_names=["apple", "banana", "grape", "orange", "ananas", "watermelon"])
