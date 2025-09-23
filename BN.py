import numpy as np

class BatchNorm1d:
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        # momentum的作用是在推理的指数滑动平均中，控制“新数据”和“旧数据”在平均值中的权重比例
        # PyTorch中默认momentum是0.1，保证统计量随训练慢慢更新，但不会因单个batch波动太大而不稳定。
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        # 可学习参数
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        # 推理（inference）阶段的BatchNorm把训练阶段积累下来的样本均值和样本方差，当作整个数据分布的“真实”均值和方差来用。
        # 推理时是假设测试数据分布和训练数据分布相同，所以直接用训练的统计值。
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x, training=True):
        if training:
            # 训练阶段：使用当前 batch 的均值和方差
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            # 更新 running 统计量，指数滑动平均，每次都会“记住一点新统计值” + “保留大部分旧统计值”
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * batch_var
            # Python 类的实例对象会保留它的属性值，所以self.running_mean和self.running_var会在多次调用forward()时持续存在并被更新

            x_hat = (x - batch_mean) / np.sqrt(batch_var + self.eps)  # 归一化
        else:
            # 推理阶段：使用 running 均值和方差
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        # 仿射变换（缩放+平移）
        out = self.gamma * x_hat + self.beta
        return out

# 初始化 BatchNorm
bn = BatchNorm1d(num_features=3)
print("==== 训练阶段 ====")
for step in range(5):  # 模拟 5 个训练 step
    x_batch = np.random.randn(4, 3) * 2 + 5  # 模拟新 batch
    out = bn.forward(x_batch, training=True)
    print(f"Step {step+1}: running_mean = {bn.running_mean}, running_var = {bn.running_var}")

# 推理阶段：使用 running_mean / var
print("\n==== 推理阶段 ====")
x_test = np.array([[5.0, 5.0, 5.0]])
output_infer = bn.forward(x_test, training=False)
print("推理输出：", output_infer)


class BatchNorm2d:
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.num_features = num_features  # num_features就是通道数
        self.momentum = momentum
        self.eps = eps
        # 可学习参数：每个通道一个gamma和beta
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        # 推理时使用的均值和方差（running estimates）
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x, training=True):  # x shape: (N, C, H, W)
        if training:
            # 训练阶段：对每个通道计算均值和方差，平均维度是batch, height, width
            batch_mean = np.mean(x, axis=(0, 2, 3))
            # 假设有一个矩阵，元素也是矩阵，axis=(0, 2, 3)的意思是对所有行按列求和除以总元素数量，最后得到一个行向量
            batch_var = np.var(x, axis=(0, 2, 3))  # (C,)
            # 更新 running 统计量，指数滑动平均
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            x_hat = (x - batch_mean[None, :, None, None]) / np.sqrt(batch_var[None, :, None, None] + self.eps)
        else:
            # 推理阶段：使用 running_mean 和 running_var
            x_hat = (x - self.running_mean[None, :, None, None]) / np.sqrt(
                self.running_var[None, :, None, None] + self.eps)
        out = self.gamma[None, :, None, None] * x_hat + self.beta[None, :, None, None]
        return out

bn2 = BatchNorm2d(num_features=3)
print("\n==== 训练阶段 ====")
for step in range(5):
    x_batch = np.random.randn(4, 3, 2, 2) * 2 + 5
    out = bn2.forward(x_batch, training=True)
    print(f"Step {step + 1}: running_mean = {bn2.running_mean}, running_var = {bn2.running_var}")
print("\n==== 推理阶段 ====")
x_test = np.array([[[[5.0, 5.0],
                     [5.0, 5.0]],
                    [[5.0, 5.0],
                     [5.0, 5.0]],
                    [[5.0, 5.0],
                     [5.0, 5.0]]
                    ]])
output_infer = bn2.forward(x_test, training=False)
print("推理输出：", output_infer)