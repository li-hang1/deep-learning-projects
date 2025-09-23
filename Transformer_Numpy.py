import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # 如果没有keepdims并且没指定axis的话，那么np.max()就是一个数
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

class LayerNorm:
    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones((1, 1, dim))
        self.beta = np.zeros((1, 1, dim))

    # __call__方法使得类的实例可以像函数一样被调用。
    def __call__(self, x):  # 输入x的形状是(batch_size, seq_len, word_embedding_dim)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * norm + self.beta
        # (batch_size, seq_len, word_embedding_dim)与(1, 1, dim)相加就是前者最内层向量每个都与后者相加

class MultiHeadSelfAttention:
    def __init__(self, dim, num_heads=4):
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # 为每个头独立初始化 W_q, W_k, W_v
        self.W_q = [np.random.randn(dim, self.head_dim) / np.sqrt(dim) for _ in range(num_heads)]
        self.W_k = [np.random.randn(dim, self.head_dim) / np.sqrt(dim) for _ in range(num_heads)]
        self.W_v = [np.random.randn(dim, self.head_dim) / np.sqrt(dim) for _ in range(num_heads)]
        # 输出线性映射
        self.W_0 = np.random.randn(dim, dim) / np.sqrt(dim)  # 除以np.sqrt(dim)是一种初始化策略

    def __call__(self, x):  # x: (batch_size, seq_len, dim)
        head_outputs = []
        for h in range(self.num_heads):
            Q = x @ self.W_q[h]  # (batch_size, seq_len, head_dim)
            K = x @ self.W_k[h]
            V = x @ self.W_v[h]
            # Attention scores
            scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.head_dim)  # (batch_size, seq_len, seq_len)
            # Attention output
            attn_output = softmax(scores, axis=-1) @ V  # (batch_size, seq_len, head_dim)
            head_outputs.append(attn_output)
        # 拼接所有头的输出
        concat = np.concatenate(head_outputs, axis=-1)  # (batch_size, seq_len, dim)
        # np.concatenate((a1, a2, ..., an), axis=0)沿指定的轴将多个数组连接成一个更大的数组,
        # (a1, a2, ..., an)是待拼接的数组序列(元组或列表形式),所有数组必须维度相同
        return concat @ self.W_0  # 输出线性变换

class FeedForwardNet:
    def __init__(self, dim, hidden_dim):
        self.W1 = np.random.randn(dim, hidden_dim) / np.sqrt(dim)  # 除以np.sqrt(dim)是一种初始化策略
        self.b1 = np.zeros((1, 1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros((1, 1, dim))

    def __call__(self, x):
        x = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return x @ self.W2 + self.b2

def positional_encoding(max_seq_len, dim):  # 预先准备好最大行数的位置编码矩阵，后面根据输入x的行数的不同选取前x.shape[1]行即可
    i = np.arange(max_seq_len)[:, None]  # None的作用是新增加一个维度，None在最后面就是在最后面增加维度
    k = np.arange(dim)[None, :]  # None在最前面就是在最前面增加维度，i是行，k是列
    angle_rates = 1 / np.power(10000, (2 * (k // 2)) / dim)  # (k // 2)) 产生的效果是 0,0,1,1,2,2,...
    angle_rads = i * angle_rates
    pos_encoding_matrix = np.zeros((max_seq_len, dim))
    pos_encoding_matrix[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # [:, 0::2]作用是选取所有行（第一维），并在列方向（第二维）上从索引 0 开始，每隔一个元素选取一次
    pos_encoding_matrix[:, 1::2] = np.cos(angle_rads[:, 1::2])
    # [:, 1::2]作用是选取所有行（第一维），并在列方向（第二维）上从索引 1 开始，每隔一个元素选取一次
    return pos_encoding_matrix[None, :, :]  # shape (1, max_seq_len, dim)

class TransformerEncoderLayer:
    def __init__(self, dim, ff_hidden_dim):
        self.attn = MultiHeadSelfAttention(dim)
        self.ln1 = LayerNorm(dim)
        self.ff = FeedForwardNet(dim, ff_hidden_dim)
        self.ln2 = LayerNorm(dim)

    def __call__(self, x):
        # 自注意力 + 残差连接 + LayerNorm
        attn_out = self.attn(x)
        x = self.ln1(x + attn_out)
        # 前馈网络 + 残差连接 + LayerNorm
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x

class TransformerEncoder:
    def __init__(self, num_layers, dim, ff_hidden_dim, max_seq_len):
        self.layers = [TransformerEncoderLayer(dim, ff_hidden_dim) for _ in range(num_layers)]
        self.pos_encoding = positional_encoding(max_seq_len, dim)

    def __call__(self, x):  # 输入的x要求一个批次中的词索引序列长度必须相同即seq_len相同
        x = x + self.pos_encoding[:, :x.shape[1], :]  # x.shape = (batch_size, seq_len, dim)
        for layer in self.layers:
            x = layer(x)
        return x

def generate_subsequent_mask(seq_len):  # shape: (seq_len, seq_len)，下三角为0，上三角为-inf，防止看未来
    mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)
    # np.triu(m, k=0)返回一个数组的上三角部分，其余部分用零填充。k>0：主对角线右上方的第 k 条对角线及以上元素。
    return mask

class MaskedMultiHeadSelfAttention:
    def __init__(self, dim, num_heads=9):
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # 每个头独立的权重
        self.W_q = [np.random.randn(dim, self.head_dim) / np.sqrt(dim) for _ in range(num_heads)]
        self.W_k = [np.random.randn(dim, self.head_dim) / np.sqrt(dim) for _ in range(num_heads)]
        self.W_v = [np.random.randn(dim, self.head_dim) / np.sqrt(dim) for _ in range(num_heads)]
        # 输出线性变换
        self.W_0 = np.random.randn(dim, dim) / np.sqrt(dim)

    def __call__(self, y):  # x: (batch_size, seq_len, dim)
        seq_len = y.shape[1]
        head_outputs = []
        mask = generate_subsequent_mask(seq_len)[None, :, :]  # (1, 1, seq_len, seq_len)
        for i in range(self.num_heads):
            Q = y @ self.W_q[i]  # (batch_size, seq_len, head_dim)
            K = y @ self.W_k[i]
            V = y @ self.W_v[i]
            # Scaled dot-product attention with mask
            scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.head_dim)  # (batch_size, seq_len, seq_len)
            weights = softmax(scores + mask, axis=-1)  # (batch_size, seq_len, seq_len)
            head_output = weights @ V  # (batch_size, seq_len, head_dim)
            head_outputs.append(head_output)
        # 拼接所有头的输出
        concat = np.concatenate(head_outputs, axis=-1)  # (batch_size, seq_len, dim)
        return concat @ self.W_0  # (batch_size, seq_len, dim)

class MultiHeadCrossAttention:
    def __init__(self, dim, num_heads=6):
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # 为每个头分别定义 W_q, W_k, W_v，都是 (dim, head_dim)
        self.W_q = [np.random.randn(dim, self.head_dim) / np.sqrt(dim) for _ in range(num_heads)]
        self.W_k = [np.random.randn(dim, self.head_dim) / np.sqrt(dim) for _ in range(num_heads)]
        self.W_v = [np.random.randn(dim, self.head_dim) / np.sqrt(dim) for _ in range(num_heads)]
        # 输出投影矩阵，统一为 (dim, dim)
        self.W_0 = np.random.randn(dim, dim) / np.sqrt(dim)

    def __call__(self, y_q, x_kv):  # x_q 和 x_kv 的 shape 都是 (batch_size, seq_len, dim)
        batch_size, seq_len_q, _ = y_q.shape
        _, seq_len_kv, _ = x_kv.shape
        head_outputs = []
        for i in range(self.num_heads):
            # 线性映射
            Q = y_q @ self.W_q[i]  # (batch_size, seq_len_q, head_dim)
            K = x_kv @ self.W_k[i]  # (batch_size, seq_len_kv, head_dim)
            V = x_kv @ self.W_v[i]  # (batch_size, seq_len_kv, head_dim)
            # 注意力权重计算
            scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.head_dim)  # (batch_size, seq_len_q, seq_len_kv)
            weights = softmax(scores, axis=-1)  # (batch_size, seq_len_q, seq_len_kv)
            # 加权求和
            attn = weights @ V  # (batch_size, seq_len_q, head_dim)
            head_outputs.append(attn)
        # 拼接所有头
        concat = np.concatenate(head_outputs, axis=-1)  # (batch_size, seq_len_q, dim)
        # 输出线性映射
        return concat @ self.W_0  # (batch_size, seq_len_q, dim)

class TransformerDecoderLayer:
    def __init__(self, dim, ff_hidden_dim):
        self.masked_attn = MaskedMultiHeadSelfAttention(dim)
        self.ln1 = LayerNorm(dim)
        self.cross_attn = MultiHeadCrossAttention(dim)
        self.ln2 = LayerNorm(dim)
        self.ff = FeedForwardNet(dim, ff_hidden_dim)
        self.ln3 = LayerNorm(dim)

    def __call__(self, y, enc_output):
        output = self.masked_attn(y)
        y = self.ln1(y + output)
        cross_out = self.cross_attn(y, enc_output)  # Encoder-Decoder Cross Attention
        y = self.ln2(y + cross_out)
        ff_out = self.ff(y)
        y = self.ln3(y + ff_out)
        return y

class TransformerDecoder:
    def __init__(self, num_layers, dim, ff_hidden_dim, max_seq_len):
        self.layers = [TransformerDecoderLayer(dim, ff_hidden_dim) for _ in range(num_layers)]
        self.pos_encoding = positional_encoding(max_seq_len, dim)

    def __call__(self, y, enc_output):
        y = y + self.pos_encoding[:, :y.shape[1], :]
        for layer in self.layers:
            y = layer(y, enc_output)
        return y

class Transformer:
    def __init__(self, num_layers_encoder, dim, ff_hidden_dim_encoder, max_seq_len_encoder, num_layers_decoder,
                 ff_hidden_dim_decoder, max_seq_len_decoder, vocab_size):
        self.encoder_layer = TransformerEncoder(num_layers_encoder, dim, ff_hidden_dim_encoder, max_seq_len_encoder)
        self.decoder_layer = TransformerDecoder(num_layers_decoder, dim, ff_hidden_dim_decoder, max_seq_len_decoder)

        self.W_out = np.random.randn(dim, vocab_size) / np.sqrt(dim)
        self.b_out = np.zeros((vocab_size,))

    def __call__(self, src, tgt):
        enc_output = self.encoder_layer(src)
        dec_output = self.decoder_layer(tgt, enc_output)
        logits = dec_output @ self.W_out + self.b_out
        probs = softmax(logits, axis=-1)
        return probs


model = Transformer(2, 36, 256, 15,
                    3, 128, 20, 5)

src = np.random.randn(2, 10, 36)  # 假设是词嵌入后的输入句子
tgt = np.random.randn(2, 15, 36)   # 假设是目标句子(实际中需向右移位)

output = model(src, tgt)

print(output.shape)
