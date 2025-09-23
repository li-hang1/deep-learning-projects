import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import os

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# get_tokenizer(tokenizer, language)获取预定义的或自定义的分词函数，它返回一个函数，该函数接受一个字符串，并返回一个字符串列表（即token序列）
token_transform = {  # 建立分词器
    SRC_LANGUAGE: get_tokenizer('spacy', language='de_core_news_sm'),  # 'spacy'表示分词器类型
    TGT_LANGUAGE: get_tokenizer('spacy', language='en_core_web_sm')
}

def yield_tokens(data_iter, language):  # 对数据集中样本进行分词
    for src, tgt in data_iter:  # data_iter是一个包含样本句对的迭代器，language：'de' 或 'en'，表示我们要处理哪一种语言。
        yield token_transform[language](src if language == SRC_LANGUAGE else tgt)  # 返回将样本字符串进行分词后的字符串列表
# yield 的核心作用是：
# 暂停函数执行：当函数执行到 yield 语句时，会返回 yield 后面的值，并暂停函数的执行状态（包括变量值、执行位置等）
# 保留函数状态：下次通过生成器的 __next__() 方法或 next() 函数调用时，会从上次暂停的位置继续执行
# 生成迭代数据：生成器函数返回的是一个迭代器，可通过循环逐步获取 yield 产生的值，而不是一次性生成所有数据

def build_vocab():  # 本函数用于构建词到索引的映射
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # Multi30k(split, language_pair)返回的是一个迭代器，每次for都返回一个元组，即一对(src, tgt)句子，所有句子都是字符串(未分词、未编码)
    # 参数split输入数据集的子集，支持 'train', 'valid', 'test', language_pair输入语言对，例如 ('de', 'en')
    # Multi30k 返回的迭代器会按照 language_pair 规定的顺序提供双语句子对
    vocab_src = build_vocab_from_iterator(yield_tokens(train_iter, SRC_LANGUAGE), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    # build_vocab_from_iterator(iterator, specials=[])接受一个迭代器（每次产生一个 token 列表），然后为该迭代器输出的所有词构建一个索引
    # 参数iterator，类型为iterable，每次返回一个字符串列表（token序列）
    # 参数specials，类型为list[str]，词表中指定添加的特殊token，比如 <unk>, <pad> 等
    # build_vocab_from_iterator会扫描整个token列表，把token加入到一个可查找的Vocab对象中，支持token到索引、索引到token的互转
    # 词表大小就是输入迭代器中所有不重复词的总数量。这个函数的作用就是为词构建索引。它的工作流程大致是：
    # 接收一个迭代器（iterator），其中每个元素是一个token列表（如分词后的句子），统计迭代器中所有token的出现频率
    # 为每个唯一的token分配一个唯一的整数索引（通常按出现频率排序），返回构建好的Vocab对象，该对象包含了token到索引的映射关系
    # 返回值vocab_src是一个类似字典的东西，可以查词的索引，可以一次查多个词，可以接受单个字符串（返回一个整数索引）或字符串列表（返回整数索引列表）
    vocab_src.set_default_index(vocab_src['<unk>'])
    # vocab_src.set_default_index设置当查询一个词表中不存在的token（词）时，返回的默认索引

    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))  # Multi30k创建的是一次性的迭代器
    vocab_tgt = build_vocab_from_iterator(yield_tokens(train_iter, TGT_LANGUAGE), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    vocab_tgt.set_default_index(vocab_tgt['<unk>'])

    return vocab_src, vocab_tgt

def data_process(raw_iter, vocab_src, vocab_tgt): # 本函数作用是将原始文本数据(src, tgt)对转化成带有<bos>和<eos>的整数索引序列张量
    src_bos_idx, src_eos_idx = vocab_src['<bos>'], vocab_src['<eos>']
    tgt_bos_idx, tgt_eos_idx = vocab_tgt['<bos>'], vocab_tgt['<eos>']
    data = []
    for src, tgt in raw_iter:
        src_tensor = torch.tensor([src_bos_idx] + vocab_src(token_transform[SRC_LANGUAGE](src)) + [src_eos_idx], dtype=torch.long)
        tgt_tensor = torch.tensor([tgt_bos_idx] + vocab_tgt(token_transform[TGT_LANGUAGE](tgt)) + [tgt_eos_idx], dtype=torch.long)
        data.append((src_tensor, tgt_tensor))
    return data

def generate_batch(data_batch):  # transformer需要一个批次中的样本seq_len相同
    # 原本单个样本形状为[seq_len, dim]，拼成一个批次的话形状为[batch_size, seq_len, dim]，如果批次中每个样本长度不同那么无法拼接
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in data_batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=1)
    # torch.nn.utils.rnn.pad_sequence的输入是一个元素为张量的列表
    # torch.nn.utils.rnn.pad_sequence用于将长度不一的序列Tensors填充到统一长度，将一个批次中所有序列统一补齐到当前batch中最长序列的长度。
    # 虽然输入是一个列表，但是输出还是一个tensor，padding_value设置填充哪个值
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=1)
    return src_batch, tgt_batch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, maxlen=5000):
        super().__init__()
        pos_encoding_matrix = self.positional_encoding(maxlen, d_model)  # shape: (1, maxlen, d_model)
        pos_encoding_tensor = torch.tensor(pos_encoding_matrix, dtype=torch.float32)  # 转为 tensor
        self.register_buffer('pos_embedding', pos_encoding_tensor)
        # self.register_buffer(name, tensor)用于注册一个不作为模型参数（不会被优化器更新），但作为模型状态（比如位置编码、均值方差等）保存的张量。
        # name: 字符串，表示这个buffer的名字（之后可以通过 self.name 访问）。tensor: 要注册的张量。
        self.dropout = nn.Dropout(p=dropout)

    def positional_encoding(self, max_seq_len, dim):
        i = np.arange(max_seq_len)[:, None]
        k = np.arange(dim)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (k // 2)) / dim)
        angle_rads = i * angle_rates
        pos_encoding_matrix = np.zeros((max_seq_len, dim))
        pos_encoding_matrix[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_encoding_matrix[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return pos_encoding_matrix[None, :, :]  # shape: (1, max_seq_len, dim)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model) or (batch_size, seq_len, d_model)
        # 为兼容常见 Transformer 实现，这里假设输入为 (seq_len, batch_size, d_model)
        seq_len = x.size(0)
        pos = self.pos_embedding[:, :seq_len, :]  # (1, seq_len, d_model)
        # self.register_buffer('pos_embedding', pos_embedding)同时完成了self.pos_embedding = pos_embedding
        pos = pos.transpose(0, 1)  # (seq_len, 1, d_model)
        return self.dropout(x + pos)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_size=256, nhead=4, num_layers=3, dim_ff=512, dropout=0.1):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, emb_size)
        # nn.Embedding(num_embeddings, embedding_dim)的作用是将整数索引（即token ID）映射成固定长度的向量表示
        # num_embeddings词汇表的大小(即最大 token 索引 + 1)(加1因为索引从0开始)，embedding_dim是词向量维度d_model
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, emb_size)
        self.pos_encoder = PositionalEncoding(emb_size, dropout)

        self.transformer = nn.Transformer(d_model=emb_size, nhead=nhead,  # 参数nhead同时指定了三个注意力模块的头数，它们的头数都相同
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          dim_feedforward=dim_ff,
                                          dropout=dropout)  # nn.Transformer封装了整个编码器-解码器结构，但没有最后的Linear和Softmax
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask):
        src_emb = self.pos_encoder(self.src_embedding(src))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt))
        output = self.transformer(src=src_emb, tgt=tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=None,
                                  src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
        # src：源序列嵌入（必选）
        # tgt：目标序列嵌入（必选）
        # src_mask：源序列自注意力掩码（可选，默认 None）
        # tgt_mask：目标序列自注意力掩码（可选，默认 None）
        # memory_mask：编码器 - 解码器注意力掩码（可选，默认 None）
        # src_padding_mask：源序列填充掩码（可选，默认 None）
        # tgt_padding_mask：目标序列填充掩码（可选，默认 None）
        # memory_padding_mask：编码器输出（memory）的填充掩码（可选，默认 None）
        # self.transformer输入源序列X（encoder 输入）和目标序列Y（decoder 输入），输出最终的解码器输出(decoder output)。
        # 输入X和Y的形状要求都是(seq_len, batch_size, embedding_dim)
        # self.transformer的输出形状和Y相同，都是(tgt_seq_len, batch_size, embedding_dim)
        # 经最后的仿射函数，形状变为(seq_len, batch_size, tgt_vocab_size)
        return self.generator(output)

def generate_subsequent_mask(seq_len, device=None):  # MaskedMultiHeadSelfAttention中的mask
    mask = np.triu(np.ones((seq_len, seq_len)) * float('-inf'), k=1)
    mask = torch.from_numpy(mask).float()
    if device:
        mask = mask.to(device)
    return mask

def create_mask(src, tgt, pad_idx):
    src_seq_len = src.size(0)
    tgt_seq_len = tgt.size(0)

    tgt_mask = generate_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    # nn.Transformer中src_padding_mask和tgt_padding_mask的形状要求(batch_size, seq_len)，所以要转置
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src_language = 'de'
tgt_language = 'en'
pad_idx = 1
batch_size = 32
num_epochs = 300

model_path = 'model.pt'
vocab_src_path = 'vocab_src.pt'
vocab_tgt_path = 'vocab_tgt.pt'
if os.path.exists(vocab_src_path) and os.path.exists(vocab_tgt_path):
    vocab_src = torch.load(vocab_src_path)
    vocab_tgt = torch.load(vocab_tgt_path)
    print("Loaded vocabularies from disk.")
else:
    vocab_src, vocab_tgt = build_vocab()
    torch.save(vocab_src, vocab_src_path)
    torch.save(vocab_tgt, vocab_tgt_path)
    print("Built and saved vocabularies.")

model = Transformer(len(vocab_src), len(vocab_tgt)).to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Loaded model parameters from disk.")
else:
    print("Training new model.")

optimizer = optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    train_iter = Multi30k(split='train', language_pair=(src_language, tgt_language))
    train_data = data_process(train_iter, vocab_src, vocab_tgt)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)
    for src, tgt in train_loader:  # src和tgt形状分别是(src_seq_len, batch_size)和(tgt_seq_len, batch_size)
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:-1, :]  # 解码器的输入，去掉最后一词
        tgt_out = tgt[1:, :]  # 解码器的期望输出，去掉第一词

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_idx)

        logits = model(src, tgt_input, src_mask.to(device), tgt_mask.to(device),
                       src_padding_mask.to(device), tgt_padding_mask.to(device))

        optimizer.zero_grad()
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))  # .reshape(-1)作用多维将张量展平成一维张量
        # .reshape(-1, num)的作用是把张量重塑成一个二维矩阵，其中第二维的大小固定为num，而第一维的大小则由PyTorch依据张量的总元素数量自动计算得出。
        # (logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))一个矩阵对应一个向量，按照tgt_out元素的值取矩阵中对应行的第该值的位置的值代入交叉熵损失
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    torch.save(model.state_dict(), model_path)
    print(f"Model saved at epoch {epoch + 1}.")
    print(f"Epoch {epoch+1}: loss = {total_loss:.4f}")

def greedy_decode(model, src_sentence, vocab_src, vocab_tgt, max_len=50, device=device):
    model.eval()
    tokens = ["<bos>"] + token_transform[SRC_LANGUAGE](src_sentence.lower()) + ["<eos>"]
    src_tensor = torch.tensor(vocab_src(tokens), dtype=torch.long).unsqueeze(1).to(device)  # (src_len, 1)
    # .unsqueeze()在指定位置插入一个维度为 1 的新维度，从而改变张量的形状。
    src_mask = torch.zeros((src_tensor.size(0), src_tensor.size(0)), device=device).type(torch.bool)
    src_padding_mask = (src_tensor == pad_idx).transpose(0, 1)  # (1, src_len)
    # src_key_padding_mask要求的维度是(batch_size, src_len)，所以要转置
    memory = model.transformer.encoder(model.pos_encoder(model.src_embedding(src_tensor)),
                                       mask=src_mask,
                                       src_key_padding_mask=src_padding_mask)
    # model.transformer.encoder()是Transformer编码器(Encoder)部分的实现，接收一个输入序列的嵌入+位置编码，输出memory，供解码器使用。
    tgt_tokens = [vocab_tgt["<bos>"]]
    for i in range(max_len):  # 从<bos>开始逐步喂给decoder，最大喂max_len长度的词
        tgt_tensor = torch.tensor(tgt_tokens, dtype=torch.long).unsqueeze(1).to(device)  # (tgt_len, 1)
        tgt_mask = generate_subsequent_mask(tgt_tensor.size(0), device=device)
        tgt_padding_mask = (tgt_tensor == pad_idx).transpose(0, 1)
        output = model.transformer.decoder(
            model.pos_encoder(model.tgt_embedding(tgt_tensor)),
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )  # model.transformer.decoder()是Transformer模型中的解码器（Decoder）部分的实现
        # 接收目标序列(shifted)和encoder输出，通过masked self-attention和cross attention，生成下一个token的表示，用于训练或逐步生成序列。
        logits = model.generator(output)  # (tgt_len, 1, vocab_size)
        next_token = logits[-1, 0].argmax().item()
        # logits[-1, 0]是一个向量，形状为 (vocab_size,)，表示当前解码出的序列最后一个token位置的预测分数
        # 二维以上tensor，用逗号,分隔不同维度的索引，-1表示最后一个维度，-1是因为只选最后一个时间步的输出作为当前词的预测
        tgt_tokens.append(next_token)
        if next_token == vocab_tgt["<eos>"]:
            break
    return " ".join(vocab_tgt.lookup_tokens(tgt_tokens[1:-1]))  # [1:-1]用于去掉<bos>和<eos>
    # 字符串的join()方法用于将可迭代对象（如列表、元组、集合等）中的元素连接成一个字符串。用前面的字符串分隔
    # vocab_tgt.lookup_tokens是torchtext的Vocab对象中的一个方法，用于将token ID（整数）序列转换为对应的词（token）序列

# 示例推理
example_sentence = "Ein kleines Mädchen klettert in ein Spielhaus."
translated = greedy_decode(model, example_sentence, vocab_src, vocab_tgt)
print(f"DE: {example_sentence}\nEN: {translated}")