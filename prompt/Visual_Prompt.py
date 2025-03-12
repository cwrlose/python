# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch
from torch import nn
from collections import OrderedDict
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

"""
截断正态分布初始化是一种权重初始化方法，它从正态分布中采样值，
但只保留位于某个特定范围内的值，通常是均值（mean）两侧的几个标准差（std）之内的值。
这有助于在训练开始时防止梯度爆炸或消失，并且可以改善模型的训练动态
"""
def trunc_normal_(x, mean=0., std=1.):
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)

"""
类利用 Transformer 架构对视频帧进行时间维度上的聚合，
通过捕获帧之间的复杂关系来增强视频表示。它使用分类标记和位置嵌入来提供额外的上下文信息，
并利用 Transformer 编码器的强大能力来学习视频的时空特征。
这种方法在 ActionCLIP 模型中用于提高动作识别的性能。
"""
class TAggregate(nn.Module):
    def __init__(self, clip_length=None, embed_dim=2048, n_layers=6):
        super(TAggregate, self).__init__()
        self.clip_length = clip_length
        drop_rate = 0.
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=nn.LayerNorm(
            embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, clip_length + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        with torch.no_grad():
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        nvids = x.shape[0]

        cls_tokens = self.cls_token.expand(nvids, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x.transpose_(1, 0)
        o = self.transformer_enc(x)

        return o[0]

"""
新的vit
"""
class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))


class visual_prompt(nn.Module):
    def __init__(self, sim_head, clip_state_dict, T):
        super().__init__()
        self.sim_header = sim_head
        self.T = T
        assert sim_head in ["meanP", "LSTM", "Transf", "Conv_1D", "Transf_cls"]

        if self.sim_header == "LSTM" or self.sim_header == "Transf" or self.sim_header == "Transf_cls" or self.sim_header == "Conv_1D" :
            #text_projection 层的作用是将这个特征向量投影到一个与图像特征向量相同的嵌入空间中
            #embed_dim 表示的是文本特征向量在投影后的嵌入维度，
            embed_dim = clip_state_dict["text_projection"].shape[1]
        #这个键对应的是位置嵌入矩阵。在处理序列数据（如文本）时，
        # 为了让模型能够感知到每个元素在序列中的位置信息，通常会使用位置嵌入。

            context_length = clip_state_dict["positional_embedding"].shape[0]
            #这是词嵌入层的权重矩阵。在 CLIP 模型中，输入的文本首先会被分词成一个个标记（token），
            #然后词嵌入层会将每个标记转换为一个连续的向量表示。词嵌入层的权重矩阵存储了所有可能标记的向量表示。
            vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
           #：这个键对应的是最后一个层归一化（Layer Normalization，简称 LN）层的权重参数。层归一化是一种常用的归一化技术，用于在神经网络的每一层中对输入进行归一化处理，以加速模型的训练和提高模型的稳定性。在 CLIP 的文本编码器中，
           #通常是基于 Transformer 架构，ln_final 是最后一个层归一化层
            transformer_width = clip_state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64



            transformer_layers = len(
                set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

            self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)#词嵌入
        if self.sim_header == "Transf" :
            self.transformer = TemporalTransformer(width=embed_dim, layers=6, heads=transformer_heads)
            print('layer=6')
        if self.sim_header == "LSTM":
            self.lstm_visual = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim,
                                       batch_first=True, bidirectional=False, num_layers=1)

        self.apply(self.init_weights)

        if self.sim_header == "Transf_cls":
            self.transformer = TAggregate(clip_length=self.T, embed_dim=embed_dim, n_layers=6)

        if self.sim_header == 'Conv_1D' :
            self.shift = nn.Conv1d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False)
            weight = torch.zeros(embed_dim, 1, 3)
            weight[:embed_dim // 4, 0, 0] = 1.0
            weight[embed_dim // 4:embed_dim // 4 + embed_dim // 2, 0, 1] = 1.0
            weight[-embed_dim // 4:, 0, 2] = 1.0
            self.shift.weight = nn.Parameter(weight)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        b, t, c = x.size()
        x = x.contiguous()#连续
        if self.sim_header == "meanP":
            pass
        elif self.sim_header == 'Conv_1D':
            x_original = x
            x = x.view(-1, c, t)
            x = self.shift(x.float())
            x = x.permute(0, 2, 1)
            x = x.type(x_original.dtype) + x_original

        elif self.sim_header == "Transf":
            x_original = x
            seq_length = t
            """
            1. position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
功能：创建一个从 0 到 seq_length - 1 的一维张量 position_ids，用于表示序列中每个位置的索引。
参数解释：
seq_length：表示序列的长度，即输入序列中元素的数量。
dtype=torch.long：指定张量的数据类型为 64 位整数（torch.long），因为位置索引通常是整数。
device=x.device：将创建的张量放置在与输入张量 x 相同的设备（如 CPU 或 GPU）上，以确保后续计算的兼容性。
2. position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
position_ids.unsqueeze(0)：
功能：在 position_ids 张量的第 0 维（即最前面）插入一个维度，将原本的一维张量转换为二维张量。例如，如果 position_ids 原本的形状是 (seq_length,)，经过 unsqueeze(0) 操作后，形状变为 (1, seq_length)。
.expand(x.size(0), -1)：
功能：将 position_ids 张量在第 0 维上进行扩展，使其第 0 维的大小与输入张量 x 的第 0 维大小相同。-1 表示在该维度上保持原来的大小不变。例如，如果 x 的形状是 (batch_size, ...)，那么扩展后的 position_ids 形状将变为 (batch_size, seq_length)，这样每个批次中的序列都有对应的位置索引。
3. frame_position_embeddings = self.frame_position_embeddings(position_ids)
功能：将扩展后的 position_ids 输入到 self.frame_position_embeddings 模块中，得到每个位置对应的位置嵌入（position embeddings）。
self.frame_position_embeddings：这是一个 torch.nn.Embedding 层，用于将整数索引（即位置索引）映射到固定维度的向量表示。例如，假设 self.frame_position_embeddings 的嵌入维度为 embed_dim，那么 frame_position_embeddings 的形状将是 (batch_size, seq_length, embed_dim)，表示每个批次中的每个位置都有一个对应的 embed_dim 维的位置嵌入向量。
            """
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            x = x + frame_position_embeddings

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = x.type(x_original.dtype) + x_original

        elif self.sim_header == "LSTM":
            x_original = x
            x, _ = self.lstm_visual(x.float())
            self.lstm_visual.flatten_parameters()
            x = torch.cat((x, x_original[:, x.size(1):, ...].contiguous()), dim=1)
            x = x.type(x_original.dtype) + x_original
        elif self.sim_header == "Transf_cls":
            x_original = x
            return self.transformer(x).type(x_original.dtype)

        else:
            raise ValueError('Unknown optimizer: {}'.format(self.sim_header))
        return x.mean(dim=1, keepdim=False)
