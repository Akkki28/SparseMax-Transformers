import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from Sparsemax import Sparsemax


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

    sparsemax = Sparsemax(dim=-1)
    attention = sparsemax(attn_logits)

    values = torch.matmul(attention, v)
    return values, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        q = self.w_q(query)  
        k = self.w_k(key)
        v = self.w_v(value)
        
        q = q.view(batch_size, -1, self.nhead, self.d_k).transpose(1,2)
        k = k.view(batch_size, -1, self.nhead, self.d_k).transpose(1,2)
        v = v.view(batch_size, -1, self.nhead, self.d_k).transpose(1,2)

        if mask is not None:
            
            mask = mask.unsqueeze(1)
        
        attn_output, attention = scaled_dot_product(q, k, v, mask)
        
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc(attn_output)
        output = self.dropout(output)
        return output, attention
    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = F.relu

    def forward(self, src, src_mask=None):
        
        attn_output, attention = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + attn_output)
        
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout(ff_output))
        return src, attention

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src, mask=None):
        output = src
        attentions = []
        for layer in self.layers:
            output, attn = layer(output, mask)
            attentions.append(attn)
        return output, attentions
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    

class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, nhead,
                 dim_feedforward=2048, dropout=0.1, num_classes=10):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder = TransformerEncoder(num_layers, d_model, nhead, dim_feedforward, dropout)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, src, mask=None):
        src = self.embedding(src) * math.sqrt(src.size(-1))
        src = self.pos_encoder(src)
        encoder_output, attentions = self.encoder(src, mask)
        
        pooled = encoder_output.mean(dim=1)
        output = self.fc_out(pooled)
        return output, attentions
