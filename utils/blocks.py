import torch
import torch.nn as nn
from utils.layers import FeedForwardNetwork, MultiHeadAttention, AddNorm

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model, d_k=d_model, masked=False)
        self.ffn = FeedForwardNetwork(in_f1=d_model, out_f1=2048, in_f2=2048, out_f2=d_model)
        self.add_norm = AddNorm(d_model=d_model)
    
    def forward(self, x: torch.Tensor):
        prev_x = x 
        x = self.mha(x)
        x = self.add_norm(x, prev_x)

        prev_x = x
        x = self.ffn(x)
        x = self.add_norm(x, prev_x)

        return x

class Encoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, n: int):
        super().__init__()
        self.block = nn.Sequential(*[EncoderBlock(d_model=d_model, num_heads=num_heads) for _ in range(n)])

    def forward(self, x: torch.Tensor):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model, d_k=d_model, masked=False)
        self.add_norm = AddNorm(d_model=d_model)
        self.ffn = FeedForwardNetwork(in_f1=d_model, out_f1=2048, in_f2=2048, out_f2=d_model)
    
    def forward(self, x: torch.Tensor, encoder_ouput: torch.Tensor):
        prev_x = x
        x = self.mha(x)
        x = self.add_norm(x, prev_x)

        prev_x = x
        x = self.mha(x=x, Q=encoder_ouput, K=encoder_ouput)
        x = self.add_norm(x, prev_x)

        prev_x = x
        x = self.ffn(x)
        x = self.add_norm(x, prev_x)

        return x

class Decoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, n: int):
        super().__init__()
        self.block = nn.Sequential(*[DecoderBlock(d_model=d_model, num_heads=num_heads) for _ in range(n)])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor):
        for block in self.block:
            x = block(x, encoder_output)
        return x