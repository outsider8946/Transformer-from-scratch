import torch
import torch.nn as nn
from utils.blocks import Encoder, Decoder
from utils.input_embeddings import InputEmbeddings

class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, n: int, vocab_size:int):
        super().__init__()
        self.input = InputEmbeddings(d_model=d_model, vocab_size=vocab_size)
        self.encoder = Encoder(d_model=d_model, num_heads=num_heads, n=n)
        self.decoder = Decoder(d_model=d_model, num_heads=num_heads, n=n)
        self.linear = nn.Linear(in_features=d_model, out_features=vocab_size)
        self.act = nn.Softmax(dim=-1)
    
    def forward(self, input: torch.Tensor):
        if input.dim() == 2:
            input = input.unsqueeze(0)
        
        x = self.input(input)
        print('Input: ', x)
        encoder_output = self.encoder(x)
        print('Encoder: ', encoder_output)
        decoder_output = self.decoder(x, encoder_output)
        print('Decoder: ', decoder_output)
        linear_output = self.linear(decoder_output)
        print('Linear: ', linear_output)
        print('Linear shape: ', linear_output.shape)

        return self.act(linear_output)
