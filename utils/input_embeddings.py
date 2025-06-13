import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.PE = self._positional_encoding(d_model=d_model, vocab_size=vocab_size)
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, dtype=torch.float32)
    
    def _positional_encoding(self,d_model: int, vocab_size: int):
        PE = torch.zeros((vocab_size, d_model), dtype=torch.float32)
        n = 10_000

        for k in range(vocab_size):
            for i in range(d_model//2):
                value = torch.tensor(k/n**(2*i/d_model), dtype=torch.float32)
                PE[k, 2*i] = torch.sin(value)
                PE[k, 2*i+1] = torch.cos(value)
        
        return PE
    
    def forward(self, tokens: torch.Tensor):
        if tokens.dim() != 2:
            tokens = tokens.unsqueeze(0)
            
        return self.embeddings(tokens) + self.PE[:tokens.size(1)]