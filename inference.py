import torch
from utils.tokenization import Tokenizer
from model import Transformer

corpus = 'Это тестовый текст для текста для трансформера текстового!!!'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = Tokenizer(corpus=corpus)

params = {
    'd_model': 512,
    'num_heads': 6,
    'n': 8,
    'vocab_size': len(tokenizer.d)
}

model = Transformer(**params).to(device)
print('Prameters: ', params)
token = tokenizer('Тестовый текст')
out = model(token)
print('Model shape: ', out.shape)
print(out)