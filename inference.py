from utils.tokenization import Tokenizer
from model import Transformer

corpus = 'Это тестовый текст для текста для трансформера текстового!!!'
tokenizer = Tokenizer(corpus=corpus)

params = {
    'd_model': 512,
    'num_heads': 6,
    'n': 6,
    'vocab_size': len(tokenizer.d)
}

model = Transformer(**params)
token = tokenizer('Тестовый текст')
print(model(token))