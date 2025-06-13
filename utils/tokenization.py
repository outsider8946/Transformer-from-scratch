import nltk
import string
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pymorphy3

nltk.download('punkt_tab')
nltk.download('stopwords')

class Tokenizer():
    def __init__(self, corpus):
        self.sw = stopwords.words('russian') + list(string.punctuation)
        self.morph = pymorphy3.MorphAnalyzer()
        self.d = {}
        self._create_dict(corpus)
        
    def _preprocessing(self, sentence):
        words = word_tokenize(sentence.lower(), language='russian')
        words = [self.morph.parse(word)[0].normal_form for word in words]
        return [word for word in words if word not in self.sw]

    def _create_dict(self, corpus):
        self.d = {word:i for i, word in enumerate(set(self._preprocessing(corpus)))}


    def __call__(self, sentence):
        return torch.tensor([self.d[word] for word in self._preprocessing(sentence)],dtype=torch.int)