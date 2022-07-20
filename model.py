import torch
import torch.nn as nn
from nltk.stem.porter import PorterStemmer
from konlpy.tag import Kkma
import numpy as np

""" model """
class NeuralNet(nn.Module):
    def __init__(self, inpt, hidn, oupt):
        super(NeuralNet, self).__init__()
        self.dropout = nn.Dropout(p=0.4)
        self.act = nn.ReLU()
        
        self.l1 = nn.Linear(inpt, hidn)
        self.l2 = nn.Linear(hidn, hidn)
        self.l3 = nn.Linear(hidn, hidn)
        self.l4 = nn.Linear(hidn, oupt)

    def forward(self, x):
        out = self.l1(x)
        out = self.act(out)
        out = self.l2(out)
        out = self.act(out)
        out = self.l3(out)
        out = self.dropout(out)
        out = self.act(out)
        out = self.l4(out)

        return out
# NerualNet

""" tokenizer """
nonsense = {'!','?','~','^','.','%','$','^','_'}

stemmer = PorterStemmer()
kkma = Kkma()
def tokenize(sentence):
    sentence = kkma.nouns(sentence)

    for i, letter in enumerate(sentence):
        if letter in nonsense: del sentence[i]

    return [stemmer.stem(word.lower()) for word in sentence]
# tokenize()

def bag_of_words(tked_sentence, all_words):
    bag = np.zeros(len(all_words), dtype=np.float32)

    for i, w in enumerate(all_words):
        if w in tked_sentence: bag[i] = 1.0

    return bag
# bag_of_words()

print("model.py... OK")
