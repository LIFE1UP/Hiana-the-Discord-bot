import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import konlpy.tag as knl

kkma = knl.Kkma()
stemmer = PorterStemmer()
no_meaning = ['!','?','~','^','.']

def tokenize(sentence):
    sentence = nltk.word_tokenize(sentence)
    for i, letter in enumerate(sentence):
        if letter in no_meaning:
            del sentence[i]
    # for
    
    return [stemmer.stem(word.lower()) for word in sentence]
# def

def kor_tokenize(sentence):
    return kkma.nouns(sentence)
# def

def bagOfWords(tked_sentence, all_words):
    bag = np.zeros(len(all_words), dtype=np.float32)
    for i, w in enumerate(all_words):
        if w in tked_sentence:
            bag[i] = 1.0
    # for

    return bag
# def

print("tokenizer.py ", end="")
