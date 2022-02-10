import json
from tokenizer import tokenize, bagOfWords
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import NeuralNet
import alive_progress as ap

## Train Set ##
with open('./txbs/textbook.json', 'r', encoding='utf-8') as f:
    trainData = json.load(f)
# with

allWords, tags, xy = [], [], []

for intent in trainData['intents']:
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
        tk_sentence = tokenize(pattern)
        allWords.extend(tk_sentence)
        xy.append((tk_sentence, intent['tag']))
# for

allWords = sorted(allWords)
tags = sorted(tags)

x = []
y = []

for (pattern_sentence, tag) in xy:
    bag = bagOfWords(pattern_sentence, allWords)
    x.append(bag)
    y.append(tags.index(tag))  # CrossEntropyLoss
# for
x = np.array(x)
y = np.array(y)
y = torch.from_numpy(y)
y = y.type(torch.long)

class myTrainSet(Dataset):
    def __init__(self):
        self.n_samples = len(x)
        self.x_data = x
        self.y_data = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
# class

mySet = myTrainSet()
trainData = DataLoader(dataset=mySet, batch_size=1, shuffle=True, num_workers=0)


## hpyer prameter tunning ##
inptSize = x.shape[1]
hidnSize = int(len(allWords) * 1.2)
ouptSize = len(tags)
model = NeuralNet(inptSize, hidnSize, ouptSize)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device=device)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
iterations = 1000

## iterations  ##
bar = ap.alive_it(range(iterations))  # progress bar
for epoch in bar:
    for (words, labels) in trainData:
        predictY = model.forward(words)
        loss = criterion(predictY, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
    # for
# for

data = {"modelState":model.state_dict(),
        "inptSize":inptSize,
        "hidnSize":hidnSize,
        "ouptSize":ouptSize,
        "allWords":allWords,
        "tags": tags}
torch.save(data, './pars/data.pth')

print(f"train is over, loss: {loss.item():.4f}\neverything completed, file saved.")
