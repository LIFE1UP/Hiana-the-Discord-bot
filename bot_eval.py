import sys
import json
import torch
from model import NeuralNet
from tokenizer import bagOfWords, tokenize
import random

print("\n\nright form of command: python bot_eval.py <json.file path> <pth.file path>")

try:
    fileName = sys.argv[-1]
    data = torch.load(fileName)
    print(f"parameters: {fileName}")
except:
    print("need paremeter path arguments... \nexit.")
    exit()

try:
    fileName = sys.argv[-2]
    with open(fileName, mode='r', encoding='utf-8') as json_data:
        intents = json.load(json_data)
    print(f"textbook: {fileName}")
except:
    print("need textbook path arguemnts... \nexit.")
# try

# load data
inptSize = data["inptSize"]
hidnSize = data["hidnSize"]
ouptSize = data["ouptSize"]
allWords = data['allWords']
tags = data['tags']
modelState = data["modelState"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(inptSize, hidnSize, ouptSize).to(device)
model.load_state_dict(modelState)
model.eval()

# context classification
while 1:
    sentence = input("me: ")
    if sentence == "exit":
        exit()

    #sentence = tagger.nouns(sentence)
    sentence = tokenize(sentence)
    case = bagOfWords(sentence, allWords)
    case = case.reshape(1, case.shape[0])
    case = torch.from_numpy(case)

    output = model(case)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.65:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"hiana: {tag}")
            # if
        # for
    else:
        print(f"hiana: idk")
# while
