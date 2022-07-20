import sys
import yaml
import torch
from model import NeuralNet bagOfWords, tokenize

try:
    path = sys.argv[-1]
    data = torch.load(path)
except: exit()

try:
    path = sys.argv[-2]
    with open(path, mode='r', encoding='utf-8') as yaml_buffer: intents = yaml.safe_load(yaml_buffer)
except: exit()

inpt_parameters = data["inpt"]
hidn_parameters = data["hidn"]
oupt_parameters = data["oupt"]
dictionary = data["dict"]
tags = data["tags"]
state = data["model_state"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(inpt_parameters, hidn_parameters, oupt_parameters).to(device)
model.load_state_dict(state)
model.eval()

""" interaction """
while 1:
    cmd = input("me: ")
    if cmd == "exit": exit()
    sentence = tokenize(cmd)
    bag = bag_of_words(cmd, dictionary)
    bag = torch.from_numpy( bag.reshape(1, bag.shape[0]) )
    output = model(bag)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75: print(f"{predicted.item()}")
    else: print(f"hiana: idk")
# while
