import sys
import os
import yaml
import torch
from model import NeuralNet, bag_of_words, tokenize
import discord
import modes

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

""" discord """
client = discord.Client( activity=discord.Game(name="Being Alone") )
@client.event
async def on_ready(): print('logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user: return
    if not message.content.startswith('히아나'): return

    case = bagOfWords( tokenize(message.content[1:]), allWords )
    case = torch.from_numpy( case.reshape(1, case.shape[0]) )

    output = model(case)
    _, prediction = torch.max(output, dim=1)
    tag = tags[prediction.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][prediction.item()]

    if prob < 0.75: await message.channel.send("뭔 말하는지 모르겠어")
    else: cmd(tag)

    return
# on_message()

print("hosting...")
client.run("OTc0NzUyNTI5ODM5NjkzODc1.GsOrL9.oVyuVdV5lAux-3kN35CRK9Y0gQURt-99sRuW3E")
