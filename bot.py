import sys
import json
import torch
from model import NeuralNet
from tokenizer import bagOfWords, tokenize
import random
import os
import discord
from response import response

"""
not!on: this system learns context of what it see, then responses with constant lines
"""

## loading arguments ##
try:
    with open('./txbs/textbook.json', 'r', encoding='utf-8') as json_file:
        intents = json.load(json_file)
    data = torch.load('./pars/data.pth')
    #tagger = nl.Kkma()
    print(f"system got loaded!")
except:
    print("something went wrong.")
    exit()

# loading a parameter of model
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

# dicsord
print("preparing the bot")

client = discord.Client()

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    # if
    if message.content == "#help":
        await message.channel.send("help: <content>")
        return
    # if

    if message.content.startswith('#'):
        sentence = tokenize(message.content[1:])
        case = bagOfWords(sentence, allWords)
        case = case.reshape(1, case.shape[0])
        case = torch.from_numpy(case)
        output = model(case)  # prediction
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]  # answer
        probs = torch.softmax(output, dim=1)  # for reliability
        prob = probs[0][predicted.item()]

        if prob < 0.65:
            await message.channel.send("I don't know what you're saying!")
            return
        # if
        for intent in intents['intents']:
            if tag == intent['tag']:
                await message.channel.send(str(random.choice(intent['responses'])))
                res = response(tag)
                
                if res == None:
                    return
                else:
                    await message.channel.send(str(res))  # every fn works as string
                    return
                # if
            else:
                pass
            # if
        # for

print("hosting...")
client.run("TYPE YOUR API KEY!")
