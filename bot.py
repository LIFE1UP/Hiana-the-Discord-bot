import sys
import json
import torch
from model import NeuralNet
from tokenizer import bagOfWords, tokenize
import random
import os
import discord
import reponse

# Pytorch
try:
    with open('./txbs/textbook.json', 'r', encoding='utf-8') as json_file: intents = json.load(json_file)
    data = torch.load('./pars/data.pth')
    print("system got loaded!")
except:
    print("something went wrong.")
    exit()
# try

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

# Discord
client = discord.Client( activity=discord.Game(name="Sonic") )

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

'''
@client.command()
async def play(ctx, url : str):
    voiceChanel = discord.utils.get(ctx.guild.vocie_channels, name='General')
    voice = discord.utils.get(cllient.voice_clients, guild=ctx.guild)
    if not voice.is_connected(): await voicheChannel.connect()
# play()

@client.command()
async def leave(ctx):
    voice = discord.utils.get(cllient.voice_clients, guild=ctx.guild)
    if voice.is_connected: await voice.disconnect()
# leave()
'''

print("hosting...")
client.run("OTc0NzUyNTI5ODM5NjkzODc1.GsOrL9.oVyuVdV5lAux-3kN35CRK9Y0gQURt-99sRuW3E")
