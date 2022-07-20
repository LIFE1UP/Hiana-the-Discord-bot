import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from model import NeuralNet, tokenize, bag_of_words
import alive_progress

""" load train set """
txb_path = "./txbs/korean.yaml"
with open(txb_path, 'r', encoding="utf-8") as yaml_buffer:
    train_set = yaml.safe_load(yaml_buffer)
# with

# load whole data
tags, dictionary, xy = [], [], []
for intent in train_set["intents"]:
    tags.append(intent["tag"])
    for pattern in intent["patterns"]:
        tked_pattern = tokenize(pattern)
        dictionary.extend(tked_pattern)
        xy.append( [tked_pattern, intent["tag"]] )
    # for
# for

dictionary = sorted(dictionary)
tags = sorted(tags)

data, target = [], []
for (x, y) in xy:
    bag = bag_of_words(x, dictionary)
    data.append(bag)
    target.append(tags.index(y))
# for

data = np.array(data)
target = (torch.from_numpy(np.array(target))).type(torch.long)

class defaultTrainSet(Dataset):
    def __init__(self):
        self.n_samples = len(data)
        self.data = data
        self.target = target
        self.data_size = self.data.shape[1]
        self.n_tags = len(tags)

    def __getitem__(self, index): return self.data[index], self.target[index]
    def __len__(self): return self.n_samples
# defaultTrainset

main_set = defaultTrainSet()
main_loader = DataLoader(dataset=main_set, batch_size=1, shuffle=True, num_workers=0)

del data, target
print("data loading... OK")

""" hyper parameter tunning """
iterations = 1000
inpt_parameters = main_set.data_size
hidn_parameters = int( len(dictionary) * 1.2 )
oupt_parameters = main_set.n_tags

device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
model = NeuralNet(inpt_parameters, hidn_parameters, oupt_parameters).to(device=device)
criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

print(f"Properties: inpt {inpt_parameters} hidn {hidn_parameters} oupt {oupt_parameters} CrossEntropyLoss and Adam")

""" tarining """
bar = alive_progress.alive_it(range(iterations))
for epoch in bar:
    for (x, y) in main_loader:
        loss = criterion(model.forward(x), y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    # for
# for

print(f"loss: {loss.item():.4f}")

""" saving """
properties = {
    "model_state" : model.state_dict(),
    "inpt" : inpt_parameters,
    "hidn" : hidn_parameters,
    "oupt" : oupt_parameters,
    "dict" : dictionary,
    "tags" : tags
} # properties

save_path = "./pars/korean.pth"
torch.save(properties, save_path)

print(f"result is saved in {save_path}")
