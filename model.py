import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, inpt, hidn, oupt):
        super(NeuralNet, self).__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()
        
        self.l1 = nn.Linear(inpt, hidn)
        self.l2 = nn.Linear(hidn, hidn)
        self.l3 = nn.Linear(hidn, hidn)
        self.l4 = nn.Linear(hidn, oupt)

    def forward(self, x):
        out = self.l1(x)
        out = self.sigmoid(out)
        out = self.l2(out)
        out = self.sigmoid(out)
        out = self.l3(out)
        out = self.dropout(out)
        out = self.sigmoid(out)
        out = self.l4(out)

        return out

# NerualNet

print("model.py ", end="")
