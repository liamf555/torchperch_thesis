import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    
    def __init__(self):
        super(QNetwork, self).__init__()
        
        self.input_layer = nn.Linear(12,512)
        self.hidden_layer_1 = nn.Linear(512,512)
        self.hidden_layer_2 = nn.Linear(512,512)
        self.hidden_layer_3 = nn.Linear(512,512)
        self.output_layer = nn.Linear(512,49)
        # Cast to double...
        self.double()

    def forward(self, x):
        x = F.relu( self.input_layer(x) )
        x = F.relu( self.hidden_layer_1(x) )
        x = F.relu( self.hidden_layer_2(x) )
        x = F.relu( self.hidden_layer_3(x) )
        return self.output_layer(x)
