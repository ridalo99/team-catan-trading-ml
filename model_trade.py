import torch
import torch.nn as nn

class TradeNet(nn.Module):
    def __init__(self, input_size: int,num_partners: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out_offer = nn.Linear(64, 5)
        self.out_request = nn.Linear(64, 5)
        self.out_partner = nn.Linear(64, num_partners)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        offer = self.out_offer(x)
        request = self.out_request(x)
        partner = self.out_partner(x)
        return offer, request, partner