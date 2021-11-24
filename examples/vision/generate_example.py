import torch
import torch.nn as nn
import torch.nn.functional as F


class exampleNet(nn.Module):
    def __init__(self):
        super(exampleNet, self).__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
        self.linear3 = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x 


model = exampleNet()
# print(model)


def generate_example(model):
    sd = model.state_dict()
    with torch.no_grad():
        sd['linear1.weight'][0][0] = 2.0
        sd['linear1.weight'][0][1] = 1.0
        sd['linear1.weight'][1][0] = -3.0
        sd['linear1.weight'][1][1] = 4.0
        sd['linear1.bias'][0] = 0
        sd['linear1.bias'][1] = 0

        sd['linear2.weight'][0][0] = 4.0
        sd['linear2.weight'][0][1] = -2.0
        sd['linear2.weight'][1][0] = 2.0
        sd['linear2.weight'][1][1] = 1.0
        sd['linear2.bias'][0] = 0
        sd['linear2.bias'][1] = 0

        sd['linear3.weight'][0][0] = -2.0
        sd['linear3.weight'][0][1] = 1.0
        sd['linear3.bias'][0] = 0

    torch.save(model.state_dict(), './pretrain/example_net.pth')


if __name__ == '__main__':
    generate_example(model)
