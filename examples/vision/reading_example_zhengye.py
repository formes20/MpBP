import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_LiRPA.bound_general_zhengye import BoundedModule
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations_zhengye import PerturbationLpNorm


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
# Optionally, load the pretrained weights.
checkpoint = torch.load(os.path.join(os.path.dirname(__file__), "pretrain/example_net.pth"), map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

reading_input = torch.Tensor([[0., 1.]])
if torch.cuda.is_available():
    reading_input = reading_input.cuda()
    model = model.cuda()

# wrap model with auto_LiRPA
# The second parameter is for constructing the trace of the computational graph, and its content is not important.
lirpa_model = BoundedModule(model, torch.empty_like(reading_input), device=reading_input.device)
print('Running on', reading_input.device)

# Compute bounds using LiRPA given a perturbation
eps = 2.
norm = float("inf")
ptb = PerturbationLpNorm(norm=norm, eps=eps)
reading_input = BoundedTensor(reading_input, ptb)


# Compute bounds for final output
# for method in ['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)', 'CROWN-Optimized (alpha-CROWN)']:
# for method in ['forward', 'IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)']:
for method in ['backward (CROWN)']:
    print("Bounding method:", method)
    lb, ub = lirpa_model.compute_bounds(x=(reading_input,), method=method.split()[0])
    print(lb, ub)

