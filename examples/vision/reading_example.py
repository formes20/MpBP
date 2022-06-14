import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# from multipath_bp import BoundedModule, BoundedTensor
# from multipath_bp.perturbations import PerturbationLpNorm

from multipath_bp.bound_general_multipath import BoundedModule
from multipath_bp import BoundedTensor
from multipath_bp.perturbations_multipath import PerturbationLpNorm


class exampleNet(nn.Module):
    def __init__(self):
        super(exampleNet, self).__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
        self.linear3 = torch.nn.Linear(2, 2)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


model = exampleNet()
# Optionally, load the pretrained weights.
checkpoint = torch.load(os.path.join(os.path.dirname(__file__), "pretrain/abstracmp_net.pth"), map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

reading_input = torch.Tensor([[0., 0.]])
if torch.cuda.is_available():
    reading_input = reading_input.cuda()
    model = model.cuda()

# wrap model with multipath_bp
# The second parameter is for constructing the trace of the computational graph, and its content is not important.
lirpa_model = BoundedModule(model, torch.empty_like(reading_input), device=reading_input.device)
print('Running on', reading_input.device)

# Compute bounds using LiRPA given a perturbation
eps = 1.
norm = float("inf")
ptb = PerturbationLpNorm(norm=norm, eps=eps)
reading_input = BoundedTensor(reading_input, ptb)


# Compute bounds for final output
# for method in ['IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)', 'CROWN-Optimized (alpha-CROWN)']:
for method in ['forward', 'IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)']:
    print("Bounding method:", method)
    lb, ub = lirpa_model.compute_bounds(x=(reading_input,), method=method.split()[0])
    print(lb)
    print(ub)

