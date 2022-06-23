"""
A simple example for bounding neural network outputs under input perturbations.

This example serves as a skeleton for robustness verification of neural networks.
"""
import os
import torch
import torch.nn as nn
import torchvision

from multipath_bp.bound_general import BoundedModule
from multipath_bp import BoundedTensor
from multipath_bp.perturbations import PerturbationLpNorm

# from multipath_bp.bound_general_multipath import BoundedModule
# from multipath_bp import BoundedTensor
# from multipath_bp.perturbations_multipath import PerturbationLpNorm

### Step 1: Define computational graph

# Models defined by nn.Module.
def mnist_ffnn():
    model = nn.Sequential(
        nn.Linear(784, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 80),
        nn.ReLU(),
        nn.Linear(80, 10)
    )
    return model

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# Models defined by nn.Sequential.
def mnist_conv():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

model = mnist_ffnn()
# Optionally, load the pretrained weights.
checkpoint = torch.load(os.path.join(os.path.dirname(__file__), "./pretrain/mnist_ffnn_10x80.pth"), map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)


### Step 2: Prepare dataset as usual
test_data = torchvision.datasets.MNIST("./data", train=False, download=False, transform=torchvision.transforms.ToTensor())
# For illustration, we only use 2 image from dataset
N = 10
n_classes = 10
# Adjust to model input shape!!!
image = test_data.data[:N].reshape(N, 784)
# Convert to float between 0. and 1.
image = image.to(torch.float32) / 255.0
true_label = test_data.targets[:N]

if torch.cuda.is_available():
    image = image.cuda()
    model = model.cuda()

### Step 3: wrap model with multipath_bp.
# The second parameter is for constructing the trace of the computational graph, and its content is not important.
lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
print('Running on', image.device)

### Step 4: Compute bounds using LiRPA given a perturbation
eps = 0.01
norm = float("inf")
ptb = PerturbationLpNorm(norm=norm, eps=eps)
image = BoundedTensor(image, ptb)
# Get model prediction as usual
pred = lirpa_model(image)
label = torch.argmax(pred, dim=1).cpu().detach().numpy()

### Step 5: Compute bounds for final output
'''
# for method in ['forward', 'IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)']:
for method in ['backward (CROWN)']:
    print("Bounding method:", method)
    lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0])

    for i in range(N):
        print("Image {} top-1 prediction {} ground-truth {}".format(i, label[i], true_label[i]))
        for j in range(n_classes):
            indicator = '(ground-truth)' if j == true_label[i] else ''
            print("f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}".format(
                j=j, l=lb[i][j].item(), u=ub[i][j].item(), ind=indicator))
    print()
'''

### An example for computing margin bounds.
# In compute_bounds() function you can pass in a specification matrix C, which is a final linear matrix applied to the last layer NN output.
# For example, if you are interested in the margin between the groundtruth class and another class, you can use C to specify the margin.
# This generally yields tighter bounds.
lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)

C = torch.zeros(size=(N, n_classes - 1, n_classes), device=image.device)
groundtruth = true_label.to(device=image.device).unsqueeze(1).unsqueeze(1)
C.scatter_(dim=2, index=groundtruth.repeat(1, n_classes - 1, 1), value=1.0)
target_labels = torch.arange(1, 10, device=image.device).repeat(N, 1, 1).transpose(1, 2)
target_labels = (target_labels + groundtruth) % n_classes
C.scatter_(dim=2, index=target_labels, value=-1.0)
# print('Computing bounds with a specification matrix:\n', C)

# for method in ['forward', 'IBP', 'IBP+backward (CROWN-IBP)']:
# for method in ['forward']:
for method in ['backward (CROWN)']:
    print("Bounding method:", method)
    if 'Optimized' in method:
        # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can increase verbosity to see per-iteration loss values.
        lirpa_model.set_bound_opts({'optimize_bound_args': {'ob_iteration': 20, 'ob_lr': 0.1, 'ob_verbose': 0}})
    lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], C=C)
    for i in range(N):
        print("Image {} top-1 prediction {} ground-truth {}".format(i, label[i], true_label[i]))
        print("lowest margin >= {l:10.5f}".format(l=torch.min(lb, dim=1)[0][i]))
