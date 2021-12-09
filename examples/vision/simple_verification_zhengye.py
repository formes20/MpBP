"""
A simple example for bounding neural network outputs under input perturbations.

This example serves as a skeleton for robustness verification of neural networks.
"""
import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from auto_LiRPA.bound_general_zhengye import BoundedModule
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations_zhengye import PerturbationLpNorm

# Step 1: Define computational graph by implementing forward()
class MNIST_FFNN(nn.Module):
    def __init__(self):
        super(MNIST_FFNN, self).__init__()
        self.linear1 = torch.nn.Linear(784, 100)
        self.linear2 = torch.nn.Linear(100, 100)
        self.linear3 = torch.nn.Linear(100, 100)
        self.linear4 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

model = MNIST_FFNN()
# Optionally, load the pretrained weights.
checkpoint = torch.load(os.path.join(os.path.dirname(__file__), "./pretrain/mnist_ffnn.pth"), map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)


# Step 2: Prepare dataset as usual
test_data = torchvision.datasets.MNIST("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
# For illustration, we only use 2 image from dataset
N = 2
n_classes = 10
# Adjust to model input shape.
image = test_data.data[:N].view(N, 784)
true_label = test_data.targets[:N]
# Convert to float.
image = image.to(torch.float32) / 255.0
if torch.cuda.is_available():
    image = image.cuda()
    model = model.cuda()

# Step 3: wrap model with auto_LiRPA.
# The second parameter is for constructing the trace of the computational graph, and its content is not important.
lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
print('Running on', image.device)

# Step 4: Compute bounds using LiRPA given a perturbation
eps = 0.1
norm = float("inf")
ptb = PerturbationLpNorm(norm=norm, eps=eps)
image = BoundedTensor(image, ptb)
# Get model prediction as usual
pred = lirpa_model(image)
label = torch.argmax(pred, dim=1).cpu().detach().numpy()

# Step 5: Compute bounds for final output
# for method in ['forward', 'IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)', ]:
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


