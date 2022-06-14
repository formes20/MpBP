import torch
import torch.nn as nn
import torchvision
import time

from auto_LiRPA.bound_general_multipath import BoundedModule
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations_multipath import PerturbationLpNorm

### Step 1: load model and dataset

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

model = mnist_ffnn()
checkpoint = torch.load("./mnist_ffnn_10x80.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

test_data = torchvision.datasets.MNIST("../../examples/vision/data", train=False, download=False, transform=torchvision.transforms.ToTensor())
N = 100
n_classes = 10

image = test_data.data[:N].reshape(N, 784)
image = image.to(torch.float32) / 255.0
true_label = test_data.targets[:N]

if torch.cuda.is_available():
    image = image.cuda()
    model = model.cuda()


### Step 2: wrap model with MultipathBP

lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
print('Running on', image.device)

### Step 3: Compute bounds using LiRPA given a perturbation
eps = 0.0010
norm = float("inf")
ptb = PerturbationLpNorm(norm=norm, eps=eps)
image = BoundedTensor(image, ptb)
# Get model prediction as usual
pred = lirpa_model(image)
label = torch.argmax(pred, dim=1).cpu().detach().numpy()

lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)

C = torch.zeros(size=(N, n_classes - 1, n_classes), device=image.device)
groundtruth = true_label.to(device=image.device).unsqueeze(1).unsqueeze(1)
C.scatter_(dim=2, index=groundtruth.repeat(1, n_classes - 1, 1), value=1.0)
target_labels = torch.arange(1, 10, device=image.device).repeat(N, 1, 1).transpose(1, 2)
target_labels = (target_labels + groundtruth) % n_classes
C.scatter_(dim=2, index=target_labels, value=-1.0)
# print('Computing bounds with a specification matrix:\n', C)

for method in ['forward', 'forward+backward', 'backward', 'IBP', 'IBP+backward']:
    print("Bounding method:", method)
    time_begin = time.time()

    lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], C=C)
    verified_robust = 0
    for i in range(N):
        if torch.min(lb, dim=1)[0][i] >= 0:
            verified_robust += 1
    print('Verified robust number:', verified_robust)
    time_elapse = time.time() - time_begin
    print('Time elapse:', time_elapse)
