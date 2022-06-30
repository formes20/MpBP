import torch
import torch.nn as nn
import torchvision
import time

from multipath_bp.bound_general_multipath import BoundedModule
from multipath_bp import BoundedTensor
from multipath_bp.perturbations_multipath import PerturbationLpNorm

### Step 1: Define computational graph
# Models defined by nn.Sequential
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
checkpoint = torch.load("./mnist_ffnn.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

### Step 2: Prepare dataset as usual
test_data = torchvision.datasets.MNIST("./data", train=False, download=False, transform=torchvision.transforms.ToTensor())
N = 100
n_classes = 10
# Adjust to model input shape!!!
image = test_data.data[:N].reshape(N, 784)
# Convert to float between 0. and 1.
image = image.to(torch.float32) / 255.0
true_label = test_data.targets[:N]

if torch.cuda.is_available():
    image = image.cuda()
    model = model.cuda()

### Step 3: wrap model with MultipathBP.
# The second parameter is for constructing the trace of the computational graph, and its content is not important.
multipath_model = BoundedModule(model, torch.empty_like(image), device=image.device)
print('Running on', image.device)

### Step 4: Compute bounds using LiRPA given a perturbation
eps = 0.010
norm = float("inf")
ptb = PerturbationLpNorm(norm=norm, eps=eps)
image = BoundedTensor(image, ptb)
# Get model prediction as usual
pred = multipath_model(image)
label = torch.argmax(pred, dim=1).cpu().detach().numpy()

multipath_model = BoundedModule(model, torch.empty_like(image), device=image.device)

C = torch.zeros(size=(N, n_classes - 1, n_classes), device=image.device)
groundtruth = true_label.to(device=image.device).unsqueeze(1).unsqueeze(1)
C.scatter_(dim=2, index=groundtruth.repeat(1, n_classes - 1, 1), value=1.0)
target_labels = torch.arange(1, 10, device=image.device).repeat(N, 1, 1).transpose(1, 2)
target_labels = (target_labels + groundtruth) % n_classes
C.scatter_(dim=2, index=target_labels, value=-1.0)
# print('Computing bounds with a specification matrix:\n', C)

# for method in ['forward', 'forward+backward', 'IBP', 'IBP+backward (CROWN-IBP)']:
method ='backward'
print("Bounding method:", method)

time_begin = time.time()
lb, ub = multipath_model.compute_bounds(x=(image,), method=method.split()[0], C=C)
time_elapse = time.time() - time_begin

verified_robust = 0
for i in range(N):
    # print("Image {} top-1 prediction {} ground-truth {}".format(i, label[i], true_label[i]))
    # print("lowest margin >= {l:10.5f}".format(l=torch.min(lb, dim=1)[0][i]))
    if torch.min(lb, dim=1)[0][i] >= 0:
        verified_robust += 1
print('Verified robust number:', verified_robust)
print('Time elapse:', time_elapse)
# print(torch.cuda.memory_summary())
