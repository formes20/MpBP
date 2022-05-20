import torch
import torch.nn as nn
import torchvision
import time

# from auto_LiRPA import BoundedModule, BoundedTensor
# from auto_LiRPA.perturbations import PerturbationLpNorm

from auto_LiRPA.bound_general_zhengye import BoundedModule
from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations_zhengye import PerturbationLpNorm

### Step 1: Define computational graph
# Models defined by nn.Sequential
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def mnist_cnn():
    model = nn.Sequential(
        nn.Conv2d(1, 8, 5, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 5, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(24 * 24 * 8, 80),
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

model = mnist_cnn()
checkpoint = torch.load("./mnist_cnn.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

### Step 2: Prepare dataset as usual
test_data = torchvision.datasets.MNIST("../../examples/vision/data", train=False, download=False, transform=torchvision.transforms.ToTensor())
N = 1
n_classes = 10

for i in range(100):
    # Adjust to model input shape!!!
    image = test_data[i][0].reshape(-1, 1, 28, 28)
    # Convert to float between 0. and 1.
    true_label = torch.tensor(test_data.targets[i])

    if torch.cuda.is_available():
        image = image.cuda()
        model = model.cuda()

    ### Step 3: wrap model with auto_LiRPA.
    # The second parameter is for constructing the trace of the computational graph, and its content is not important.
    lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
    # print('Running on', image.device)

    ### Step 4: Compute bounds using LiRPA given a perturbation
    eps = 0.0025
    norm = float("inf")
    ptb = PerturbationLpNorm(norm=norm, eps=eps)
    image = BoundedTensor(image, ptb)
    # Get model prediction as usual
    pred = lirpa_model(image)
    label = torch.argmax(pred, dim=1).cpu().detach().numpy()

    if label.item() != true_label.item():
        continue

    begin_time = time.time()
    C = torch.zeros(size=(1, n_classes - 1, n_classes), device=image.device)
    groundtruth = true_label.to(device=image.device).unsqueeze(0).unsqueeze(1).unsqueeze(2)
    C.scatter_(dim=2, index=groundtruth.repeat(1, n_classes - 1, 1), value=1.0)
    target_labels = torch.arange(1, 10, device=image.device).repeat(1, 1, 1).transpose(1, 2)
    target_labels = (target_labels + groundtruth) % n_classes
    C.scatter_(dim=2, index=target_labels, value=-1.0)
    # print('Computing bounds with a specification matrix:\n', C)

    method = 'forward+backward'
    lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], C=C)
    if torch.min(lb, dim=1)[0][0] >= 0:
        print('Verified')
    else:
        print('Unverified')

    end_time = time.time()
    print("elapsed_time:", end_time - begin_time)
