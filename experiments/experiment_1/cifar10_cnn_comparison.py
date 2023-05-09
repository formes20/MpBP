import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

from multipath_bp import BoundedModule, BoundedTensor
from multipath_bp.perturbations import PerturbationLpNorm

# from multipath_bp.bound_general_multipath import BoundedModule
# from multipath_bp import BoundedTensor
# from multipath_bp.perturbations_multipath import PerturbationLpNorm

### Step 1: Define computational graph
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def cifar10_conv():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 5, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, 5, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(28 * 28 * 16, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

model = cifar10_conv()
checkpoint = torch.load('./cifar10_conv.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

### Step 2: Prepare dataset as usual
# normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
test_data = torchvision.datasets.CIFAR10("../../examples/vision/data", train=False, download=False,
                                         transform=transforms.Compose([transforms.ToTensor()]))
n_classes = 10

for i in range(15):
    # Adjust to model input shape!!!
    image = test_data[i][0].reshape(-1, 3, 32, 32)
    # Convert to float between 0. and 1.
    true_label = torch.tensor(test_data.targets[i])

    if torch.cuda.is_available():
        image = image.cuda()
        model = model.cuda()

    ### Step 3: wrap model with MultipathBP.
    # The second parameter is for constructing the trace of the computational graph, and its content is not important.
    lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
    # print('Running on', image.device)

    ### Step 4: Compute bounds using LiRPA given a perturbation
    eps = 0.0014
    norm = float("inf")
    ptb = PerturbationLpNorm(norm=norm, eps=eps)
    image = BoundedTensor(image, ptb)
    # Get model prediction as usual
    pred = lirpa_model(image)
    label = torch.argmax(pred, dim=1).cpu().detach().numpy()

    if label.item() != true_label.item():
        continue

    C = torch.zeros(size=(1, n_classes - 1, n_classes), device=image.device)
    groundtruth = true_label.to(device=image.device).unsqueeze(0).unsqueeze(1).unsqueeze(2)
    C.scatter_(dim=2, index=groundtruth.repeat(1, n_classes - 1, 1), value=1.0)
    target_labels = torch.arange(1, 10, device=image.device).repeat(1, 1, 1).transpose(1, 2)
    target_labels = (target_labels + groundtruth) % n_classes
    C.scatter_(dim=2, index=target_labels, value=-1.0)
    # print('Computing bounds with a specification matrix:\n', C)

    begin_time = time.time()

    method = 'forward+backward'
    lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], C=C)
    if torch.min(lb, dim=1)[0][0] >= 0:
        print('Verified')
    else:
        print('Unverified')

    end_time = time.time()
    print("elapsed_time:", end_time - begin_time)
    # print(torch.cuda.memory_summary())

