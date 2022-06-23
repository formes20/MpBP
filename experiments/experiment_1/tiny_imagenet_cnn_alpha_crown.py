import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

from multipath_bp import BoundedModule, BoundedTensor
from multipath_bp.perturbations import PerturbationLpNorm

### Step 1: Define computational graph
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def tiny_imagenet_cnn():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 5, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 5, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(52 * 52 * 8, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 200)
    )
    return model

model = tiny_imagenet_cnn()
checkpoint = torch.load('./tiny_imagenet_cnn_v2.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

### Step 2: Prepare dataset as usual
# normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
test_data = torchvision.datasets.ImageFolder('./tiny-imagenet-200/val',
                                transform=transforms.Compose([
                                    # transforms.RandomHorizontalFlip(),
                                    transforms.CenterCrop(56),
                                    transforms.ToTensor(),
                                ]))
n_classes = 200

for i in range(2000):
    # Adjust to model input shape!!!
    image = test_data[i][0].reshape(-1, 3, 56, 56)
    # Convert to float between 0. and 1.
    true_label = torch.tensor(test_data.targets[i])

    # if torch.cuda.is_available():
    #     image = image.cuda()
    #     model = model.cuda()

    ### Step 3: wrap model with auto_LiRPA.
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
    target_labels = torch.arange(1, 200, device=image.device).repeat(1, 1, 1).transpose(1, 2)
    target_labels = (target_labels + groundtruth) % n_classes
    C.scatter_(dim=2, index=target_labels, value=-1.0)
    # print('Computing bounds with a specification matrix:\n', C)

    method = 'CROWN-Optimized'
    if 'Optimized' in method:
        lirpa_model.set_bound_opts({'optimize_bound_args': {'ob_iteration': 20, 'ob_lr': 0.1, 'ob_verbose': 0}})
    lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], C=C)
    if torch.min(lb, dim=1)[0][0] >= 0:
        print('Verified')
    else:
        print('Unverified')
