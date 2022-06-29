import os
import sys
pre_path = os.path.abspath('../')
sys.path.append(pre_path)

import torch
import torch.nn as nn
import torchvision
import argparse
import time

from multipath_bp.bound_general_multipath import BoundedModule
from multipath_bp import BoundedTensor
from multipath_bp.perturbations_multipath import PerturbationLpNorm

from specfication import Specification

parser = argparse.ArgumentParser(description='Multipath Usage List')
parser.add_argument('--net', type=str, default='./mnist_ffnn.pth', help='give the DNN model to be verified.')
parser.add_argument('--spec', type=str, default='./specificaiton.py', help='give a specificaiton file.')
parser.add_argument('--path', type=int, default=4, help='give the number of propagation paths.')
parser.add_argument('--bp', type=str, default='bbp', help='choose a bound propagation method.')
parser.add_argument('--verbose', default=False, help='output verbose information (default no).')

args = parser.parse_args()

### Step 1: load model and dataset
checkpoint = torch.load(args.net, map_location=torch.device('cpu'))
if Specification.dataset == 'MNIST':
    test_data = torchvision.datasets.MNIST("../examples/vision/data", train=False, download=False, transform=torchvision.transforms.ToTensor())
elif Specification.dataset == 'CIFAR-10':
    test_data = torchvision.datasets.CIFAR10("../examples/vision/data", train=False, download=False, transform=torchvision.transforms.ToTensor())
else:
    test_data = torchvision.datasets.ImageFolder(Specification.dataset, transform=torchvision.transforms.ToTensor())

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
model.load_state_dict(checkpoint)

if Specification.batch > 0:
    N = Specification.batch
    n_classes = 10
    image = test_data.data[:N].reshape(N, 784)
    image = image.to(torch.float32) / 255.0
    true_label = test_data.targets[:N]

else:
    N = 1
    n_classes = 10
    image = test_data.data[Specification.x_0:Specification.x_0 + 1].reshape(1, 784)
    image = image.to(torch.float32) / 255.0
    true_label = test_data.targets[Specification.x_0:Specification.x_0 + 1]

if torch.cuda.is_available():
    image = image.cuda()
    model = model.cuda()
print('Running on', image.device)

if args.path == 1:
    from multipath_bp.bound_general import BoundedModule
    from multipath_bp.perturbations import PerturbationLpNorm

### Step 2: wrap model with MultipathBP
multipath_bp = BoundedModule(model, torch.empty_like(image), device=image.device)

### Step 3: set specification
delta = Specification.delta
norm = Specification.norm
ptb = PerturbationLpNorm(norm=norm, eps=delta)
image = BoundedTensor(image, ptb)
pred = multipath_bp(image)
label = torch.argmax(pred, dim=1).cpu().detach().numpy()
multipath_bp = BoundedModule(model, torch.empty_like(image), device=image.device)

### Step 4: compute reachable region
method_dict = {'fbp': 'forward', 'fbbp': 'forward+backward', 'bbp': 'backward', 'ibp': 'IBP'}
method = method_dict[args.bp]
print("Bounding method:", method)

if bool(args.verbose) == True:
    print('')
    print('=============== Reachable region ===============')
    lb, ub = multipath_bp.compute_bounds(x=(image,), method=method.split()[0])
    for i in range(N):
        print("Image {} top-1 prediction {} ground-truth {}".format(i, label[i], true_label[i]))
        for j in range(n_classes):
            indicator = '(ground-truth)' if j == true_label[i] else ''
            print("f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}".format(
                j=j, l=lb[i][j].item(), u=ub[i][j].item(), ind=indicator))

### specification matrix
if Specification.unsafe != 'untarget':
    unsafe_list = Specification.unsafe
    C = torch.zeros(size=(N, len(unsafe_list), n_classes), device=image.device)
    groundtruth = true_label.to(device=image.device).unsqueeze(1).unsqueeze(1)
    C.scatter_(dim=2, index=groundtruth.repeat(1, len(unsafe_list), 1), value=1.0)
    target_labels = torch.tensor(unsafe_list, device=image.device).repeat(N, 1, 1).transpose(1, 2)
    C.scatter_(dim=2, index=target_labels, value=-1.0)
    # print('Computing bounds with a specification matrix:\n', C)

else:
    C = torch.zeros(size=(N, n_classes - 1, n_classes), device=image.device)
    groundtruth = true_label.to(device=image.device).unsqueeze(1).unsqueeze(1)
    C.scatter_(dim=2, index=groundtruth.repeat(1, n_classes - 1, 1), value=1.0)
    target_labels = torch.arange(1, 10, device=image.device).repeat(N, 1, 1).transpose(1, 2)
    target_labels = (target_labels + groundtruth) % n_classes
    C.scatter_(dim=2, index=target_labels, value=-1.0)
    # print('Computing bounds with a specification matrix:\n', C)

time_begin = time.time()
lb, ub = multipath_bp.compute_bounds(x=(image,), method=method.split()[0], C=C)
time_elapse = time.time() - time_begin

if bool(args.verbose) == True:
    print()
    print('================== Robustness ==================')
    for i in range(N):
        if torch.min(lb, dim=1)[0][i] >= 0:
                print('Image {}: verified'.format(i))
        else:
            print('Image {}: unknown'.format(i))
else:
    verified_robust = 0
    for i in range(N):
        if torch.min(lb, dim=1)[0][i] >= 0:
            verified_robust += 1
    print('================== Statistics ==================')
    print('Verified number out of {i:>3}: {j}'.format(i=N, j=verified_robust))

print('Time elapsed:', time_elapse)
