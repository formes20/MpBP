import torch
import torch.nn as nn
import torchvision

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

# from auto_LiRPA.bound_general_zhengye import BoundedModule
# from auto_LiRPA import BoundedTensor
# from auto_LiRPA.perturbations_zhengye import PerturbationLpNorm

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
checkpoint = torch.load("../examples/vision/pretrain/mnist_ffnn_10x80.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

### Step 2: Prepare dataset as usual
test_data = torchvision.datasets.MNIST("../examples/vision/data", train=False, download=True, transform=torchvision.transforms.ToTensor())
n_classes = 10
for img_index in range(20):
    # Adjust to model input shape!!!
    image = test_data.data[img_index].reshape(1, 784)
    # Convert to float between 0. and 1.
    image = image.to(torch.float32) / 255.0
    true_label = test_data.targets[img_index]

    if torch.cuda.is_available():
        image = image.cuda()
        model = model.cuda()

    ### Step 3: wrap model with auto_LiRPA.
    # The second parameter is for constructing the trace of the computational graph, and its content is not important.
    lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
    print('Running on', image.device)

    ### Step 4: Compute bounds using LiRPA given a perturbation
    left, right = 0, 1000
    robust_radius = 0
    while left <= right:
        mid = int((left + right) / 2)
        norm = float("inf")
        ptb = PerturbationLpNorm(norm=norm, eps=mid/1000)
        image = BoundedTensor(image, ptb)
        # Get model prediction as usual
        pred = lirpa_model(image)
        label = torch.argmax(pred, dim=1).cpu().detach().numpy()

        lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)

        C = torch.zeros(size=(1, n_classes - 1, n_classes), device=image.device)
        groundtruth = true_label.to(device=image.device).unsqueeze(0).unsqueeze(1).unsqueeze(2)
        C.scatter_(dim=2, index=groundtruth.repeat(1, n_classes - 1, 1), value=1.0)
        target_labels = torch.arange(1, 10, device=image.device).repeat(1, 1, 1).transpose(1, 2)
        target_labels = (target_labels + groundtruth) % n_classes
        C.scatter_(dim=2, index=target_labels, value=-1.0)
        # print('Computing bounds with a specification matrix:\n', C)

        method = 'CROWN-Optimized'
        # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can increase verbosity to see per-iteration loss values.
        lirpa_model.set_bound_opts({'optimize_bound_args': {'ob_iteration': 20, 'ob_lr': 0.1, 'ob_verbose': 0}})
        lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], C=C)

        for i in range(1):
            # print("Image {} top-1 prediction {} ground-truth {}".format(i, label[i], true_label[i]))
            # print("lowest margin >= {l:10.5f}".format(l=torch.min(lb, dim=1)[0][i]))
            if torch.min(lb, dim=1)[0][i] >= 0:
                left = mid + 1
                robust_radius = mid / 1000
            else:
                right = mid - 1
    print('Verified robust radius:', robust_radius)
