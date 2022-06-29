class Specification:

    # Optional datasets are 'MNIST', 'CIFAR-10' that PyTorch provides,
    # or a path to a user-defined dataset, like '../examples/vision/data/MNIST'
    dataset = 'MNIST'

    # `x_0 = idx` specifies which x_0 we choose for verification.
    # Note that when `batch > 0`, this parameter will be covered.
    x_0 = 0

    # `batch (> 0)` specifies the first batch inputs in the dataset for verification.
    # batch = 100
    batch = -1

    # `delta = (float)` specifies the perturbation size.
    delta = 0.001

    # `norm = float('inf')` specifies the perturbation type is infinity norm distance.
    # Other optional types are `norm = 1`, `norm = 2`.
    norm = float('inf')

    # `unsafe = untarget` sets all other incorrect labels as the unsafe labels,
    # `unsafe = [idx]` sets label list [idx] as unsafe labels.
    unsafe = 'untarget'
    # unsafe = [0, 1, 2]

