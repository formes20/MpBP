import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def mnist_loaders(batch_size): 
    train_set = datasets.CIFAR10('/data/zhengye/LiRPA_new/examples/vision/data', train=True, download=False, transform=transforms.ToTensor())
    test_set = datasets.CIFAR10('/data/zhengye/LiRPA_new/examples/vision/data', train=False, download=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader


batch_size = 64
test_batch_size = 1000

train_loader, _ = mnist_loaders(batch_size)
_, test_loader = mnist_loaders(test_batch_size)


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
# print(model)
num_epochs = 70


def train(model, train_loader):
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(num_epochs):
        batch_loss = 0
        total_batches = 0

        for i, (images, labels) in enumerate(train_loader):
            # Reshape images to (batch_size, input_size)
            images = images.reshape(-1, 3, 32, 32)
            # print(images.shape)
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   

            total_batches += 1     
            batch_loss += loss.item()

        avg_loss_epoch = batch_loss / total_batches
        print('Epoch [{}/{}], Averge Loss:for epoch[{}, {:.4f}]'.format(epoch + 1, num_epochs, epoch + 1, avg_loss_epoch))

    torch.save(model.state_dict(), './cifar10_conv.pth')


def accuracy_test(model_path, test_loader):
    correct = 0
    total = 0

    model = cifar10_conv()
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, 3, 32, 32)
            # print(labels)
            outputs_test = model(images)
            _, predicted = torch.max(outputs_test.data, 1)
            # print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 1000 test images: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    # train(model, train_loader)
    accuracy_test('./cifar10_conv.pth', test_loader)
