import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def tiny_imagenet_loaders(batch_size):
    # normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
    train_set = datasets.ImageFolder('/data/zhengye/LiRPA_new/experiments/experiment_2/tiny-imagenet-200/train',
                                     transform=transforms.Compose([
                                          transforms.RandomHorizontalFlip(),
                                          transforms.CenterCrop(56),
                                          transforms.ToTensor()
                                      ]))
    test_set = datasets.ImageFolder('/data/zhengye/LiRPA_new/experiments/experiment_2/tiny-imagenet-200/val',
                                transform=transforms.Compose([
                                    # transforms.RandomHorizontalFlip(),
                                    transforms.CenterCrop(56),
                                    transforms.ToTensor()
                                ]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader


batch_size = 64
test_batch_size = 1000

train_loader, _ = tiny_imagenet_loaders(batch_size)
_, test_loader = tiny_imagenet_loaders(test_batch_size)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def tiny_imagenet_cnn():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(56 * 56 * 3, 200),
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


model = tiny_imagenet_cnn().cuda()
# print(model)
num_epochs = 100

def train(model, train_loader):
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(num_epochs):
        batch_loss = 0
        total_batches = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()
            # Reshape images to (batch_size, input_size)
            images = images.reshape(-1, 3, 56, 56)
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

    torch.save(model.state_dict(), './tiny_imagenet_cnn.pth')


def accuracy_test(model_path, test_loader):
    correct = 0
    total = 0

    model = tiny_imagenet_cnn()
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, 3, 56, 56)
            # print(labels)
            outputs_test = model(images)
            _, predicted = torch.max(outputs_test.data, 1)
            # print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 1000 test images: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    train(model, train_loader)
    accuracy_test('./tiny_imagenet_cnn.pth', test_loader)
